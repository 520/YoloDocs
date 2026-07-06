"""
YOLOv8 Keras — Training loop.

Train  : box_loss + cls_loss + dfl_loss only  (no NMS, no mAP)
Val    : losses + DetectionEvaluator → mAP@0.5, mAP@0.5:0.95,
         Recall (keras_cv), Precision + F1 (from TP/FP counts),
         per-class AP50, P, R, F1

Checkpointing : best.weights.h5  (best mAP50-95 OR lowest val_loss)
                last.weights.h5  (every epoch, overwritten)
Logging       : console table  +  results.csv
"""

import os
import csv
import math
import time
import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras

from losses.yolov8_loss import YOLOv8Loss
from losses.tal_assigner import make_anchors, dfl_decode, dist2bbox
from training.metrics import DetectionEvaluator


# ============================================================================
# LR / momentum schedule
# ============================================================================

def get_lr(epoch, epochs, warmup_epochs, lr0, lrf, cos_lr=True):
    if epoch < warmup_epochs:
        return lr0 * (epoch + 1) / warmup_epochs
    p = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    if cos_lr:
        return lr0 * (lrf + 0.5 * (1 - lrf) * (1 + math.cos(math.pi * p)))
    else:
        # Linear decay — Ultralytics default (cos_lr=False)
        return lr0 * (lrf + (1 - lrf) * max(1 - p, 0))


def get_momentum(epoch, warmup_epochs, warmup_mom, final_mom):
    if epoch < warmup_epochs:
        return warmup_mom + (epoch + 1) / warmup_epochs * (final_mom - warmup_mom)
    return final_mom


# ============================================================================
# EMA
# ============================================================================

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = [
            tf.Variable(tf.cast(w, tf.float32), trainable=False)
            for w in model.trainable_variables
        ]

    @tf.function
    def update(self, model):
        d = self.decay
        for s, v in zip(self.shadow, model.trainable_variables):
            s.assign(d * s + (1 - d) * tf.cast(v, tf.float32))

    def apply(self, model):
        orig = [v.numpy() for v in model.trainable_variables]
        for s, v in zip(self.shadow, model.trainable_variables):
            v.assign(tf.cast(s, v.dtype))
        return orig

    def restore(self, model, orig):
        for v, o in zip(model.trainable_variables, orig):
            v.assign(o)


# ============================================================================
# Decode predictions → boxes + scores
# ============================================================================

def decode_preds(raw_outputs, reg_max=16, strides=(8, 16, 32), imgsz=640):
    all_raw   = tf.concat(raw_outputs, axis=1)
    pred_dist = all_raw[..., :4 * reg_max]
    pred_cls  = all_raw[..., 4 * reg_max:]

    shapes, strides_used = [], []
    for p, s in zip(raw_outputs, strides):
        n    = p.shape[1] or tf.shape(p)[1]
        side = int(round(math.sqrt(int(n))))
        shapes.append((side, side)); strides_used.append(s)

    anc, stride_t = make_anchors(shapes, strides_used, offset=0.5)
    boxes  = dist2bbox(dfl_decode(pred_dist, reg_max) * stride_t[tf.newaxis],
                       (anc * stride_t)[tf.newaxis])
    # Clip to image bounds — matches Ultralytics post-decode clipping
    boxes  = tf.clip_by_value(boxes, 0.0, float(imgsz))
    scores = tf.sigmoid(pred_cls)
    return boxes.numpy(), scores.numpy()


# ============================================================================
# Trainer
# ============================================================================

class YOLOv8Trainer:
    """
    Parameters
    ----------
    model        : YOLOv8 (already built with dummy forward pass)
    train_loader : COCOLoader  (augment=True, mosaic=True)
    val_loader   : COCOLoader  (augment=False, mosaic=False)
    cfg          : TRAIN_CFG dict
    save_dir     : output directory for weights + CSV
    nc           : number of classes
    class_names  : optional list[str] for per-class console table
    """

    def __init__(self, model, train_loader, val_loader,
                 cfg=None, save_dir="runs/train",
                 nc=80, class_names=None, resume=False):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = dict(cfg or {})
        self.save_dir     = save_dir
        self.nc           = nc
        self.class_names  = class_names
        self.strides      = [8, 16, 32]
        os.makedirs(save_dir, exist_ok=True)

        if self.cfg["amp"]:
            keras.mixed_precision.set_global_policy("mixed_float16")

        # Loss
        self.loss_fn = YOLOv8Loss(
            nc=nc, reg_max=self.cfg["reg_max"], strides=self.strides,
            box_gain=self.cfg["box"], cls_gain=self.cfg["cls"],
            dfl_gain=self.cfg["dfl"],
            tal_topk=self.cfg["tal_topk"],
            tal_alpha=self.cfg["tal_alpha"],
            tal_beta=self.cfg["tal_beta"],
            tal_min_anchor_size=self.cfg.get("tal_min_anchor_size", 8.0),
            tal_expand_anchor_size=self.cfg.get("tal_expand_anchor_size", 16.0),
        )

        # Weight decay — applied manually in _train_step.
        # Ultralytics: WD on conv/linear kernels only — not on biases or BN params.
        # Adam/SGD : coupled WD   → grad += wd * param  (WD is adapted by Adam's 2nd moment)
        # AdamW    : decoupled WD → param *= (1 - lr * wd)  AFTER gradient step (true AdamW)
        # tf.keras.optimizers.legacy.AdamW does NOT exist in TF 2.14; we simulate AdamW
        # by using legacy Adam + a post-step param update, which avoids the retracing
        # issue triggered by the new-style tf.keras.optimizers.AdamW inside @tf.function.
        self._wd = self.cfg["weight_decay"] * self.cfg["batch"] / self.cfg["nbs"]
        # OLD: self._wd_apply = ["kernel" in v.name ... for v in model.trainable_variables]
        # Removed: stale pre-computed list breaks when freeze changes trainable_variables.
        # WD condition is now computed inline in _train_step using v.name (a Python constant
        # at @tf.function trace time), so it stays correct after freeze/unfreeze retrace.

        # Optimizer — use legacy optimizers to avoid Keras 2 new-style optimizer
        # which internally uses _update_step_xla and causes repeated retracing.
        opt_name = self.cfg.get("optimizer", "SGD").upper()
        self._is_adam  = opt_name in ("ADAM", "ADAMW")
        self._is_adamw = opt_name == "ADAMW"
        if self._is_adam:
            self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.cfg["lr0"],
                beta_1=self.cfg["momentum"],
            )
        else:
            self.optimizer = tf.keras.optimizers.legacy.SGD(
                learning_rate=self.cfg["lr0"],
                momentum=self.cfg["warmup_momentum"],
                nesterov=self.cfg["nesterov"],
            )
        if self.cfg["amp"]:
            self.optimizer = keras.mixed_precision.LossScaleOptimizer(
                self.optimizer)
        # Cache the inner (unwrapped) optimizer for LR/momentum access inside @tf.function
        self._base_opt = (self.optimizer.inner_optimizer
                          if self.cfg["amp"] else self.optimizer)

        self.ema = ModelEMA(model, decay=self.cfg["ema_decay"])

        # Metrics evaluator (val only)
        self.evaluator = DetectionEvaluator(
            nc=nc,
            imgsz=self.cfg["imgsz"],
            conf_thres=0.001,
            iou_thres=0.7,
            max_det=300,
        )

        # CSV
        self.csv_path = os.path.join(save_dir, "results.csv")
        if not resume:
            self._csv_init()

        self.best_map5095  = 0.0
        self.best_val_loss = float("inf")

    # ── CSV ────────────────────────────────────────────────────────────

    def _csv_init(self):
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch",
                "train/box_loss", "train/cls_loss", "train/dfl_loss",
                "val/box_loss",   "val/cls_loss",   "val/dfl_loss",
                "metrics/precision", "metrics/recall", "metrics/F1",
                "metrics/mAP50",     "metrics/mAP50-95",
                "lr",
            ])

    def _csv_append(self, epoch, tl, vl, m, lr):
        row = [epoch + 1,
               tl["box"], tl["cls"], tl["dfl"],
               vl["box"], vl["cls"], vl["dfl"],
               m["precision"], m["recall"], m["f1"],
               m["map50"],     m["map5095"], lr]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [f"{v:.6f}" if isinstance(v, float) else v for v in row])

    # ── Freeze / unfreeze backbone ─────────────────────────────────────

    def _freeze_backbone_layers(self):
        """Freeze all Conv2D (no-bias) and BN layers — backbone + neck + head intermediates.

        Detection head final projections (Conv2D with bias) stay trainable so the
        cls head can adapt from scratch while pretrained features are protected.
        After freezing, EMA is re-initialized to match the new trainable_variables list.
        """
        # OLD: no freeze logic
        n_frozen = 0
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                n_frozen += 1
            elif isinstance(layer, tf.keras.layers.Conv2D) and not layer.use_bias:
                layer.trainable = False
                n_frozen += 1
        # Re-init EMA so its shadow list matches the new (shorter) trainable_variables.
        self.ema = ModelEMA(self.model, decay=self.cfg["ema_decay"])
        print(f"\n── Froze {n_frozen} backbone/neck/head-intermediate layers "
              f"for first {self.cfg.get('freeze_epochs', 0)} epochs. "
              f"EMA re-initialized. ──\n")

    def _unfreeze_all_layers(self):
        """Restore all layers to trainable and re-initialize EMA."""
        # OLD: no unfreeze logic
        for layer in self.model.layers:
            layer.trainable = True
        # Re-init EMA so its shadow list matches the restored (full) trainable_variables.
        self.ema = ModelEMA(self.model, decay=self.cfg["ema_decay"])
        print(f"\n── Unfroze all layers. EMA re-initialized. ──\n")

    # ── Single train step ───────────────────────────────────────────────

    @tf.function
    def _train_step(self, batch):
        with tf.GradientTape() as tape:
            preds    = self.model(batch["images"], training=True)
            loss, ld = self.loss_fn(preds, batch)
            if self.cfg["amp"]:
                scaled = self.optimizer.get_scaled_loss(loss)
            else:
                scaled = loss
        grads = tape.gradient(scaled, self.model.trainable_variables)
        if self.cfg["amp"]:
            grads = self.optimizer.get_unscaled_gradients(grads)

        # Weight decay — kernels only, not biases or BN params (Ultralytics convention).
        # Adam/SGD: coupled WD — add wd*param to gradient before optimizer update.
        # AdamW   : skip here; decoupled WD is applied directly to params after update.
        # OLD: iterated zip(..., self._wd_apply) where _wd_apply is a pre-computed list
        #      that goes stale after freeze/unfreeze changes model.trainable_variables.
        # NEW: inline name check — v.name is a Python string constant at @tf.function
        #      trace time, so it is re-evaluated correctly on each retrace after freeze.
        if self._wd > 0 and not self._is_adamw:
            grads = [
                g + tf.cast(self._wd * tf.cast(v, tf.float32), g.dtype)
                if ("kernel" in v.name and "batch_normalization" not in v.name) else g
                for g, v in zip(grads, self.model.trainable_variables)
            ]

        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # AdamW: decoupled weight decay — w ← w · (1 − lr · wd), kernels only.
        # Applied after gradient step so Adam's m/v estimates are NOT affected by WD.
        if self._is_adamw and self._wd > 0:
            lr = tf.cast(self._base_opt.learning_rate, tf.float32)
            for v in self.model.trainable_variables:
                if "kernel" in v.name and "batch_normalization" not in v.name:
                    v.assign(tf.cast(v, tf.float32) * (1.0 - lr * self._wd))

        return loss, ld

    # ── Validation ─────────────────────────────────────────────────────

    def _validate(self):
        """
        Runs on EMA weights (caller swaps/restores).
        Returns (val_losses dict, metrics dict).
        """
        self.evaluator.reset()
        vbox = vcls = vdfl = vn = 0.0

        for batch in self.val_loader.build_dataset():
            # losses
            preds    = self.model(batch["images"], training=False)
            _, ld    = self.loss_fn(preds, batch)
            vbox += ld["box"].numpy(); vcls += ld["cls"].numpy()
            vdfl += ld["dfl"].numpy(); vn   += 1

            # decode for metrics
            boxes_b, scores_b = decode_preds(
                preds, reg_max=self.cfg["reg_max"], strides=self.strides)


            # ── one-time debug print (first batch, first val run only) ──
            if vn == 1 and not getattr(self, '_debug_printed', False):
                self._debug_printed = True
                b0_boxes  = boxes_b[0]
                b0_scores = scores_b[0]
                max_scores = b0_scores.max(axis=-1)
                print("\n[DEBUG] decode_preds — first val batch, image 0:")
                print(f"  pred[0] shape  : {preds[0].shape}")
                print(f"  boxes  range   : {b0_boxes.min():.3f} → {b0_boxes.max():.3f}  (expected 0-640)")
                print(f"  scores range   : {b0_scores.min():.6f} → {b0_scores.max():.6f}  (expected 0-1)")
                print(f"  scores > 0.001 : {(max_scores > 0.001).sum()} / {len(max_scores)} anchors")
                print(f"  scores > 0.1   : {(max_scores > 0.1).sum()} / {len(max_scores)} anchors")
                print(f"  scores > 0.5   : {(max_scores > 0.5).sum()} / {len(max_scores)} anchors")
                raw = preds[0].numpy()
                print(f"  raw[0,:,:64] (box distri): {raw[0,:,:64].min():.3f} → {raw[0,:,:64].max():.3f}")
                print(f"  raw[0,:,64:] (cls logits): {raw[0,:,64:].min():.3f} → {raw[0,:,64:].max():.3f}")
                # Top-10 detections for image 0
                top_idx = max_scores.argsort()[::-1][:10]
                print(f"  Top-10 detections (image 0):")
                for rank, ai in enumerate(top_idx):
                    cls_idx = b0_scores[ai].argmax()
                    name = self.class_names[cls_idx] if self.class_names else f"cls{cls_idx}"
                    x1,y1,x2,y2 = b0_boxes[ai]
                    print(f"    [{rank+1}] {name:20s}  conf={max_scores[ai]:.4f}"
                          f"  box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
                print()
                
            # feed evaluator
            self.evaluator.update(
                boxes_b, scores_b,
                batch["gt_bboxes"], batch["gt_labels"], batch["mask_gt"],
            )

        vl = {k: v / max(vn, 1)
              for k, v in [("box",vbox),("cls",vcls),("dfl",vdfl)]}
        vl["total"] = vl["box"] + vl["cls"] + vl["dfl"]

        metrics = self.evaluator.compute()
        return vl, metrics

    # ── Main training loop ──────────────────────────────────────────────

    def train(self, start_epoch=0):
        cfg      = self.cfg
        epochs   = cfg["epochs"]
        close_ep = epochs - cfg["close_mosaic"]
        lr0        = cfg["lr0"]
        lrf        = cfg["lrf"]
        cos_lr     = cfg.get("cos_lr", True)
        warmup_ep  = cfg["warmup_epochs"]
        warmup_mom = cfg["warmup_momentum"]
        final_mom  = cfg["momentum"]
        # OLD: no freeze logic
        # NEW: freeze backbone for first freeze_epochs epochs when using pretrained weights
        freeze_ep = cfg.get("freeze_epochs", 0)
        if freeze_ep > 0 and start_epoch == 0:
            self._freeze_backbone_layers()

        # Step-level warmup — matches Ultralytics exactly:
        # LR ramps 0 → lr0, momentum ramps warmup_mom → momentum, over warmup_iters steps.
        steps_per_ep = self.train_loader.steps_per_epoch()
        warmup_iters = max(round(warmup_ep * steps_per_ep), 100)
        global_step  = start_epoch * steps_per_ep
        base = self.optimizer.inner_optimizer if cfg["amp"] else self.optimizer

        sched_name = "cosine" if cos_lr else "linear"
        print(f"\n  schedule={sched_name}  warmup_iters={warmup_iters}  steps_per_ep≈{steps_per_ep}\n")
        print(f"\n{'Ep':>4}  "
              f"{'box':>8} {'cls':>8} {'cls_w':>7} {'dfl':>8}  │  "
              f"{'P':>6} {'R':>6} {'F1':>6} "
              f"{'mAP50':>8} {'mAP50-95':>10}  "
              f"{'lr':>9}")
        print("─" * 97)

        for epoch in range(start_epoch, epochs):
            t0 = time.time()

            # OLD: no freeze/unfreeze logic in loop
            # NEW: unfreeze backbone at freeze_ep boundary (EMA re-initialized inside)
            if freeze_ep > 0 and epoch == freeze_ep:
                self._unfreeze_all_layers()

            # close-mosaic
            if epoch == close_ep:
                print(f"\n── Epoch {epoch+1}: disabling mosaic ──\n")
                self.train_loader.mosaic = False

            # Epoch-level LR (cosine or linear) — applied only after warmup is complete.
            # During warmup it is overridden per-step in the inner loop below.
            if global_step >= warmup_iters:
                base.learning_rate = get_lr(epoch, epochs, warmup_ep, lr0, lrf, cos_lr)
                if not self._is_adam:
                    base.momentum = final_mom

            # ── train ───────────────────────────────────────────────
            tbox = tcls = tdfl = tn = 0.0
            t_ep = time.time()
            for batch in self.train_loader.build_dataset():
                global_step += 1

                # Step-level warmup: linear ramp 0→lr0.
                # Momentum warmup only applies to SGD; Adam's beta_1 stays fixed.
                if global_step <= warmup_iters:
                    frac = global_step / warmup_iters
                    base.learning_rate = lr0 * frac
                    if not self._is_adam:
                        base.momentum = warmup_mom + frac * (final_mom - warmup_mom)

                if tn == 0:
                    print(f"  Ep {epoch+1}: @tf.function tracing first step …",
                          flush=True)
                _, ld = self._train_step(batch)
                self.ema.update(self.model)
                tbox += ld["box"].numpy()
                tcls += ld["cls"].numpy()
                tdfl += ld["dfl"].numpy()
                tn   += 1
                if tn == 1:
                    dt1 = time.time() - t_ep
                    print(f"  Ep {epoch+1}: step 1 in {dt1:.1f}s "
                          f"(includes @tf.function trace)", flush=True)
                elif tn % 100 == 0:
                    elapsed = time.time() - t_ep
                    ms_per  = elapsed / tn * 1000
                    cls_raw = tcls / tn
                    cls_w   = cls_raw * self.loss_fn.cls_gain
                    print(f"  Ep {epoch+1} [{tn:>5}]  "
                          f"box {tbox/tn:.3f}  cls {cls_raw:.3f}/{cls_w:.3f}  "
                          f"dfl {tdfl/tn:.3f}  {ms_per:.0f}ms/step",
                          flush=True)

            tl  = dict(box=tbox/max(tn,1), cls=tcls/max(tn,1), dfl=tdfl/max(tn,1))
            lr  = float(base.learning_rate)   # actual LR at epoch end

            # ── validate with EMA weights ───────────────────────────
            orig    = self.ema.apply(self.model)
            vl, met = self._validate()

            # ── checkpoint (EMA weights still active → best = EMA) ──
            improved = (met["map5095"] > self.best_map5095 or
                        vl["total"]    < self.best_val_loss)
            if improved:
                self.best_map5095  = max(met["map5095"], self.best_map5095)
                self.best_val_loss = min(vl["total"],    self.best_val_loss)
                self.model.save_weights(
                    os.path.join(self.save_dir, "best.weights.h5"))

            self.ema.restore(self.model, orig)

            self.model.save_weights(
                os.path.join(self.save_dir, "last.weights.h5"))

            # ── log ─────────────────────────────────────────────────
            self._csv_append(epoch, tl, vl, met, lr)
            self._print_row(epoch, epochs, tl, vl, met, lr,
                            time.time()-t0, improved)

            if (epoch + 1) % 10 == 0:
                self._print_per_class(met)

        print(f"\nDone.  Best mAP50-95={self.best_map5095:.4f}  "
              f"val_loss={self.best_val_loss:.4f}")
        print(f"Weights → {self.save_dir}/best.weights.h5")
        print(f"CSV     → {self.csv_path}")

    # ── Console print ───────────────────────────────────────────────────

    def _print_row(self, epoch, epochs, tl, vl, met, lr, dt, star):
        try:
            import pynvml; pynvml.nvmlInit()
            h  = pynvml.nvmlDeviceGetHandleByIndex(0)
            gm = f"  {pynvml.nvmlDeviceGetMemoryInfo(h).used/1e9:.1f}G"
        except Exception:
            gm = ""
        flag  = " ✓" if star else ""
        cls_w = tl['cls'] * self.loss_fn.cls_gain
        print(f"{epoch+1:>3}/{epochs}  "
              f"{tl['box']:>8.4f} {tl['cls']:>8.4f} {cls_w:>7.4f} {tl['dfl']:>8.4f}  │  "
              f"{met['precision']:>6.3f} {met['recall']:>6.3f} {met['f1']:>6.3f} "
              f"{met['map50']:>8.4f} {met['map5095']:>10.4f}  "
              f"{lr:>9.2e}  {dt:.0f}s{flag}{gm}")

    def _print_per_class(self, met):
        names = self.class_names or [f"cls{c:02d}" for c in range(self.nc)]
        print(f"\n  {'class':<22} {'#GT':>6} {'P':>6} {'R':>6} "
              f"{'AP50':>8} {'F1':>6}")
        print("  " + "─" * 54)
        for c in range(self.nc):
            if met["n_gt_per_cls"][c] == 0:
                continue
            print(f"  {names[c]:<22} {met['n_gt_per_cls'][c]:>6} "
                  f"{met['p_per_cls'][c]:>6.3f} {met['r_per_cls'][c]:>6.3f} "
                  f"{met['ap50_per_cls'][c]:>8.4f} {met['f1_per_cls'][c]:>6.3f}")
        print()
