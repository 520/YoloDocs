"""
YOLOv8 Keras — PASCAL VOC entry point.

Mirrors ../main.py exactly, with:
  - VOCLoader instead of COCOLoader (reads YOLO txt labels, letterbox)
  - nc=20, VOC class names
  - LOAD_PRETRAINED: False → train from scratch; True → transfer COCO backbone
  - cos_lr=False, momentum=0.937 (Ultralytics MuSGD default)

Transfer learning note (LOAD_PRETRAINED=True):
  Backbone + neck + head intermediates + box finals are loaded from COCO weights.
  Cls finals are SKIPPED (nc=80 vs nc=20); they keep the VOC cls-bias init.
  Recommended: lower lr0 (e.g. 0.0001) in voc_cfg.py when fine-tuning.

Run from PASCAL_VOC/ directory:
    python main_voc.py

Data paths (set below):
    TRAIN_DIR : full path to train images dir
    VAL_DIR   : full path to val images dir
    Labels are derived automatically: /images/ → /labels/
"""

import os
import platform
from pathlib import Path
import cv2
import numpy as np

# Required for tf.keras.optimizers.legacy.* on TensorFlow/Keras 3 stacks.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# XLA needs libdevice.10.bc — it lives one directory up in nvvm/libdevice/.
# Without this, swish sigmoid triggers a JIT compilation failure when running
# from a subdirectory (XLA searches CWD as a fallback and misses the file).
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if platform.system() != "Darwin":
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={_PARENT}"

from voc_cfg import TRAIN_CFG, VOC_CATEGORIES

import tensorflow as tf


# ── config ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "kitti"
TRAIN_DIR = str(DATASET_ROOT / "images" / "train")
VAL_DIR   = str(DATASET_ROOT / "images" / "val")

VARIANT   = "s"
# VARIANT   = "l"
SAVE_DIR  = "runs/detect/train"
TASK      = "detect"
# NUM_CLASS = 20
NUM_CLASS = 8

# Set True to transfer COCO pretrained backbone/neck weights (recommended).
# Set False to train from scratch.
# When True: backbone+neck+box_head transferred; cls_head skipped (nc mismatch).
# PRETRAINED_PT: path to local .pt file, or None to auto-download.
LOAD_PRETRAINED = True
PRETRAINED_PT   = str(PROJECT_ROOT / "Ultralytics_Models" / f"yolov8{VARIANT}.pt")


def save_train_previews(train_loader, output_dir, max_images=8):
    """Save one processed train batch with ground-truth boxes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(train_loader.build_dataset()))
    images = batch["images"].numpy()
    boxes = batch["gt_bboxes"].numpy()
    labels = batch["gt_labels"].numpy()
    masks = batch["mask_gt"].numpy()
    imgsz = int(TRAIN_CFG["imgsz"])
    save_count = min(len(images), max_images)

    for i in range(save_count):
        image = (np.clip(images[i], 0.0, 1.0) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for box, class_id, valid in zip(boxes[i], labels[i], masks[i]):
            if not valid:
                continue

            x1, y1, x2, y2 = np.rint(
                box * np.array([imgsz, imgsz, imgsz, imgsz])
            ).astype(int)
            x1, x2 = np.clip([x1, x2], 0, imgsz - 1)
            y1, y2 = np.clip([y1, y2], 0, imgsz - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(class_id)
            color = (
                int((37 * class_id + 80) % 255),
                int((17 * class_id + 160) % 255),
                int((29 * class_id + 220) % 255),
            )
            class_name = (
                VOC_CATEGORIES[class_id]
                if 0 <= class_id < len(VOC_CATEGORIES)
                else str(class_id)
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                class_name,
                (x1, max(y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        output_path = output_dir / f"train_processed_{i:02d}.jpg"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Failed to save processed train image: {output_path}")

    print(f"Saved {save_count} processed train images → {output_dir.resolve()}")




def gpu_allocation(memory_limit_mb):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs found, running on CPU.")
        return
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
        )
        logical = tf.config.list_logical_devices('GPU')
        print(f"GPU configured: {len(gpus)} physical, {len(logical)} logical "
              f"({memory_limit_mb} MB limit)")
    except RuntimeError as e:
        print(f"GPU config error (must be called before any TF op): {e}")

gpu_allocation(TRAIN_CFG["gpu_memory"])

tf.config.optimizer.set_jit(False)

from yolov8_architecture.yolov8_architecture import build_yolov8_pair
from voc_loader import VOCLoader
from training.trainer import YOLOv8Trainer, decode_preds
from training.metrics import DetectionEvaluator
from losses.yolov8_loss import YOLOv8Loss

tf.random.set_seed(TRAIN_CFG["seed"])


# ── Variant-specific config overrides ──────────────────────────────────────
# OLD: single shared TRAIN_CFG for all variants (no per-variant tuning)
# NEW: for large/x + pretrained, apply three overrides critical for fine-tuning:
#   - freeze_epochs=5  : freeze backbone+neck for first 5 epochs so only the cls head
#                        adapts from scratch; prevents high early-LR from destroying
#                        pretrained features (root cause of the epoch-4-peak issue)
#   - ema_decay=0.9998 : slower EMA (~3.5 ep half-life vs 0.7 ep for 0.9990) —
#                        avoids chasing noise in a large stable model
#   - cos_lr is already True in voc_cfg.py (was False previously)
#   - lr0 stays 0.0001 with cosine → decays from 1e-4 to 1e-6 over 100 epochs
TRAIN_CFG = dict(TRAIN_CFG)  # copy so voc_cfg original is not mutated
if VARIANT in ("l", "x") and LOAD_PRETRAINED:
    TRAIN_CFG["freeze_epochs"] = 5
    TRAIN_CFG["ema_decay"]     = 0.9998

if __name__ == "__main__":

    # ── 1. Models ──────────────────────────────────────────────────────────
    print("\n── 1. Building models ──")

    # train_model and inf_model share the same backbone/neck/head Variables
    # (see build_paired_models) — no separate inf_model build + weight
    # transfer step needed; training train_model updates are automatically
    # visible in inf_model.
    train_model, inf_model = build_yolov8_pair(
        variant=VARIANT,
        task=TASK,
        input_shape=TRAIN_CFG["imgsz"],
        num_classes=NUM_CLASS,
    )
    print(f"\nTraining model outputs: {[o.shape for o in train_model.outputs]}")
    print(f"Inference model output: {inf_model.output_shape}")

    if LOAD_PRETRAINED:
        from pretrained_loader_voc import load_ultralytics_weights_voc
        load_ultralytics_weights_voc(train_model, variant=VARIANT,
                                     pt_path=PRETRAINED_PT, verbose=True)

    # ── 2. Data loaders ────────────────────────────────────────────────────
    print("\n── 2. Setting up data loaders ──")
    train_loader = VOCLoader(
        image_dir=TRAIN_DIR,
        imgsz=TRAIN_CFG["imgsz"],
        batch_size=TRAIN_CFG["batch"],
        cfg=TRAIN_CFG,
        augment=True,
        mosaic=False,
    )

    val_loader = VOCLoader(
        image_dir=VAL_DIR,
        imgsz=TRAIN_CFG["imgsz"],
        batch_size=TRAIN_CFG["batch"],
        cfg=TRAIN_CFG,
        augment=False,
        mosaic=False,
    )

    # Save processed train images with ground-truth boxes for visual checking.
    save_train_previews(
        train_loader,
        Path(SAVE_DIR) / "train_preprocess_preview",
    )

    # ── 2.5. Pre-training evaluation (only when pretrained weights loaded) ──
    if LOAD_PRETRAINED:
        print("\n── 2.5. Baseline eval — COCO pretrained before VOC fine-tuning ──")
        print("   (backbone+neck+box transferred; cls head is random → expect low P)")

        loss_fn_pre = YOLOv8Loss(
            nc=NUM_CLASS,
            reg_max=TRAIN_CFG["reg_max"],
            strides=[8, 16, 32],
            box_gain=TRAIN_CFG["box"],
            cls_gain=TRAIN_CFG["cls"],
            dfl_gain=TRAIN_CFG["dfl"],
            tal_topk=TRAIN_CFG["tal_topk"],
            tal_alpha=TRAIN_CFG["tal_alpha"],
            tal_beta=TRAIN_CFG["tal_beta"],
        )

        evaluator_pre = DetectionEvaluator(
            nc=NUM_CLASS,
            imgsz=TRAIN_CFG["imgsz"],
            conf_thres=0.001,
            iou_thres=0.7,
            max_det=300,
        )
        evaluator_pre.reset()
        pbox = pcls = pdfl = pn = 0.0

        for batch in val_loader.build_dataset():
            preds = train_model(batch["images"], training=False)
            _, ld = loss_fn_pre(preds, batch)
            pbox += ld["box"].numpy()
            pcls += ld["cls"].numpy()
            pdfl += ld["dfl"].numpy()
            pn   += 1
            boxes_b, scores_b = decode_preds(
                preds,
                reg_max=TRAIN_CFG["reg_max"],
                strides=[8, 16, 32],
            )
            evaluator_pre.update(
                boxes_b, scores_b,
                batch["gt_bboxes"], batch["gt_labels"], batch["mask_gt"],
            )

        pre = evaluator_pre.compute()
        print("\n" + "═" * 55)
        print("  Pre-training baseline (COCO pretrained → VOC)")
        print("═" * 55)
        print(f"  box_loss   : {pbox/max(pn,1):.4f}")
        print(f"  cls_loss   : {pcls/max(pn,1):.4f}")
        print(f"  dfl_loss   : {pdfl/max(pn,1):.4f}")
        print(f"  Precision  : {pre['precision']:.4f}")
        print(f"  Recall     : {pre['recall']:.4f}")
        print(f"  F1         : {pre['f1']:.4f}")
        print(f"  mAP@0.5    : {pre['map50']:.4f}")
        print(f"  mAP@0.5:95 : {pre['map5095']:.4f}")
        print("═" * 55)

    # ── 3. Train ───────────────────────────────────────────────────────────
    print("\n── 3. Training ──")
    trainer = YOLOv8Trainer(
        model=train_model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=TRAIN_CFG,
        save_dir=SAVE_DIR,
        nc=NUM_CLASS,
        class_names=VOC_CATEGORIES,
    )

    trainer.train()

    # ── 4. Final evaluation on val (best weights) ──────────────────────────
    print("\n── 4. Final evaluation on val set ──")
    train_model.load_weights(os.path.join(SAVE_DIR, "best.weights.h5"))

    loss_fn = YOLOv8Loss(
        nc=NUM_CLASS,
        reg_max=TRAIN_CFG["reg_max"],
        strides=[8, 16, 32],
        box_gain=TRAIN_CFG["box"],
        cls_gain=TRAIN_CFG["cls"],
        dfl_gain=TRAIN_CFG["dfl"],
        tal_topk=TRAIN_CFG["tal_topk"],
        tal_alpha=TRAIN_CFG["tal_alpha"],
        tal_beta=TRAIN_CFG["tal_beta"],
    )

    evaluator = DetectionEvaluator(
        nc=NUM_CLASS,
        imgsz=TRAIN_CFG["imgsz"],
        conf_thres=0.001,
        iou_thres=0.7,
        max_det=300,
    )

    vbox = vcls = vdfl = vn = 0.0
    evaluator.reset()

    for batch in val_loader.build_dataset():
        preds = train_model(batch["images"], training=False)
        _, ld = loss_fn(preds, batch)
        vbox += ld["box"].numpy()
        vcls += ld["cls"].numpy()
        vdfl += ld["dfl"].numpy()
        vn   += 1

        boxes_b, scores_b = decode_preds(
            preds,
            reg_max=TRAIN_CFG["reg_max"],
            strides=[8, 16, 32],
        )
        evaluator.update(
            boxes_b, scores_b,
            batch["gt_bboxes"], batch["gt_labels"], batch["mask_gt"],
        )

    final = evaluator.compute()

    print("\n" + "═" * 55)
    print("  Final val results (best.weights.h5)")
    print("═" * 55)
    print(f"  box_loss   : {vbox/max(vn,1):.4f}")
    print(f"  cls_loss   : {vcls/max(vn,1):.4f}")
    print(f"  dfl_loss   : {vdfl/max(vn,1):.4f}")
    print(f"  Precision  : {final['precision']:.4f}")
    print(f"  Recall     : {final['recall']:.4f}")
    print(f"  F1         : {final['f1']:.4f}")
    print(f"  mAP@0.5    : {final['map50']:.4f}")
    print(f"  mAP@0.5:95 : {final['map5095']:.4f}")
    print("═" * 55)

    print(f"\n  {'class':<22} {'#GT':>6} {'P':>6} {'R':>6} "
          f"{'AP50':>8} {'F1':>6}")
    print("  " + "─" * 54)
    for c in range(NUM_CLASS):
        if final["n_gt_per_cls"][c] == 0:
            continue
        name = VOC_CATEGORIES[c]
        print(f"  {name:<22} {final['n_gt_per_cls'][c]:>6} "
              f"{final['p_per_cls'][c]:>6.3f} "
              f"{final['r_per_cls'][c]:>6.3f} "
              f"{final['ap50_per_cls'][c]:>8.4f} "
              f"{final['f1_per_cls'][c]:>6.3f}")

    # ── 5. Save inference model ────────────────────────────────────────────
    print("\n── 5. Saving inference model ──")
    # inf_model shares Variables with train_model (build_yolov8_pair), so the
    # best-checkpoint weights loaded into train_model above are already
    # reflected here — no transfer step needed.
    inf_model.save_weights(os.path.join(SAVE_DIR, "best_inference.weights.h5"))
    print(f"Inference weights → {SAVE_DIR}/best_inference.weights.h5")

    print(f"\nResults CSV : {os.path.join(SAVE_DIR, 'results.csv')}")
    print(f"Best weights: {os.path.join(SAVE_DIR, 'best.weights.h5')}")
    print("\nSuccessfully DONE!")
