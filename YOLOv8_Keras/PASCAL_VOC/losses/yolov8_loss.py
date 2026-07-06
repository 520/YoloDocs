"""
YOLOv8 Loss — matches current Ultralytics v8DetectionLoss exactly.

Three components:
  1. cls_loss  → BCE (sigmoid cross-entropy)
                 tf.nn.sigmoid_cross_entropy_with_logits  [standard TF]

  2. box_loss  → CIoU
                 _ciou_loss() below — direct TF implementation, no library.
                 Replaces keras_cv.losses.CIoULoss which had API instability.

  3. dfl_loss  → Distribution Focal Loss
                 ~15 lines of tf cross-entropy — no library exists for this.

Loss weights (from TRAIN_CFG / default.yaml):
  box = 7.5,  cls = 0.5,  dfl = 1.5

Normalisation: divide every component by sum of target_scores (not num_fg).
This matches the current Ultralytics implementation exactly:
  target_scores_sum = max(target_scores.sum(), 1)
"""

import math
import tensorflow as tf

from losses.tal_assigner import (
    TaskAlignedAssigner,
    make_anchors,
    dfl_decode,
    dist2bbox,
    bbox2dist,
)




# ---------------------------------------------------------------------------
# CIoU loss  — matches Ultralytics bbox_iou(CIoU=True) exactly
# ---------------------------------------------------------------------------

def _ciou_loss(pred_xyxy, gt_xyxy, eps=1e-7):
    """
    Complete-IoU loss, element-wise.
    pred_xyxy, gt_xyxy: [N, 4]  float32  xyxy absolute pixels
    Returns [N] loss values  (= 1 - CIoU).
    """
    px1, py1, px2, py2 = tf.unstack(pred_xyxy, axis=-1)
    gx1, gy1, gx2, gy2 = tf.unstack(gt_xyxy,  axis=-1)

    pw = px2 - px1;  ph = py2 - py1
    gw = gx2 - gx1;  gh = gy2 - gy1
    pcx = (px1 + px2) / 2;  pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2;  gcy = (gy1 + gy2) / 2

    # Intersection
    ix1 = tf.maximum(px1, gx1);  iy1 = tf.maximum(py1, gy1)
    ix2 = tf.minimum(px2, gx2);  iy2 = tf.minimum(py2, gy2)
    inter = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)

    # Union
    p_area = pw * ph
    g_area = gw * gh
    union   = p_area + g_area - inter + eps
    iou     = inter / union

    # Enclosing box
    ex1 = tf.minimum(px1, gx1);  ey1 = tf.minimum(py1, gy1)
    ex2 = tf.maximum(px2, gx2);  ey2 = tf.maximum(py2, gy2)
    c2  = tf.square(ex2 - ex1) + tf.square(ey2 - ey1) + eps

    # Centre distance
    rho2 = tf.square(pcx - gcx) + tf.square(pcy - gcy)

    # Aspect-ratio consistency term v and alpha
    v = (4.0 / (math.pi ** 2)) * tf.square(
        tf.atan(gw / (gh + eps)) - tf.atan(pw / (ph + eps)))
    alpha_ciou = v / (1.0 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha_ciou * v
    return 1.0 - ciou   # loss = 1 - CIoU


# ---------------------------------------------------------------------------
# DFL loss  (~10 lines — no library exists)
# ---------------------------------------------------------------------------

def dfl_loss(pred_dist_logits, target_dist, reg_max=16):
    """
    Distribution Focal Loss.
    Ultralytics source (loss.py / DFLoss.__call__):
      tl = target.long()
      tr = tl + 1
      wl = tr - target   (weight left)
      wr = 1 - wl        (weight right)
      loss = CE(pred, tl)*wl + CE(pred, tr)*wr

    pred_dist_logits : [N_fg, 4, reg_max]  raw logits
    target_dist      : [N_fg, 4]           continuous in [0, reg_max-1]
    Returns scalar.
    """
    target_dist = tf.clip_by_value(target_dist, 0.0, reg_max - 1 - 0.01)
    tl = tf.cast(tf.math.floor(target_dist), tf.int32)          # [N, 4]
    tr = tf.minimum(tl + 1, reg_max - 1)
    wl = tf.cast(tr, tf.float32) - target_dist                  # weight left
    wr = 1.0 - wl                                               # weight right

    # cross_entropy expects logits [N*4, reg_max], labels [N*4]
    N  = tf.shape(pred_dist_logits)[0]
    logits_flat = tf.reshape(pred_dist_logits, [-1, reg_max])   # [N*4, reg_max]
    tl_flat     = tf.reshape(tl, [-1])                          # [N*4]
    tr_flat     = tf.reshape(tr, [-1])

    ce_l = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tl_flat, logits=logits_flat)                     # [N*4]
    ce_r = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tr_flat, logits=logits_flat)

    ce_l = tf.reshape(ce_l, [N, 4])
    ce_r = tf.reshape(ce_r, [N, 4])

    loss = (ce_l * wl + ce_r * wr)                              # [N, 4]
    return tf.reduce_mean(loss, axis=-1)                        # [N] — mean over 4 sides


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------

class YOLOv8Loss:
    """
    YOLOv8 detection loss.

    Parameters come from TRAIN_CFG:
        nc, reg_max, strides,
        box_gain=7.5, cls_gain=0.5, dfl_gain=1.5,
        tal_topk=10, tal_alpha=0.5, tal_beta=6.0

    Usage:
        loss_fn = YOLOv8Loss(nc=80, reg_max=16)
        total_loss, loss_dict = loss_fn(predictions, batch)
    """

    def __init__(
        self,
        nc=80,
        reg_max=16,
        strides=None,
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        tal_topk=10,
        tal_alpha=0.5,
        tal_beta=6.0,
        tal_min_anchor_size=8.0,
        tal_expand_anchor_size=16.0,
    ):
        self.nc       = nc
        self.reg_max  = reg_max
        self.strides  = strides or [8, 16, 32]
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, alpha=tal_alpha, beta=tal_beta,
            min_anchor_size=tal_min_anchor_size,
            expand_anchor_size=tal_expand_anchor_size,
        )
        # one-shot diagnostic flag — works inside @tf.function
        self._diag_done = tf.Variable(False, trainable=False, dtype=tf.bool)

    def __call__(self, predictions, batch):
        """
        Args
        ----
        predictions : list of [B, Hi*Wi, 4*reg_max + nc]  (raw head outputs)
        batch       : dict
            "images"    [B, H, W, 3]
            "gt_bboxes" [B, max_gt, 4]  xyxy normalised 0-1
            "gt_labels" [B, max_gt]     int32  (-1 = padding)
            "mask_gt"   [B, max_gt]     bool

        Returns
        -------
        total_loss : scalar tf.float32
        loss_dict  : {"box": ..., "cls": ..., "dfl": ...}
        """
        dtype  = predictions[0].dtype
        imgsz  = tf.cast(tf.shape(batch["images"])[1], tf.float32)

        # ── split reg / cls ──────────────────────────────────────────────
        pred_distri = tf.concat(
            [p[..., :4 * self.reg_max] for p in predictions], axis=1)  # [B,N,4r]
        pred_scores = tf.concat(
            [p[..., 4 * self.reg_max:] for p in predictions], axis=1)  # [B,N,nc]

        # ── anchors ──────────────────────────────────────────────────────
        import math
        feature_shapes, stride_list = [], []
        for p, s in zip(predictions, self.strides):
            n    = p.shape[1] or tf.shape(p)[1]
            side = int(round(math.sqrt(int(n))))
            feature_shapes.append((side, side))
            stride_list.append(s)

        anc_pts, stride_t = make_anchors(feature_shapes, stride_list, offset=0.5)
        anc_px = anc_pts * stride_t                                   # [N,2] pixels

        # ── decode predicted boxes ────────────────────────────────────────
        pred_dist_dec = dfl_decode(pred_distri, reg_max=self.reg_max)  # [B,N,4]
        pred_dist_px  = pred_dist_dec * stride_t[tf.newaxis]           # pixels
        pred_bboxes   = dist2bbox(pred_dist_px, anc_px[tf.newaxis])    # [B,N,4] xyxy

        # ── scale gt to pixels ───────────────────────────────────────────
        gt_bboxes_px = batch["gt_bboxes"] * imgsz                      # [B,max_gt,4]

        # ── TAL assignment ───────────────────────────────────────────────
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            tf.sigmoid(tf.cast(pred_scores, tf.float32)),
            tf.stop_gradient(tf.cast(pred_bboxes, tf.float32)),
            anc_px,
            batch["gt_labels"],
            tf.stop_gradient(tf.cast(gt_bboxes_px, tf.float32)),
            batch["mask_gt"],
        )
        # target_scores : [B, N, nc]   soft alignment-weighted one-hot
        # fg_mask       : [B, N]       bool

        # ── normalisation denominator (Ultralytics: target_scores.sum) ───
        target_scores_sum = tf.maximum(
            tf.reduce_sum(tf.cast(target_scores, tf.float32)), 1.0)

        # ── one-shot diagnostic via tf.print + tf.Variable flag ─────────
        # Works inside @tf.function (no .numpy() calls).
        # Fires exactly once on the first training step, then self-disables.
        def _print_diag():
            n_fg = tf.reduce_sum(tf.cast(fg_mask, tf.float32))
            valid_labels = tf.boolean_mask(
                tf.reshape(batch["gt_labels"], [-1]),
                tf.reshape(batch["mask_gt"],   [-1]),
            )
            tf.print("\n[LOSS DIAG] ─────────────────────────────────────────")
            tf.print("  n_fg (fg anchors)  :", n_fg,
                     "  (expected ~1456 for batch=16, ~7 GTs/img, topk=10)")
            tf.print("  target_scores_sum  :", target_scores_sum,
                     "  (expected: hundreds)")
            tf.print("  gt_bboxes [norm]   : min =", tf.reduce_min(batch["gt_bboxes"]),
                     " max =", tf.reduce_max(batch["gt_bboxes"]),
                     "  (expected [0, 1])")
            tf.print("  gt_bboxes [pixel]  : min =", tf.reduce_min(gt_bboxes_px),
                     " max =", tf.reduce_max(gt_bboxes_px),
                     "  (expected [0, 640])")
            tf.print("  gt_labels [valid]  : min =", tf.reduce_min(valid_labels),
                     " max =", tf.reduce_max(valid_labels),
                     "  (expected [0, 19])")
            tf.print("[LOSS DIAG] ─────────────────────────────────────────\n")
            return self._diag_done.assign(True)

        tf.cond(
            tf.logical_not(self._diag_done),
            _print_diag,
            lambda: tf.identity(self._diag_done),
        )

        # ════════════════════════════════════════════════════════════════
        # 1. CLS loss — BCE  (standard TF, no library needed)
        #    Ultralytics: bce(pred_scores, target_scores).sum() / tss
        #    Always in float32: PyTorch AMP autocasts BCEWithLogitsLoss to
        #    float32 for numerical stability; we mirror that here.
        # ════════════════════════════════════════════════════════════════
        cls_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(target_scores, tf.float32),
                logits=tf.cast(pred_scores,   tf.float32),
            )
        ) / target_scores_sum

        # ════════════════════════════════════════════════════════════════
        # 2. BOX loss — CIoU  (_ciou_loss, direct TF, no library)
        #    Only on foreground anchors.
        #    Weighted by target_scores (alignment score), matching Ultralytics:
        #      iou = bbox_iou(pred, target, CIoU=True)
        #      loss_box = ((1 - iou) * weight).sum() / tss
        # ════════════════════════════════════════════════════════════════
        fg_idx = tf.where(fg_mask)                             # [n_fg, 2]

        pred_bboxes_fg   = tf.gather_nd(pred_bboxes,   fg_idx)  # [n_fg, 4]
        target_bboxes_fg = tf.gather_nd(target_bboxes, fg_idx)  # [n_fg, 4]
        weight_fg        = tf.gather_nd(
            tf.reduce_max(target_scores, axis=-1), fg_idx)       # [n_fg]

        if tf.shape(fg_idx)[0] > 0:
            # CIoU loss — computed directly, matches Ultralytics bbox_iou(CIoU=True)
            # No keras_cv dependency needed; avoids API version issues entirely.
            ciou_per = _ciou_loss(
                tf.cast(pred_bboxes_fg,   tf.float32),
                tf.cast(target_bboxes_fg, tf.float32),
            )                                                         # [n_fg]
            box_loss = tf.reduce_sum(
                ciou_per * tf.cast(weight_fg, tf.float32)
            ) / target_scores_sum
        else:
            box_loss = tf.constant(0.0, dtype=tf.float32)

        # ════════════════════════════════════════════════════════════════
        # 3. DFL loss  (custom ~10 lines)
        #    Encode gt boxes → distance targets, then cross-entropy
        # ════════════════════════════════════════════════════════════════
        if tf.shape(fg_idx)[0] > 0:
            # stride per fg anchor
            stride_fg = tf.gather(stride_t, tf.cast(fg_idx[:, 1], tf.int32))  # [n_fg,1]
            anc_fg    = tf.gather(anc_pts,  tf.cast(fg_idx[:, 1], tf.int32))  # [n_fg,2]

            # target distances in stride units
            target_dist = bbox2dist(
                tf.cast(target_bboxes_fg, tf.float32) / tf.cast(stride_fg, tf.float32),
                anc_fg,
                self.reg_max,
            )  # [n_fg, 4]

            # predicted distribution logits for fg anchors
            pred_dist_fg = tf.gather_nd(pred_distri, fg_idx)            # [n_fg,4*reg_max]
            pred_dist_fg = tf.reshape(
                pred_dist_fg, [-1, 4, self.reg_max])                     # [n_fg,4,reg_max]

            dfl_per = dfl_loss(
                tf.cast(pred_dist_fg, tf.float32),
                target_dist,
                self.reg_max,
            )  # [n_fg]
            dfl_loss_val = tf.reduce_sum(
                dfl_per * tf.cast(weight_fg, tf.float32)
            ) / target_scores_sum
        else:
            dfl_loss_val = tf.constant(0.0, dtype=tf.float32)

        # ── weighted sum ─────────────────────────────────────────────────
        total = (self.box_gain * tf.cast(box_loss,      tf.float32) +
                 self.cls_gain * tf.cast(cls_loss,      tf.float32) +
                 self.dfl_gain * tf.cast(dfl_loss_val,  tf.float32))

        return total, {
            "box": tf.cast(box_loss,     tf.float32),
            "cls": tf.cast(cls_loss,     tf.float32),
            "dfl": tf.cast(dfl_loss_val, tf.float32),
        }