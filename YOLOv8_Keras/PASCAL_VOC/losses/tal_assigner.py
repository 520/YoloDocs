"""
Task-Aligned Learning (TAL) anchor assigner + box decode utilities.
Matches Ultralytics implementation exactly:
  - topk=10, alpha=0.5, beta=6.0
  - alignment score: (cls_score^alpha) * (iou^beta)
"""

import tensorflow as tf
import math


# -----------------------------------------------------------------------
# Box utilities
# -----------------------------------------------------------------------

def dist2bbox(distance, anchor_points, xywh=False):
    """
    Decode DFL distances [lt, rb] -> xyxy (or xywh).
    distance: [B, N, 4]  (lt_x, lt_y, rb_x, rb_y in grid units)
    anchor_points: [N, 2]  (cx, cy in grid units)
    """
    lt, rb = tf.split(distance, 2, axis=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return tf.concat([c_xy, wh], axis=-1)
    return tf.concat([x1y1, x2y2], axis=-1)


def bbox2dist(bbox_xyxy, anchor_points, reg_max):
    """
    Encode gt boxes to DFL targets clamped to [0, reg_max-1-eps].
    bbox_xyxy: [N, 4]
    anchor_points: [N, 2]
    """
    lt = anchor_points - bbox_xyxy[..., :2]
    rb = bbox_xyxy[..., 2:] - anchor_points
    dist = tf.concat([lt, rb], axis=-1)
    return tf.clip_by_value(dist, 0.0, reg_max - 1 - 1e-3)


def box_iou(b1, b2):
    """
    Compute IoU between two sets of boxes [N,4] and [M,4].
    Returns [N, M] IoU matrix.  Boxes in xyxy format.
    """
    # b1: [N, 1, 4], b2: [1, M, 4]
    b1 = tf.expand_dims(b1, 1)
    b2 = tf.expand_dims(b2, 0)
    inter_lt = tf.maximum(b1[..., :2], b2[..., :2])
    inter_rb = tf.minimum(b1[..., 2:], b2[..., 2:])
    inter_wh = tf.maximum(inter_rb - inter_lt, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter_area + 1e-9
    return inter_area / union


def ciou_loss(pred_xyxy, gt_xyxy, eps=1e-7):
    """
    Complete-IoU loss element-wise.
    pred_xyxy, gt_xyxy: [N, 4]
    Returns [N] loss values.
    """
    px1, py1, px2, py2 = tf.unstack(pred_xyxy, axis=-1)
    gx1, gy1, gx2, gy2 = tf.unstack(gt_xyxy, axis=-1)

    pw, ph = px2 - px1, py2 - py1
    gw, gh = gx2 - gx1, gy2 - gy1
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2

    # IoU
    inter_x1 = tf.maximum(px1, gx1)
    inter_y1 = tf.maximum(py1, gy1)
    inter_x2 = tf.minimum(px2, gx2)
    inter_y2 = tf.minimum(py2, gy2)
    inter_area = tf.maximum(inter_x2 - inter_x1, 0) * tf.maximum(inter_y2 - inter_y1, 0)
    p_area = pw * ph
    g_area = gw * gh
    union = p_area + g_area - inter_area + eps
    iou = inter_area / union

    # enclosing box
    enc_x1 = tf.minimum(px1, gx1)
    enc_y1 = tf.minimum(py1, gy1)
    enc_x2 = tf.maximum(px2, gx2)
    enc_y2 = tf.maximum(py2, gy2)
    c2 = tf.square(enc_x2 - enc_x1) + tf.square(enc_y2 - enc_y1) + eps

    # centre distance
    rho2 = tf.square(pcx - gcx) + tf.square(pcy - gcy)

    # v (aspect ratio consistency term)
    v = (4 / (math.pi ** 2)) * tf.square(
        tf.atan(gw / (gh + eps)) - tf.atan(pw / (ph + eps))
    )
    alpha_ciou = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha_ciou * v
    return 1 - ciou


# -----------------------------------------------------------------------
# DFL — decode raw distribution to scalar distance
# -----------------------------------------------------------------------

def dfl_decode(pred_dist, reg_max=16):
    """
    Convert raw DFL logits to continuous box distances.
    pred_dist: [B, N, 4*reg_max]
    Returns:   [B, N, 4]  (expected value of each distribution)
    """
    B = tf.shape(pred_dist)[0]
    N = tf.shape(pred_dist)[1]
    # reshape to [B, N, 4, reg_max]
    pred_dist = tf.reshape(pred_dist, [B, N, 4, reg_max])
    weights = tf.cast(tf.range(reg_max), pred_dist.dtype)  # [reg_max]
    # softmax then expected value
    dist = tf.reduce_sum(tf.nn.softmax(pred_dist, axis=-1) * weights, axis=-1)
    return dist  # [B, N, 4]


# -----------------------------------------------------------------------
# Anchor grid generator (matches Ultralytics stride/offset convention)
# -----------------------------------------------------------------------

def make_anchors(feature_shapes, strides, offset=0.5):
    """
    Generate anchor centre points for all scales.

    Args:
        feature_shapes: list of (H, W) per scale
        strides: list of strides (e.g. [8, 16, 32])
        offset: 0.5 to place anchor at cell centre

    Returns:
        anchor_points: [total_anchors, 2]  in *stride units* (not pixels)
        stride_tensor: [total_anchors, 1]  stride value per anchor
    """
    anchor_list = []
    stride_list = []
    for (h, w), s in zip(feature_shapes, strides):
        sx = tf.cast(tf.range(w), tf.float32) + offset
        sy = tf.cast(tf.range(h), tf.float32) + offset
        grid_x, grid_y = tf.meshgrid(sx, sy)
        grid = tf.stack([grid_x, grid_y], axis=-1)          # [H, W, 2]
        grid = tf.reshape(grid, [-1, 2])                     # [H*W, 2]
        anchor_list.append(grid)
        stride_list.append(tf.fill([h * w, 1], float(s)))
    anchor_points = tf.concat(anchor_list, axis=0)           # [N, 2]
    stride_tensor = tf.concat(stride_list, axis=0)           # [N, 1]
    return anchor_points, stride_tensor


# -----------------------------------------------------------------------
# TAL Assigner
# -----------------------------------------------------------------------

class TaskAlignedAssigner:
    """
    Task-Aligned Learning assigner from YOLOv8.
    For each ground-truth box, select topk anchors by alignment score.

    Params (from Ultralytics default):
        topk   = 10
        alpha  = 0.5   (cls score power)
        beta   = 6.0   (iou power)
        eps    = 1e-9
    """

    def __init__(self, topk=10, alpha=0.5, beta=6.0, eps=1e-9,
                 min_anchor_size=8.0, expand_anchor_size=16.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.min_anchor_size = min_anchor_size      # expand GT boxes smaller than this (px)
        self.expand_anchor_size = expand_anchor_size  # expand to this size (px)

    def __call__(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores:  [B, N, nc]   sigmoid predictions
            pd_bboxes:  [B, N, 4]    decoded xyxy predictions (in pixel space)
            anc_points: [N, 2]       anchor centre points in pixel space
            gt_labels:  [B, max_gt]  int class labels (-1 for padding)
            gt_bboxes:  [B, max_gt, 4] xyxy in pixel space
            mask_gt:    [B, max_gt]  bool, True = valid gt

        Returns:
            target_labels:   [B, N]      assigned class per anchor (-1 = bg)
            target_bboxes:   [B, N, 4]   assigned gt box per anchor
            target_scores:   [B, N, nc]  soft label weights (alignment score)
            fg_mask:         [B, N]      bool, True = foreground anchor
        """
        B = tf.shape(gt_bboxes)[0]
        max_gt = tf.shape(gt_bboxes)[1]
        N = tf.shape(anc_points)[0]
        nc = pd_scores.shape[-1]

        # 1. Anchors inside gt boxes
        # anc_points: [N, 2] -> check against each gt box [B, max_gt, 4]
        ax = anc_points[:, 0]  # [N]
        ay = anc_points[:, 1]

        # gt boxes [B, max_gt, 4] xyxy
        gx1 = gt_bboxes[:, :, 0]  # [B, max_gt]
        gy1 = gt_bboxes[:, :, 1]
        gx2 = gt_bboxes[:, :, 2]
        gy2 = gt_bboxes[:, :, 3]

        # Expand tiny GT boxes so small objects still get anchor candidates.
        # Mirrors Ultralytics select_candidates_in_gts: w/h < stride[0] → stride_val.
        gcx = (gx1 + gx2) / 2;  gcy = (gy1 + gy2) / 2
        gw  = gx2 - gx1;         gh  = gy2 - gy1
        valid = tf.cast(mask_gt, tf.bool)
        min_s = tf.cast(self.min_anchor_size,    tf.float32)
        exp_s = tf.cast(self.expand_anchor_size, tf.float32)
        gw_ck = tf.where((gw < min_s) & valid, tf.ones_like(gw) * exp_s, gw)
        gh_ck = tf.where((gh < min_s) & valid, tf.ones_like(gh) * exp_s, gh)
        gx1_ck = gcx - gw_ck / 2;  gx2_ck = gcx + gw_ck / 2
        gy1_ck = gcy - gh_ck / 2;  gy2_ck = gcy + gh_ck / 2

        # [B, max_gt, N]
        in_left  = tf.expand_dims(ax, 0)[tf.newaxis] > tf.expand_dims(gx1_ck, -1)
        in_top   = tf.expand_dims(ay, 0)[tf.newaxis] > tf.expand_dims(gy1_ck, -1)
        in_right = tf.expand_dims(ax, 0)[tf.newaxis] < tf.expand_dims(gx2_ck, -1)
        in_bot   = tf.expand_dims(ay, 0)[tf.newaxis] < tf.expand_dims(gy2_ck, -1)
        is_in_box = in_left & in_top & in_right & in_bot  # [B, max_gt, N]

        # apply mask_gt [B, max_gt, 1]
        is_in_box = is_in_box & tf.expand_dims(mask_gt, -1)  # [B, max_gt, N]

        # 2. Alignment score = cls_score^alpha * iou^beta
        # Get predicted cls score for each gt class
        gt_label_idx = tf.maximum(tf.cast(gt_labels, tf.int32), 0)  # [B, max_gt]
        # gather predicted scores: [B, max_gt, N]
        # pd_scores: [B, N, nc] -> [B, 1, N, nc] -> gather on last dim
        pd_scores_t = tf.transpose(pd_scores, [0, 2, 1])  # [B, nc, N]
        # for each gt, get its class score: index into nc dim
        gt_scores = tf.gather(pd_scores_t, gt_label_idx, batch_dims=1)  # [B, max_gt, N]

        # IoU between pd_bboxes [B, N, 4] and gt_bboxes [B, max_gt, 4]
        # Compute batch-wise: for each image separately via map_fn or vectorised
        iou_mat = self._batch_box_iou(gt_bboxes, pd_bboxes)  # [B, max_gt, N]

        align_score = (tf.pow(gt_scores, self.alpha) *
                       tf.pow(tf.maximum(iou_mat, 0.0), self.beta))  # [B, max_gt, N]

        # Zero out anchors not inside any gt box
        align_score = tf.where(is_in_box, align_score, tf.zeros_like(align_score))

        # 3. TopK selection per gt
        # For each gt: take top-k anchors by alignment score
        topk_scores, topk_idx = tf.math.top_k(align_score, k=self.topk)  # [B, max_gt, topk]
        # Build topk mask: [B, max_gt, N]
        topk_mask = tf.reduce_any(
            tf.equal(
                tf.expand_dims(tf.range(N), 0)[tf.newaxis],  # [1,1,N]
                tf.expand_dims(topk_idx, -1),                 # [B, max_gt, topk, 1]... need reshape
            ),
            axis=-2
        )
        # Simpler: scatter topk_idx into mask
        topk_mask = self._topk_to_mask(topk_idx, N, B, max_gt)  # [B, max_gt, N]
        topk_mask = topk_mask & is_in_box

        # 4. Resolve conflicts: anchor assigned to gt with highest align_score
        # [B, max_gt, N] -> sum over max_gt: if > 1, anchor is contested
        mask_sum = tf.reduce_sum(tf.cast(topk_mask, tf.float32), axis=1)  # [B, N]
        # For contested: keep only the gt with highest align score
        # [B, max_gt, N]: align_score masked
        masked_align = tf.where(topk_mask, align_score, tf.zeros_like(align_score))
        best_gt = tf.argmax(masked_align, axis=1, output_type=tf.int32)  # [B, N]

        # Rebuild: each anchor -> at most one gt
        one_hot = tf.one_hot(best_gt, max_gt)  # [B, N, max_gt]
        one_hot = tf.transpose(one_hot, [0, 2, 1])  # [B, max_gt, N]
        topk_mask = topk_mask & tf.cast(one_hot, tf.bool)

        fg_mask = tf.reduce_any(topk_mask, axis=1)  # [B, N]

        # 5. Gather targets
        # For each anchor, which gt is assigned?
        assign_gt_idx = tf.argmax(tf.cast(topk_mask, tf.float32), axis=1,
                                  output_type=tf.int32)  # [B, N]

        # target boxes: gather from gt_bboxes
        target_bboxes = tf.gather(gt_bboxes, assign_gt_idx, batch_dims=1)  # [B, N, 4]

        # target labels
        target_labels = tf.gather(gt_labels, assign_gt_idx, batch_dims=1)  # [B, N]
        target_labels = tf.where(fg_mask, tf.cast(target_labels, tf.int32),
                                 tf.fill(tf.shape(target_labels), -1))

        # target_scores: Ultralytics per-GT normalization (tal.py lines 138-142)
        # norm = (align / max_align_per_GT) * max_iou_per_GT, then amax over GTs per anchor
        masked_align  = tf.where(topk_mask, align_score, tf.zeros_like(align_score))  # [B, max_gt, N]
        pos_align_max = tf.reduce_max(masked_align, axis=-1, keepdims=True)            # [B, max_gt, 1]
        masked_iou    = tf.where(topk_mask, iou_mat,    tf.zeros_like(iou_mat))        # [B, max_gt, N]
        pos_iou_max   = tf.reduce_max(masked_iou,   axis=-1, keepdims=True)            # [B, max_gt, 1]
        norm_align    = masked_align * pos_iou_max / (pos_align_max + self.eps)        # [B, max_gt, N]
        norm_align_per_anchor = tf.reduce_max(norm_align, axis=1)                      # [B, N]

        # Expand to one-hot class scores: [B, N, nc]
        cls_one_hot = tf.one_hot(
            tf.maximum(target_labels, 0), nc, dtype=tf.float32
        )  # [B, N, nc]
        target_scores = tf.expand_dims(norm_align_per_anchor, -1) * cls_one_hot
        target_scores = tf.where(
            tf.expand_dims(fg_mask, -1),
            target_scores,
            tf.zeros_like(target_scores)
        )

        return target_labels, target_bboxes, target_scores, fg_mask

    # ---- helpers ----

    def _batch_box_iou(self, gt_bboxes, pd_bboxes):
        """
        [B, max_gt, 4] x [B, N, 4] -> [B, max_gt, N]
        """
        # gt: [B, max_gt, 1, 4], pd: [B, 1, N, 4]
        gt = tf.expand_dims(gt_bboxes, 2)
        pd = tf.expand_dims(pd_bboxes, 1)
        inter_lt = tf.maximum(gt[..., :2], pd[..., :2])
        inter_rb = tf.minimum(gt[..., 2:], pd[..., 2:])
        inter_wh = tf.maximum(inter_rb - inter_lt, 0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        area_gt = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
        area_pd = (pd[..., 2] - pd[..., 0]) * (pd[..., 3] - pd[..., 1])
        union = area_gt + area_pd - inter_area + self.eps
        return inter_area / union  # [B, max_gt, N]

    def _topk_to_mask(self, topk_idx, N, B, max_gt):
        """Convert top-k indices [B, max_gt, topk] to boolean mask [B, max_gt, N]."""
        # Scatter 1 at each topk index position
        B_dyn = tf.shape(topk_idx)[0]
        k = tf.shape(topk_idx)[2]
        # flat indices for scatter
        b_idx = tf.tile(tf.reshape(tf.range(B_dyn), [B_dyn, 1, 1]), [1, max_gt, k])
        g_idx = tf.tile(tf.reshape(tf.range(max_gt), [1, max_gt, 1]), [B_dyn, 1, k])
        indices = tf.stack([b_idx, g_idx, topk_idx], axis=-1)  # [B, max_gt, k, 3]
        updates = tf.ones([B_dyn, max_gt, k], dtype=tf.float32)
        shape = tf.stack([B_dyn, max_gt, N])
        mask_f = tf.tensor_scatter_nd_update(
            tf.zeros(shape, dtype=tf.float32), 
            tf.reshape(indices, [-1, 3]),
            tf.reshape(updates, [-1])
        )
        return tf.cast(mask_f, tf.bool)
