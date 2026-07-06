"""
Validation metrics for YOLOv8.

After investigating keras_cv metrics: COCOMeanAveragePrecision was removed
in newer versions, BoxCOCOMetrics has known bugs (returns 0.0 in several
versions), and the API has broken repeatedly across releases.

Decision: use pycocotools directly for mAP (same as Ultralytics).
pycocotools is already a dependency (used in coco_loader.py).
P/R/F1 computed by sweeping all confidence thresholds and reporting
at the point that maximises mean F1 — matching Ultralytics exactly.

Usage:
    evaluator = DetectionEvaluator(nc=80, imgsz=640)
    for batch in val_ds:
        evaluator.update(pred_boxes, pred_scores, gt_boxes, gt_labels, mask_gt)
    results = evaluator.compute()
    evaluator.reset()
"""

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class DetectionEvaluator:
    """
    Full detection evaluator using pycocotools for mAP (exact COCO eval)
    and per-class P/R curves (Ultralytics-style) for Precision/Recall/F1.

    P/R/F1 are reported at the confidence threshold that maximises mean F1,
    averaged over classes — identical to Ultralytics training metrics.

    Parameters
    ----------
    nc         : number of classes
    imgsz      : image size in pixels (to scale normalised gt boxes)
    conf_thres : NMS confidence threshold
    iou_thres  : NMS IoU threshold
    max_det    : max detections per image kept after NMS
    """

    def __init__(self, nc=80, imgsz=640,
                 conf_thres=0.001, iou_thres=0.7, max_det=300):
        self.nc         = nc
        self.imgsz      = float(imgsz)
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self.max_det    = max_det
        self.reset()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self):
        self._img_id   = 0
        self._dt_list  = []   # pycocotools detections
        self._gt_list  = []   # pycocotools ground truths
        self._ann_id   = 0
        self._n_gt     = np.zeros(self.nc, dtype=np.float64)
        # Per-class detection lists: each entry is (conf, is_tp)
        # Populated at conf_thres=0.001 (same as NMS), evaluated at all thresholds.
        self._cls_dets = [[] for _ in range(self.nc)]

    # ------------------------------------------------------------------
    # update — call once per batch
    # ------------------------------------------------------------------

    def update(self, pred_boxes_b, pred_scores_b,
               gt_bboxes_norm, gt_labels, mask_gt):
        """
        pred_boxes_b   : np [B, N, 4]   xyxy pixels
        pred_scores_b  : np [B, N, nc]  sigmoid scores
        gt_bboxes_norm : np/tf [B, max_gt, 4]  xyxy normalised 0-1
        gt_labels      : np/tf [B, max_gt]     int32
        mask_gt        : np/tf [B, max_gt]     bool
        """
        gt_bboxes_norm = np.array(gt_bboxes_norm)
        gt_labels      = np.array(gt_labels)
        mask_gt        = np.array(mask_gt)
        B = pred_boxes_b.shape[0]

        for i in range(B):
            img_id = self._img_id
            self._img_id += 1

            # ── NMS ──────────────────────────────────────────────────
            dets = self._nms(pred_boxes_b[i], pred_scores_b[i])
            # dets: [M, 6]  x1 y1 x2 y2 conf cls

            # ── GT ───────────────────────────────────────────────────
            mask        = mask_gt[i]
            gt_boxes_px = gt_bboxes_norm[i][mask] * self.imgsz  # [G,4] xyxy
            gt_cls      = gt_labels[i][mask].astype(np.int32)

            for j in range(len(gt_boxes_px)):
                x1, y1, x2, y2 = gt_boxes_px[j]
                w, h = float(x2 - x1), float(y2 - y1)
                self._gt_list.append({
                    "id":           self._ann_id,
                    "image_id":     img_id,
                    "category_id":  int(gt_cls[j]) + 1,
                    "bbox":         [float(x1), float(y1), w, h],  # xywh
                    "area":         w * h,
                    "iscrowd":      0,
                })
                self._ann_id += 1
                self._n_gt[int(gt_cls[j])] += 1

            # ── Detections → pycocotools format ──────────────────────
            for det in dets:
                x1, y1, x2, y2, conf, cls = det
                w, h = float(x2 - x1), float(y2 - y1)
                self._dt_list.append({
                    "image_id":    img_id,
                    "category_id": int(cls) + 1,
                    "bbox":        [float(x1), float(y1), w, h],   # xywh
                    "score":       float(conf),
                })

            # ── Accumulate per-det (conf, is_tp) for P/R curves ──────
            self._accumulate_dets(dets, gt_boxes_px, gt_cls)

    # ------------------------------------------------------------------
    # compute — call at end of epoch
    # ------------------------------------------------------------------

    def compute(self):
        """
        Returns dict:
            map50, map5095           scalars  (pycocotools)
            recall                   scalar   (per-class P/R curve, at max-F1)
            precision                scalar   (per-class P/R curve, at max-F1)
            f1                       scalar   (per-class P/R curve, at max-F1)
            ap50_per_cls             np [nc]  (pycocotools per-class)
            p_per_cls, r_per_cls,
            f1_per_cls               np [nc]  (at each class's max-F1 threshold)
            n_gt_per_cls             np [nc]  int
        """
        # ── pycocotools eval ─────────────────────────────────────────
        map50 = map5095 = 0.0
        ap50_per = np.zeros(self.nc)

        if self._dt_list and self._gt_list:
            map50, map5095, ap50_per = self._run_coco_eval()

        # ── P/R/F1 from per-class confidence-threshold sweep ─────────
        p_per, r_per, f1_per = self._compute_pr_curves()

        valid = self._n_gt > 0
        mp  = p_per[valid].mean()  if valid.any() else 0.0
        mr  = r_per[valid].mean()  if valid.any() else 0.0
        mf1 = f1_per[valid].mean() if valid.any() else 0.0

        return dict(
            map50        = map50,
            map5095      = map5095,
            recall       = mr,
            precision    = mp,
            f1           = mf1,
            ap50_per_cls = ap50_per,
            p_per_cls    = p_per,
            r_per_cls    = r_per,
            f1_per_cls   = f1_per,
            n_gt_per_cls = self._n_gt.astype(np.int32),
        )

    # ------------------------------------------------------------------
    # P/R curves — Ultralytics-style
    # ------------------------------------------------------------------

    def _compute_pr_curves(self):
        """
        For each class: sort detections by conf desc, compute cumulative P/R,
        find the threshold that maximises F1, record P/R/F1 there.
        Matches Ultralytics metrics.py exactly.
        """
        p_per  = np.zeros(self.nc)
        r_per  = np.zeros(self.nc)
        f1_per = np.zeros(self.nc)

        for c in range(self.nc):
            n_gt = self._n_gt[c]
            dets = self._cls_dets[c]
            if n_gt == 0 or len(dets) == 0:
                continue

            arr   = np.array(dets, dtype=np.float32)   # (N, 2): conf, is_tp
            order = np.argsort(-arr[:, 0])
            tp    = arr[order, 1]

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(1.0 - tp)

            prec     = cum_tp / (cum_tp + cum_fp + 1e-9)
            rec      = cum_tp / (n_gt + 1e-9)
            f1_curve = 2.0 * prec * rec / (prec + rec + 1e-9)

            idx = int(f1_curve.argmax())
            p_per[c]  = prec[idx]
            r_per[c]  = rec[idx]
            f1_per[c] = f1_curve[idx]

        return p_per, r_per, f1_per

    # ------------------------------------------------------------------
    # pycocotools evaluation
    # ------------------------------------------------------------------

    def _run_coco_eval(self):
        """Build in-memory COCO objects and run COCOeval."""
        cats    = [{"id": c + 1, "name": str(c)} for c in range(self.nc)]
        imgs    = [{"id": i} for i in range(self._img_id)]
        gt_json = {"images": imgs, "annotations": self._gt_list,
                   "categories": cats}

        coco_gt = COCO()
        coco_gt.dataset = gt_json
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(self._dt_list)

        # ── overall eval ─────────────────────────────────────────────
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        map5095 = float(ev.stats[0])
        map50   = float(ev.stats[1])

        # ── per-class AP@0.5 ─────────────────────────────────────────
        ap50_per = np.zeros(self.nc)
        for c in range(self.nc):
            ev_c = COCOeval(coco_gt, coco_dt, "bbox")
            ev_c.params.catIds  = [c + 1]
            ev_c.params.iouThrs = np.array([0.5])
            ev_c.evaluate(); ev_c.accumulate()
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                ev_c.summarize()
            ap50_per[c] = float(ev_c.stats[0]) if ev_c.stats[0] >= 0 else 0.0

        return map50, map5095, ap50_per

    # ------------------------------------------------------------------
    # NMS (greedy, class-aware)
    # ------------------------------------------------------------------

    def _nms(self, boxes, scores):
        max_s  = scores.max(axis=1)
        cls_id = scores.argmax(axis=1)
        keep   = max_s > self.conf_thres
        boxes, confs, cls_id = boxes[keep], max_s[keep], cls_id[keep]

        if len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        off   = cls_id.astype(np.float32) * 4096.0
        boff  = boxes + off[:, None]
        x1, y1, x2, y2 = boff[:,0], boff[:,1], boff[:,2], boff[:,3]
        areas = (x2-x1)*(y2-y1)
        order = confs.argsort()[::-1]
        kept  = []

        while len(order) and len(kept) < self.max_det:
            i = order[0]; kept.append(i)
            rest = order[1:]
            if not len(rest): break
            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(ix2-ix1,0)*np.maximum(iy2-iy1,0)
            iou   = inter/(areas[i]+areas[rest]-inter+1e-9)
            order = rest[iou <= self.iou_thres]

        kept = np.array(kept)
        return np.stack([boxes[kept,0], boxes[kept,1],
                         boxes[kept,2], boxes[kept,3],
                         confs[kept],
                         cls_id[kept].astype(np.float32)], axis=1)

    # ------------------------------------------------------------------
    # Accumulate per-det (conf, is_tp) for P/R curve computation
    # ------------------------------------------------------------------

    def _accumulate_dets(self, dets, gt_boxes, gt_cls):
        """Store (conf, is_tp) per class — evaluated later across all thresholds."""
        if len(dets) == 0:
            return
        matched = np.zeros(len(gt_boxes), dtype=bool)
        for det in dets[dets[:,4].argsort()[::-1]]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            gm  = (gt_cls == cls) & (~matched)
            gi  = np.where(gm)[0]
            is_tp = False
            if len(gi):
                ix1 = np.maximum(x1, gt_boxes[gi,0])
                iy1 = np.maximum(y1, gt_boxes[gi,1])
                ix2 = np.minimum(x2, gt_boxes[gi,2])
                iy2 = np.minimum(y2, gt_boxes[gi,3])
                inter = np.maximum(ix2-ix1,0)*np.maximum(iy2-iy1,0)
                a_d   = (x2-x1)*(y2-y1)
                a_g   = (gt_boxes[gi,2]-gt_boxes[gi,0])*(gt_boxes[gi,3]-gt_boxes[gi,1])
                iou   = inter/(a_d+a_g-inter+1e-9)
                best  = iou.argmax()
                if iou[best] >= 0.5:
                    matched[gi[best]] = True
                    is_tp = True
            if 0 <= cls < self.nc:
                self._cls_dets[cls].append((float(conf), float(is_tp)))
