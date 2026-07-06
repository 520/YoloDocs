"""
YOLOv8 PASCAL VOC data loader — mirrors data/coco_loader.py exactly.

Reads directly from:
    {image_dir}/*.jpg
    {label_dir}/*.txt   (YOLO format: cls cx cy w h, normalised)

where label_dir is derived from image_dir by replacing /images/ with /labels/.

Augmentation pipeline is identical to COCOLoader:
    Training (mosaic=True):
        mosaic4 → random_affine → HSV → flip
    Training (mosaic=False, close-mosaic phase):
        letterbox → random_affine → HSV → flip
    Validation:
        letterbox only

Output batch dict (same keys as COCOLoader):
    images    [B, imgsz, imgsz, 3]  float32  RGB  0-1
    gt_bboxes [B, max_gt, 4]        float32  xyxy normalised 0-1
    gt_labels [B, max_gt]           int32    (-1 = padding)
    mask_gt   [B, max_gt]           bool
"""

import os
import cv2
import random
import numpy as np
import tensorflow as tf


class VOCLoader:
    """
    PASCAL VOC loader using YOLO-format label files.

    Parameters
    ----------
    image_dir  : str   — full path to images directory (e.g. .../images/train)
    imgsz      : int   — output image size (square), default 640
    batch_size : int
    cfg        : dict  — TRAIN_CFG (reads augmentation params)
    augment    : bool  — master switch; False → letterbox only (val)
    mosaic     : bool  — enable mosaic; set False for close-mosaic phase
    max_gt     : int   — max GT boxes per image (padding target)
    """

    def __init__(
        self,
        image_dir,
        imgsz=640,
        batch_size=16,
        cfg=None,
        augment=True,
        mosaic=True,
        max_gt=100,
    ):
        self.imgsz      = imgsz
        self.batch_size = batch_size
        self.augment    = augment
        self.mosaic     = mosaic
        self.max_gt     = max_gt

        cfg = cfg or {}
        self.mosaic_prob     = float(cfg.get("mosaic",      1.0))
        self.mixup_prob      = float(cfg.get("mixup",       0.0))
        self.hsv_h           = float(cfg.get("hsv_h",       0.015))
        self.hsv_s           = float(cfg.get("hsv_s",       0.7))
        self.hsv_v           = float(cfg.get("hsv_v",       0.4))
        self.degrees         = float(cfg.get("degrees",     0.0))
        self.translate       = float(cfg.get("translate",   0.1))
        self.scale           = float(cfg.get("scale",       0.5))
        self.shear           = float(cfg.get("shear",       0.0))
        self.perspective     = float(cfg.get("perspective", 0.0))
        self.fliplr          = float(cfg.get("fliplr",      0.5))
        self.flipud          = float(cfg.get("flipud",      0.0))

        self.img_dir   = image_dir
        self.label_dir = image_dir.replace("/images/", "/labels/")

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"[VOCLoader] {len(self.img_files)} images from {self.img_dir}")

    # ------------------------------------------------------------------
    # Single image loader
    # ------------------------------------------------------------------

    def _load_image_label(self, idx):
        """
        Load one image and its YOLO-format labels.
        Returns:
            img    : uint8 RGB [H, W, 3]
            labels : float32 [N, 5]  (cls, cx, cy, w, h) normalised
            (h, w) : original image shape
        """
        img_path   = os.path.join(self.img_dir,   self.img_files[idx])
        label_path = os.path.join(self.label_dir,
                                  os.path.splitext(self.img_files[idx])[0] + '.txt')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([float(x) for x in parts])

        if labels:
            return img, np.array(labels, dtype=np.float32), (h, w)
        return img, np.zeros((0, 5), dtype=np.float32), (h, w)

    # ------------------------------------------------------------------
    # Letterbox
    # ------------------------------------------------------------------

    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        """Aspect-ratio-preserving resize + grey padding. Matches Ultralytics."""
        shape = img.shape[:2]   # (H, W)
        r     = min(new_shape / shape[0], new_shape / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (W, H)
        dw = new_shape - new_unpad[0]
        dh = new_shape - new_unpad[1]
        dw /= 2;  dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top    = int(round(dh - 0.1));  bottom = int(round(dh + 0.1))
        left   = int(round(dw - 0.1));  right  = int(round(dw + 0.1))
        img    = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    # ------------------------------------------------------------------
    # Mosaic
    # ------------------------------------------------------------------

    def _load_mosaic(self, index):
        """
        4-image mosaic. Matches Ultralytics load_mosaic() exactly.
        Uses original image dimensions (no pre-resize).
        Returns: img [2s, 2s, 3] uint8, labels [N, 5] (cls cx cy w h) pixels
        """
        s  = self.imgsz
        yc = int(random.uniform(s // 2, 2 * s - s // 2))
        xc = int(random.uniform(s // 2, 2 * s - s // 2))

        indices = [index] + random.choices(range(len(self.img_files)), k=3)

        mosaic_img    = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []

        for i, idx in enumerate(indices):
            img, labels, (h, w) = self._load_image_label(idx)

            if   i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc-w, 0), max(yc-h, 0), xc, yc
                x1b, y1b, x2b, y2b = w-(x2a-x1a), h-(y2a-y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc-h, 0), min(xc+w, s*2), yc
                x1b, y1b, x2b, y2b = 0, h-(y2a-y1a), min(w, x2a-x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc-w, 0), yc, xc, min(s*2, yc+h)
                x1b, y1b, x2b, y2b = w-(x2a-x1a), 0, w, min(y2a-y1a, h)
            else:          # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc+w, s*2), min(s*2, yc+h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a-x1a), min(y2a-y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(labels):
                # Convert YOLO norm → pixel on original image → shift to canvas
                l = labels.copy()
                l[:, 1] = (labels[:, 1] * w) + padw  # cx
                l[:, 2] = (labels[:, 2] * h) + padh  # cy
                l[:, 3] =  labels[:, 3] * w           # bw
                l[:, 4] =  labels[:, 4] * h           # bh
                mosaic_labels.append(l)

        if mosaic_labels:
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 1:], 0, 2 * s, out=mosaic_labels[:, 1:])
        else:
            mosaic_labels = np.zeros((0, 5), dtype=np.float32)

        return mosaic_img, mosaic_labels   # labels: pixel cxcywh on 2s canvas

    # ------------------------------------------------------------------
    # HSV augmentation
    # ------------------------------------------------------------------

    def _augment_hsv(self, img):
        """Random HSV jitter. img: uint8 RGB."""
        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        x = np.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_h),
                             cv2.LUT(sat, lut_s),
                             cv2.LUT(val, lut_v)))
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # ------------------------------------------------------------------
    # Random affine
    # ------------------------------------------------------------------

    def _random_affine(self, img, labels, border=(0, 0)):
        """
        Combined affine: scale + translate + shear + perspective.
        Matches Ultralytics random_perspective() with border trick.
        labels: [N, 5] (cls, cx, cy, w, h) in pixels on current canvas.
        Returns: img, labels in same format.
        """
        import math
        height = img.shape[0] + border[0] * 2
        width  = img.shape[1] + border[1] * 2

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)

        # Rotation + scale
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        # Center — shift image origin to its center before scale/rotate/shear
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2
        C[1, 2] = -img.shape[0] / 2

        M = T @ S @ R @ P @ C
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height),
                                          flags=cv2.INTER_LINEAR,
                                          borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=(width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=(114, 114, 114))

        # Transform boxes (cxcywh → corners → transform → new xyxy)
        n = len(labels)
        if n:
            xy = np.ones((n * 4, 3))
            cx = labels[:, 1]; cy = labels[:, 2]
            bw = labels[:, 3]; bh = labels[:, 4]
            x1 = cx - bw/2;  x2 = cx + bw/2
            y1 = cy - bh/2;  y2 = cy + bh/2
            xy[:, :2] = np.stack([x1,y1, x2,y1, x1,y2, x2,y2], 1
                                  ).reshape(n*4, 2)
            xy = xy @ M.T
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
            else:
                xy = xy[:, :2].reshape(n, 8)

            x  = xy[:, [0,2,4,6]]
            y  = xy[:, [1,3,5,7]]
            x1n = x.min(1); x2n = x.max(1)
            y1n = y.min(1); y2n = y.max(1)

            x1n = np.clip(x1n, 0, width);  x2n = np.clip(x2n, 0, width)
            y1n = np.clip(y1n, 0, height); y2n = np.clip(y2n, 0, height)

            w_new = x2n - x1n;  h_new = y2n - y1n
            w_old = labels[:, 3]; h_old = labels[:, 4]
            ar    = np.maximum(w_new/(h_new+1e-9), h_new/(w_new+1e-9))
            area_ratio = (w_new*h_new) / (w_old*h_old*s*s + 1e-9)
            keep  = (w_new > 2) & (h_new > 2) & (area_ratio > 0.1) & (ar < 100)

            labels_out = np.zeros((keep.sum(), 5), dtype=np.float32)
            labels_out[:, 0] = labels[keep, 0]
            labels_out[:, 1] = (x1n[keep] + x2n[keep]) / 2
            labels_out[:, 2] = (y1n[keep] + y2n[keep]) / 2
            labels_out[:, 3] = x2n[keep] - x1n[keep]
            labels_out[:, 4] = y2n[keep] - y1n[keep]
            labels = labels_out

        return img, labels

    # ------------------------------------------------------------------
    # Convert labels to our loss format
    # ------------------------------------------------------------------

    def _to_xyxy_norm(self, labels_cxcywh_px, imgsz):
        """
        Convert [N, 5] (cls, cx, cy, w, h) pixel → [N, 4] xyxy normalised 0-1
        and [N] int class labels.
        """
        if len(labels_cxcywh_px) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)
        s = float(imgsz)
        cls  = labels_cxcywh_px[:, 0].astype(np.int32)
        cx   = labels_cxcywh_px[:, 1] / s
        cy   = labels_cxcywh_px[:, 2] / s
        bw   = labels_cxcywh_px[:, 3] / s
        bh   = labels_cxcywh_px[:, 4] / s
        x1   = np.clip(cx - bw/2, 0, 1)
        y1   = np.clip(cy - bh/2, 0, 1)
        x2   = np.clip(cx + bw/2, 0, 1)
        y2   = np.clip(cy + bh/2, 0, 1)
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        keep  = ((x2-x1) > 2/s) & ((y2-y1) > 2/s)
        return boxes[keep], cls[keep]

    # ------------------------------------------------------------------
    # Full sample pipeline
    # ------------------------------------------------------------------

    def _process_sample(self, idx):
        """
        Full augmentation pipeline for one sample.
        Returns: (img_rgb_float [s,s,3], boxes_norm [N,4], labels [N])
        """
        s = self.imgsz

        if self.mosaic and random.random() < self.mosaic_prob:
            # ── Mosaic path ───────────────────────────────────────────
            img, labels = self._load_mosaic(idx)
            img, labels = self._random_affine(
                img, labels,
                border=(-s // 2, -s // 2))

        else:
            # ── Single image path ─────────────────────────────────────
            img_raw, labels_raw, (h, w) = self._load_image_label(idx)
            img, r, (dw, dh) = self._letterbox(img_raw, new_shape=s)

            labels = labels_raw.copy()
            if len(labels):
                labels[:, 1] = labels_raw[:, 1] * w * r + dw   # cx
                labels[:, 2] = labels_raw[:, 2] * h * r + dh   # cy
                labels[:, 3] = labels_raw[:, 3] * w * r         # bw
                labels[:, 4] = labels_raw[:, 4] * h * r         # bh

            if self.augment:
                img, labels = self._random_affine(img, labels, border=(0, 0))

        # ── Per-image augmentation ────────────────────────────────────
        if self.augment:
            img = self._augment_hsv(img)

            if random.random() < self.fliplr:
                img = img[:, ::-1]
                if len(labels):
                    labels[:, 1] = s - labels[:, 1]

            if random.random() < self.flipud:
                img = img[::-1]
                if len(labels):
                    labels[:, 2] = s - labels[:, 2]

        # ── Convert to output format ──────────────────────────────────
        img_float = img.astype(np.float32) / 255.0
        img_float = np.ascontiguousarray(img_float)
        boxes, cls = self._to_xyxy_norm(labels, s)
        return img_float, boxes, cls.astype(np.int32)

    # ------------------------------------------------------------------
    # tf.data wrapper
    # ------------------------------------------------------------------

    def _tf_wrapper(self, idx_tensor):
        idx = int(idx_tensor.numpy())
        img, boxes, labels = self._process_sample(idx)

        n  = min(len(boxes), self.max_gt)
        mg = self.max_gt
        bp = np.zeros((mg, 4),  dtype=np.float32)
        lp = np.full( (mg,),  -1, dtype=np.int32)
        mp = np.zeros((mg,),      dtype=bool)
        if n > 0:
            bp[:n] = boxes[:n]
            lp[:n] = labels[:n]
            mp[:n] = True

        return (img.astype(np.float32),
                bp.astype(np.float32),
                lp.astype(np.int32),
                mp)

    def _tf_load(self, idx):
        s  = self.imgsz
        mg = self.max_gt
        img, boxes, labels, mask = tf.py_function(
            func=self._tf_wrapper,
            inp=[idx],
            Tout=[tf.float32, tf.float32, tf.int32, tf.bool],
        )
        img.set_shape([s, s, 3])
        boxes.set_shape([mg, 4])
        labels.set_shape([mg])
        mask.set_shape([mg])
        return img, boxes, labels, mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dataset(self):
        """
        Returns tf.data.Dataset yielding dicts:
            images    [B, imgsz, imgsz, 3]  float32  RGB  0-1
            gt_bboxes [B, max_gt, 4]        float32  xyxy normalised
            gt_labels [B, max_gt]           int32    (-1 = padding)
            mask_gt   [B, max_gt]           bool
        """
        n        = len(self.img_files)
        autotune = tf.data.AUTOTUNE

        ds = tf.data.Dataset.range(n)

        if self.augment and n > 0:
            ds = ds.shuffle(n, reshuffle_each_iteration=True)

        ds = (
            ds
            .map(self._tf_load, num_parallel_calls=autotune)
            .batch(self.batch_size, drop_remainder=True)
            .map(lambda img, b, l, m: {
                "images":    img,
                "gt_bboxes": b,
                "gt_labels": l,
                "mask_gt":   m,
            })
            .prefetch(autotune)
        )
        return ds

    def steps_per_epoch(self):
        return len(self.img_files) // self.batch_size

    def __len__(self):
        return len(self.img_files)
