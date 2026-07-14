"""
diagnose_data_voc.py  —  VOC data pipeline visual + numerical sanity check.

Checks:
  1. Train batch (mosaic + augmentation): image range, box range, labels
  2. Val batch (letterbox only): same checks
  3. Saves annotated images so you can visually confirm boxes align
  4. Prints per-class distribution over N batches
  5. Flags degenerate boxes (x1>=x2, y1>=y2, out-of-range labels)

Run from PASCAL_VOC/ directory:
    python diagnose_data_voc.py

Output images → PASCAL_VOC/diagnose_out/
"""

import os, sys
from pathlib import Path
import numpy as np
import cv2

# ── paths ──────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "diagnose_out")
os.makedirs(OUT_DIR, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "kitti"

# Suppress TF noise
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"]         = "--tf_xla_enable_xla_devices=false"

import tensorflow as tf
tf.config.optimizer.set_jit(False)

sys.path.insert(0, _HERE)
from voc_cfg import TRAIN_CFG, VOC_CATEGORIES
from voc_loader import VOCLoader

# ── config ──────────────────────────────────────────────────────────────────
TRAIN_DIR = str(DATASET_ROOT / "images" / "train")
VAL_DIR   = str(DATASET_ROOT / "images" / "val")

IMGSZ      = TRAIN_CFG["imgsz"]   # 640
BATCH      = 4                     # small for speed; increase to check more
N_STAT_BATCHES = 20               # batches to collect class stats

NC    = 8
NAMES = VOC_CATEGORIES

# Distinct colours per class (BGR for OpenCV)
_PALETTE = [
    (230,  25,  75), (60, 180,  75), (255, 225,  25), ( 0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]

# ── helpers ─────────────────────────────────────────────────────────────────

def draw_boxes(img_float, gt_bboxes, gt_labels, mask_gt, imgsz):
    """
    Draw GT boxes on a float32 RGB [H,W,3] image.
    Returns uint8 BGR image for cv2.imwrite.
    """
    img = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for j in range(len(mask_gt)):
        if not mask_gt[j]:
            break
        x1n, y1n, x2n, y2n = gt_bboxes[j]
        cls = int(gt_labels[j])

        x1 = int(x1n * imgsz); y1 = int(y1n * imgsz)
        x2 = int(x2n * imgsz); y2 = int(y2n * imgsz)

        color = _PALETTE[cls % len(_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{NAMES[cls] if 0 <= cls < NC else '?'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def check_batch(batch_np, split_name, save_prefix):
    """
    Numerical checks + save annotated images for one batch dict.
    Returns per-class GT counts for that batch.
    """
    imgs     = batch_np["images"]       # [B, H, W, 3]
    bboxes   = batch_np["gt_bboxes"]    # [B, max_gt, 4]
    labels   = batch_np["gt_labels"]    # [B, max_gt]
    masks    = batch_np["mask_gt"]      # [B, max_gt]

    B        = imgs.shape[0]
    cls_counts = np.zeros(NC, dtype=np.int64)

    print(f"\n{'─'*60}")
    print(f"  [{split_name}]  batch shape: images={imgs.shape}")
    print(f"{'─'*60}")

    # ── image range ──────────────────────────────────────────────────────
    img_min = imgs.min(); img_max = imgs.max()
    img_ok  = (img_min >= 0.0) and (img_max <= 1.0)
    print(f"  image  range : {img_min:.4f} → {img_max:.4f}   "
          f"{'✓' if img_ok else '✗ EXPECTED 0-1'}")
    print(f"  image  dtype : {imgs.dtype}   {'✓' if imgs.dtype == np.float32 else '✗ EXPECTED float32'}")
    print(f"  image  shape : {imgs.shape}  {'✓' if list(imgs.shape[1:]) == [IMGSZ, IMGSZ, 3] else '✗'}")

    # ── box range ─────────────────────────────────────────────────────────
    valid_boxes = bboxes[masks]          # [N_valid, 4]
    valid_labels = labels[masks]         # [N_valid]

    if len(valid_boxes) == 0:
        print("  ✗ ZERO valid GT boxes in this batch!")
        return cls_counts

    box_min = valid_boxes.min(); box_max = valid_boxes.max()
    box_ok  = (box_min >= 0.0) and (box_max <= 1.0)
    print(f"\n  gt_bboxes range: {box_min:.4f} → {box_max:.4f}   "
          f"{'✓' if box_ok else '✗ EXPECTED 0-1'}")

    # x1<x2, y1<y2
    degen_wh = ((valid_boxes[:,2] - valid_boxes[:,0]) <= 0) | \
               ((valid_boxes[:,3] - valid_boxes[:,1]) <= 0)
    if degen_wh.any():
        print(f"  ✗ {degen_wh.sum()} degenerate boxes (x1>=x2 or y1>=y2)!")
    else:
        print(f"  box geometry: x1<x2, y1<y2  ✓")

    # ── label range ───────────────────────────────────────────────────────
    lbl_min = valid_labels.min(); lbl_max = valid_labels.max()
    lbl_ok  = (lbl_min >= 0) and (lbl_max < NC)
    print(f"  gt_labels range: {lbl_min} → {lbl_max}   "
          f"{'✓ (expected 0-' + str(NC-1) + ')' if lbl_ok else '✗ OUT OF RANGE'}")

    # ── per-image GT counts ────────────────────────────────────────────────
    counts = masks.sum(axis=1)
    print(f"\n  GT per image  : min={counts.min()}  max={counts.max()}  "
          f"mean={counts.mean():.1f}")
    print(f"  Zero-GT images: {(counts == 0).sum()} / {B}")

    # ── class distribution ─────────────────────────────────────────────────
    for c in valid_labels:
        if 0 <= c < NC:
            cls_counts[c] += 1

    # ── box size distribution ──────────────────────────────────────────────
    w = valid_boxes[:,2] - valid_boxes[:,0]
    h = valid_boxes[:,3] - valid_boxes[:,1]
    area = w * h
    tiny  = (area < 0.01).sum()     # < 10% of image side
    small = ((area >= 0.01) & (area < 0.04)).sum()
    med   = ((area >= 0.04) & (area < 0.16)).sum()
    large = (area >= 0.16).sum()
    print(f"\n  Box sizes (normalized area):")
    print(f"    tiny  (<1%):  {tiny:4d}   small (1-4%): {small:4d}")
    print(f"    medium(4-16%):{med:4d}   large  (>16%):{large:4d}")

    # ── save annotated images ─────────────────────────────────────────────
    for i in range(min(B, 4)):
        vis = draw_boxes(imgs[i], bboxes[i], labels[i], masks[i], IMGSZ)
        out_path = os.path.join(OUT_DIR, f"{save_prefix}_img{i:02d}.jpg")
        cv2.imwrite(out_path, vis)

    print(f"\n  Saved {min(B,4)} annotated images → {OUT_DIR}/{save_prefix}_img*.jpg")
    return cls_counts


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print(f"VOC DATA DIAGNOSTIC")
    print(f"  IMGSZ     : {IMGSZ}")
    print(f"  BATCH     : {BATCH}")
    print(f"  NC        : {NC}")
    print(f"  TRAIN_DIR : {TRAIN_DIR}")
    print(f"  VAL_DIR   : {VAL_DIR}")
    print(f"  OUT_DIR   : {OUT_DIR}")

    # ── 1. Train loader (mosaic + augmentation) ──────────────────────────
    print("\n\n══════════════════════════════════════")
    print("  1. TRAIN LOADER (mosaic=True, augment=True)")
    print("══════════════════════════════════════")

    train_loader = VOCLoader(
        image_dir=TRAIN_DIR,
        imgsz=IMGSZ,
        batch_size=BATCH,
        cfg=TRAIN_CFG,
        augment=True,
        mosaic=True,
    )

    ds_train = train_loader.build_dataset()
    batch_train = next(iter(ds_train))
    batch_np = {k: v.numpy() for k, v in batch_train.items()}
    total_cls = check_batch(batch_np, "TRAIN (mosaic)", "train_mosaic")

    # Second train batch — no mosaic (close-mosaic phase simulation)
    print("\n── Train loader (mosaic=False, augment=True) ──")
    train_loader.mosaic = False
    batch_train2 = next(iter(train_loader.build_dataset()))
    batch_np2 = {k: v.numpy() for k, v in batch_train2.items()}
    check_batch(batch_np2, "TRAIN (no mosaic)", "train_nomosaic")
    train_loader.mosaic = True   # restore

    # ── 2. Val loader (letterbox only) ────────────────────────────────────
    print("\n\n══════════════════════════════════════")
    print("  2. VAL LOADER (mosaic=False, augment=False)")
    print("══════════════════════════════════════")

    val_loader = VOCLoader(
        image_dir=VAL_DIR,
        imgsz=IMGSZ,
        batch_size=BATCH,
        cfg=TRAIN_CFG,
        augment=False,
        mosaic=False,
    )

    ds_val = val_loader.build_dataset()
    batch_val = next(iter(ds_val))
    batch_np_val = {k: v.numpy() for k, v in batch_val.items()}
    check_batch(batch_np_val, "VAL (letterbox)", "val_letterbox")

    # ── 3. Class distribution over N_STAT_BATCHES ─────────────────────────
    print(f"\n\n══════════════════════════════════════")
    print(f"  3. CLASS DISTRIBUTION ({N_STAT_BATCHES} train batches)")
    print(f"══════════════════════════════════════")

    cls_total = np.zeros(NC, dtype=np.int64)
    ds_stat   = train_loader.build_dataset()
    for i, batch in enumerate(ds_stat):
        if i >= N_STAT_BATCHES:
            break
        masks  = batch["mask_gt"].numpy()
        labels = batch["gt_labels"].numpy()
        valid  = labels[masks]
        for c in valid:
            if 0 <= c < NC:
                cls_total[c] += 1

    total_gt = cls_total.sum()
    print(f"\n  Total GT instances: {total_gt}  "
          f"(from {N_STAT_BATCHES} batches × {BATCH} images = {N_STAT_BATCHES*BATCH} images)")
    print(f"\n  {'class':<22} {'count':>6}  {'%':>6}  {'bar'}")
    print(f"  {'─'*55}")
    max_c = cls_total.max()
    for c in range(NC):
        bar = '█' * int(20 * cls_total[c] / max(max_c, 1))
        print(f"  {NAMES[c]:<22} {cls_total[c]:>6}  "
              f"{100*cls_total[c]/max(total_gt,1):>5.1f}%  {bar}")

    zero_cls = [NAMES[c] for c in range(NC) if cls_total[c] == 0]
    if zero_cls:
        print(f"\n  ✗ Classes with 0 GT in sample: {zero_cls}")
    else:
        print(f"\n  ✓ All {NC} classes present in sample")

    # ── 4. Label–image alignment spot check ──────────────────────────────
    print(f"\n\n══════════════════════════════════════")
    print(f"  4. LABEL–IMAGE ALIGNMENT (train, 4 samples with single crop)")
    print(f"══════════════════════════════════════")

    loader_single = VOCLoader(
        image_dir=TRAIN_DIR,
        imgsz=IMGSZ,
        batch_size=BATCH,
        cfg=TRAIN_CFG,
        augment=False,   # no aug → boxes should fit tightly on letterboxed image
        mosaic=False,
    )
    batch_single = next(iter(loader_single.build_dataset()))
    batch_np_single = {k: v.numpy() for k, v in batch_single.items()}
    check_batch(batch_np_single, "TRAIN (letterbox, no aug)", "train_letterbox_noaug")

    print(f"\n{'═'*60}")
    print(f"  DONE.  Check {OUT_DIR}/ for annotated images.")
    print(f"  If boxes are aligned with objects → data pipeline is correct.")
    print(f"  If boxes are shifted/scaled/swapped → there is a bug.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
