"""
YOLOv8 PASCAL VOC training config.

Mirrors yolov8_architecture/yolov8_cfg.py with VOC-specific settings:
  - nc=20, 20 VOC class names
  - cos_lr=False  (linear schedule — Ultralytics default for VOC)
  - momentum=0.937  (Ultralytics "MuSGD" default)
  - epochs=300  (from scratch)
  - imgsz=640  (YOLOv8 default; change to 416 for ~2.4x faster training)
"""

# VOC_CATEGORIES = [
#     "aeroplane", "bicycle", "bird", "boat", "bottle",
#     "bus", "car", "cat", "chair", "cow",
#     "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor",
# ]
VOC_CATEGORIES = [
    "car",
    "van",
    "truck",
    "pedestrian",
    "Person_sitting",
    "cyclist",
    "tram",
    "misc",
]

TRAIN_CFG = {
    # ---- optimizer ----
    "optimizer":        "SGD",
    "lr0":              0.01,       # SGD default
    # "optimizer":        "Adam",     # coupled WD (wd adapted by 2nd moment)
    # "optimizer":        "AdamW",    # decoupled WD (true AdamW, recommended)
    # "lr0":              0.001,      # Adam/AdamW from scratch
    # "lr0":              0.0001,     # Adam/AdamW fine-tune (pretrained backbone)
    "lrf":              0.01,       # final LR = lr0 * lrf  → 1e-5
    "momentum":         0.937,      # beta_1 for Adam (momentum for SGD)
    "weight_decay":     0.0005,
    "nesterov":         True,       # ignored by Adam

    # ---- warmup ----
    "warmup_epochs":    3.0,
    "warmup_momentum":  0.8,
    "warmup_bias_lr":   0.1,

    # ---- schedule ----
    "epochs":           300,
    "cos_lr":           False,      # linear LR
    "close_mosaic":     10,         # disable mosaic for last N epochs

    # ---- freeze (pretrained fine-tuning) ----
    # Epochs to freeze backbone+neck layers so only the head adapts first.
    # Prevents high early-epoch LR from destroying pretrained features.
    # Default 0 = disabled. Override to 5 in main_voc.py for large/x + pretrained.
    "freeze_epochs":    0,

    # ---- loss weights (same as COCO) ----
    "box":              7.5,
    "cls":              0.5,
    "dfl":              1.5,

    # ---- loss / assigner ----
    "nbs":              64,
    "tal_topk":         10,
    "tal_alpha":        0.5,
    "tal_beta":         6.0,
    "reg_max":          16,
    "tal_min_anchor_size":    8.0,
    "tal_expand_anchor_size": 16.0,

    # ---- augmentation (same as COCO) ----
    "hsv_h":            0.015,
    "hsv_s":            0.7,
    "hsv_v":            0.4,
    "degrees":          0.0,
    "translate":        0.1,
    "scale":            0.5,
    "shear":            0.0,
    "perspective":      0.0,
    "flipud":           0.0,
    "fliplr":           0.5,
    "mosaic":           1.0,
    "mixup":            0.0,
    "copy_paste":       0.0,

    # ---- data ----
    "imgsz":            640,        # change to 416 for faster training
    "batch":            16,
    "workers":          8,

    # ---- misc ----
    "amp":              True,
    # EMA decay — half-life = log(0.5)/log(decay) steps.
    # 0.9990 ≈ half-life 693 steps (~0.7 ep) — fast tracking, good for small models
    #          and from-scratch where the cls head needs rapid adaptation.
    # 0.9998 ≈ half-life 3466 steps (~3.5 ep) — smoother, better for large pretrained
    #          models where features are stable and EMA should not chase noise.
    # Default here is for nano/small from-scratch. Override in main_voc.py for large.
    "ema_decay":        0.9990,
    # "ema_decay":      0.9998,  # for large/x pretrained fine-tuning (set in main_voc.py)
    "seed":             0,

    # ---- GPU VRAM ----
    "gpu_memory":       12000,      # MB
}
