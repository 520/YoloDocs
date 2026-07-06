"""YOLOv8 architecture configuration definitions.

This module defines the official YOLOv8 backbone / neck / head layout
using cfg-style block descriptions. The variant scaling parameters are
kept separate from the block definitions so the same architecture can
be instantiated for n, s, m, l, and x variants.
"""

YOLOV8_VARIANTS = {
    "n": {
        "depth_mult": 0.33,
        "width_mult": 0.25,
        "max_channels": 1024,
        "head_channels": 80,
        "intermediate_reg_max": 16,
    },
    "s": {
        "depth_mult": 0.33,
        "width_mult": 0.50,
        "max_channels": 1024,
        "head_channels": 128,
        "intermediate_reg_max": 16,
    },
    "m": {
        "depth_mult": 0.67,
        "width_mult": 0.75,
        "max_channels": 768,
        "head_channels": 192,
        "intermediate_reg_max": 16,
    },
    "l": {
        "depth_mult": 1.00,
        "width_mult": 1.00,
        "max_channels": 512,
        "head_channels": 256,
        "intermediate_reg_max": 16,
    },
    "x": {
        "depth_mult": 1.00,
        "width_mult": 1.25,
        "max_channels": 512,
        "head_channels": 320,
        "intermediate_reg_max": 20,
    },
}

# Each layer configuration uses the format:
# [from, repeats, module_name, args]
# - from: index or list of indices relative to layers_out
# - repeats: number of repeated modules (scaled by depth_mult)
# - module_name: one of Conv, C2f, SPPF, UpSample, Concat, Detect
# - args: module-specific arguments
BACKBONE_CFG = [
    [-1, 1, "Conv", [64, 3, 2]],
    [-1, 1, "Conv", [128, 3, 2]],
    [-1, 3, "C2f", [128, True]],
    [-1, 1, "Conv", [256, 3, 2]],
    [-1, 6, "C2f", [256, True]],
    [-1, 1, "Conv", [512, 3, 2]],
    [-1, 6, "C2f", [512, True]],
    [-1, 1, "Conv", [1024, 3, 2]],
    [-1, 3, "C2f", [1024, True]],
    [-1, 1, "SPPF", [1024, 5]],
]

NECK_CFG = [
    [-1, 1, "UpSample", [2]],
    [[-1, 7], 1, "Concat", []],
    [-1, 3, "C2f", [512, False]],
    [-1, 1, "UpSample", [2]],
    [[-1, 5], 1, "Concat", []],
    [-1, 3, "C2f", [256, False]],
    [-1, 1, "Conv", [256, 3, 2]],
    [[-1, 13], 1, "Concat", []],
    [-1, 3, "C2f", [512, False]],
    [-1, 1, "Conv", [512, 3, 2]],
    [[-1, 10], 1, "Concat", []],
    [-1, 3, "C2f", [1024, False]],
]

HEAD_CFG = {
    "from": [16, 19, 22],
    "module": "Detect",
    "args": [],
}
