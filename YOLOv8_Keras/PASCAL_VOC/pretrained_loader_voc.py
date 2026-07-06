"""
pretrained_loader_voc.py

Transfer Ultralytics YOLOv8 pretrained COCO weights (nc=80) into a VOC
model (nc=20).  Logic is identical to ../pretrained_loader.py except for
one difference in the final-layer transfer:

  6 biased head projections in Keras topological order:
      [0] box_s8   [1] cls_s8
      [2] box_s16  [3] cls_s16
      [4] box_s32  [5] cls_s32

  Box finals (even indices): 4*reg_max = 64 filters — same regardless of nc.
      → TRANSFERRED.
  Cls finals (odd indices): nc filters — COCO=80, VOC=20 → shape mismatch.
      → SKIPPED (Keras layer keeps its VOC cls-bias init).

All backbone / neck / head-intermediate weights are transferred verbatim.
"""

import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO

_DETECT_IDX = 22


# ---------------------------------------------------------------------------
# Step 1: Extract PyTorch weights (identical to parent pretrained_loader.py)
# ---------------------------------------------------------------------------

def _collect_pt(pt_model, variant):
    """
    Extract Conv2d and BN weights in Keras topological order.

    Returns:
        bb_convs : list of ndarray [O,I,H,W]  — backbone+neck, topological order
        bb_bns   : list of (gamma, beta, mean, var)
        head     : dict 'cv2'/'cv3' → list of 3 stride dicts
    """
    from yolov8_architecture.yolov8_cfg import (
        BACKBONE_CFG, NECK_CFG, YOLOV8_VARIANTS)

    depth_mult = YOLOV8_VARIANTS[variant]["depth_mult"]
    mod = {name: m for name, m in pt_model.named_modules()}

    def _conv(path):
        return mod[path].weight.detach().float().numpy()

    def _bn(path):
        m = mod[path]
        return (m.weight.detach().float().numpy(),
                m.bias.detach().float().numpy(),
                m.running_mean.detach().float().numpy(),
                m.running_var.detach().float().numpy())

    bb_convs, bb_bns = [], []

    backbone_neck = (
        [(i, cfg)      for i, cfg in enumerate(BACKBONE_CFG)] +
        [(10 + i, cfg) for i, cfg in enumerate(NECK_CFG)]
    )

    for pt_idx, block_cfg in backbone_neck:
        _, repeats, module_name, _ = block_cfg
        base = f"model.{pt_idx}"

        if module_name == 'Conv':
            bb_convs.append(_conv(f"{base}.conv"))
            bb_bns.append(_bn(f"{base}.bn"))

        elif module_name == 'C2f':
            n = max(round(repeats * depth_mult), 1)
            bb_convs.append(_conv(f"{base}.cv1.conv"))
            bb_bns.append(_bn(f"{base}.cv1.bn"))
            for k in range(n):
                bb_convs.append(_conv(f"{base}.m.{k}.cv1.conv"))
                bb_bns.append(_bn(f"{base}.m.{k}.cv1.bn"))
                bb_convs.append(_conv(f"{base}.m.{k}.cv2.conv"))
                bb_bns.append(_bn(f"{base}.m.{k}.cv2.bn"))
            bb_convs.append(_conv(f"{base}.cv2.conv"))
            bb_bns.append(_bn(f"{base}.cv2.bn"))

        elif module_name == 'SPPF':
            bb_convs.append(_conv(f"{base}.cv1.conv"))
            bb_bns.append(_bn(f"{base}.cv1.bn"))
            bb_convs.append(_conv(f"{base}.cv2.conv"))
            bb_bns.append(_bn(f"{base}.cv2.bn"))

        # UpSample / Concat: no weights — skip

    head = {
        'cv2': [{} for _ in range(3)],
        'cv3': [{} for _ in range(3)],
    }
    for branch in ('cv2', 'cv3'):
        for s in range(3):
            base = f"model.{_DETECT_IDX}.{branch}.{s}"
            tgt = head[branch][s]
            tgt['c1']      = _conv(f"{base}.0.conv")
            tgt['bn1']     = _bn(f"{base}.0.bn")
            tgt['c2']      = _conv(f"{base}.1.conv")
            tgt['bn2']     = _bn(f"{base}.1.bn")
            m_fin          = mod[f"{base}.2"]
            tgt['final_w'] = m_fin.weight.detach().float().numpy()
            tgt['final_b'] = (m_fin.bias.detach().float().numpy()
                              if m_fin.bias is not None else None)

    return bb_convs, bb_bns, head


# ---------------------------------------------------------------------------
# Step 2: Collect Keras layers
# ---------------------------------------------------------------------------

def _collect_keras(keras_model):
    k_no_bias, k_with_bias, k_bns = [], [], []
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.trainable:
            (k_with_bias if layer.use_bias else k_no_bias).append(layer)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            k_bns.append(layer)
    return k_no_bias, k_with_bias, k_bns


# ---------------------------------------------------------------------------
# Step 3: Build reordered PT lists that match Keras layer order
# ---------------------------------------------------------------------------

def _build_pt_lists(bb_convs, bb_bns, head):
    nb_convs = list(bb_convs)
    nb_bns   = list(bb_bns)

    # Head intermediates c1: per stride, box then cls
    for s in range(3):
        nb_convs.append(head['cv2'][s]['c1'])
        nb_convs.append(head['cv3'][s]['c1'])
        nb_bns.append(head['cv2'][s]['bn1'])
        nb_bns.append(head['cv3'][s]['bn1'])

    # Head intermediates c2
    for s in range(3):
        nb_convs.append(head['cv2'][s]['c2'])
        nb_convs.append(head['cv3'][s]['c2'])
        nb_bns.append(head['cv2'][s]['bn2'])
        nb_bns.append(head['cv3'][s]['bn2'])

    # Head finals: [box_s8, cls_s8, box_s16, cls_s16, box_s32, cls_s32]
    wb = []
    for s in range(3):
        wb.append((head['cv2'][s]['final_w'], head['cv2'][s]['final_b']))
        wb.append((head['cv3'][s]['final_w'], head['cv3'][s]['final_b']))

    return nb_convs, nb_bns, wb


# ---------------------------------------------------------------------------
# Step 4: Transfer weights — skip cls finals (nc mismatch)
# ---------------------------------------------------------------------------

def _transfer_partial(nb_convs, nb_bns, wb,
                      k_no_bias, k_with_bias, k_bns, verbose):
    """
    Transfer all weights except the cls head finals.

    wb / k_with_bias order: [box_s8, cls_s8, box_s16, cls_s16, box_s32, cls_s32]
    Even indices (0,2,4) = box finals  → 4*reg_max=64 filters, nc-independent → TRANSFER
    Odd  indices (1,3,5) = cls finals  → nc=80 (COCO) vs nc=20 (VOC) → SKIP
    """
    def _check(label, pt_list, k_list):
        if len(pt_list) != len(k_list):
            raise ValueError(
                f"{label} count mismatch — PyTorch: {len(pt_list)}, Keras: {len(k_list)}."
            )

    _check("Conv (no-bias)", nb_convs, k_no_bias)
    _check("Conv (bias)",    wb,       k_with_bias)
    _check("BatchNorm",      nb_bns,   k_bns)

    # No-bias convs: backbone + neck + head intermediates
    for i, (pt_w, k_layer) in enumerate(zip(nb_convs, k_no_bias)):
        keras_w = np.transpose(pt_w, (2, 3, 1, 0))
        k_layer.set_weights([keras_w])
        if verbose and i < 3:
            print(f"  conv[{i:3d}]  PT {list(pt_w.shape)} → Keras {list(keras_w.shape)}")
    if verbose:
        print(f"  ... ({len(nb_convs)} no-bias convs transferred)")

    # With-bias convs: head finals — skip cls (odd indices)
    transferred_finals = 0
    for i, ((pt_w, pt_b), k_layer) in enumerate(zip(wb, k_with_bias)):
        if i % 2 == 1:
            pt_nc   = pt_w.shape[0]
            voc_nc  = k_layer.bias.shape[0]
            if verbose:
                print(f"  final[{i}] SKIP  cls projection  "
                      f"PT nc={pt_nc} vs VOC nc={voc_nc}  layer={k_layer.name}")
            continue
        keras_w = np.transpose(pt_w, (2, 3, 1, 0))
        weights = [keras_w, pt_b] if pt_b is not None else [keras_w]
        k_layer.set_weights(weights)
        transferred_finals += 1
        if verbose:
            bias_str = f"  bias={list(pt_b.shape)}" if pt_b is not None else ""
            print(f"  final[{i}]  {list(keras_w.shape)}{bias_str}  layer={k_layer.name}")

    if verbose:
        print(f"  ... ({transferred_finals}/6 finals transferred; 3 cls skipped)")

    # BN
    for i, ((g, b, m, v), k_layer) in enumerate(zip(nb_bns, k_bns)):
        k_layer.set_weights([g, b, m, v])
    if verbose:
        print(f"  ... ({len(nb_bns)} BN layers transferred)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ultralytics_weights_voc(keras_model, variant="n",
                                  pt_path=None, verbose=True):
    """
    Load Ultralytics yolov8{variant}.pt (COCO, nc=80) into a VOC model (nc=20).

    Backbone + neck + head intermediate convs + box final convs are transferred.
    Cls final convs are skipped (nc mismatch: 80 vs 20); they retain the
    VOC-specific cls bias initialisation: log(5 / 20 / (imgsz/stride)^2).

    Args:
        keras_model : Keras training model (nc=20, built with training=True)
        variant     : "n", "s", "m", "l", or "x"
        pt_path     : path to local .pt file (if None, downloads automatically)
        verbose     : print per-layer info
    """
    pt_filename = pt_path or f"yolov8{variant}.pt"
    print(f"\n── Loading Ultralytics yolov8{variant}.pt → VOC model ──")
    print(f"   Source: {pt_filename}")
    print(f"   Backbone + neck + head intermediates + box finals: TRANSFERRED")
    print(f"   Cls finals (nc=80→nc=20): SKIPPED (keep VOC cls-bias init)")

    yolo     = YOLO(pt_filename)
    pt_model = yolo.model.eval()

    print("  Extracting PyTorch weights ...")
    bb_convs, bb_bns, head = _collect_pt(pt_model, variant)
    print(f"  PT backbone+neck: {len(bb_convs)} convs, {len(bb_bns)} BNs")

    nb_convs, nb_bns, wb = _build_pt_lists(bb_convs, bb_bns, head)
    k_no_bias, k_with_bias, k_bns = _collect_keras(keras_model)
    print(f"  Keras: {len(k_no_bias)} no-bias convs, "
          f"{len(k_with_bias)} bias convs, {len(k_bns)} BNs")

    print("\n  Transferring weights ...")
    _transfer_partial(nb_convs, nb_bns, wb, k_no_bias, k_with_bias, k_bns, verbose)

    print(f"\n  ✓ Partial transfer complete: backbone+neck+head_intermediate+box_finals loaded")
    return keras_model
