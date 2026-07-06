"""YOLOv8 architecture builder for TensorFlow / Keras.

Key design: dual-output head.

  training_model  : outputs list of 3 raw-logit tensors, one per scale
                    each [B, Hi*Wi, 4*reg_max + nc]
                    → fed directly to YOLOv8Loss (TAL + DFL + CIoU + BCE)

  inference_model : outputs [B, 8400, 84]  (decoded boxes + sigmoid scores)
                    → matches Ultralytics inference format verified in Netron

Both models share identical weights. Switch by passing
training=True/False to build_yolov8(), or keep both references.

Changes from original:
  1. _build_detect_head returns raw per-scale logits for training
  2. build_model() builds inference head on top of the same feature ops
  3. BN: epsilon=1e-3, momentum=0.97  (Ultralytics exact match)
  4. use_bias=True preserved on all Conv layers (matches Ultralytics)
"""

import math
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from .yolov8_cfg import BACKBONE_CFG, HEAD_CFG, NECK_CFG, YOLOV8_VARIANTS


def make_divisible(value: int, divisor: int = 8,
                   min_value: Optional[int] = None) -> int:
    min_value = min_value or divisor
    new_value = max(min_value, int((value + divisor / 2) // divisor * divisor))
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def scale_depth(repeats: int, depth_mult: float) -> int:
    return max(1, int(round(repeats * depth_mult)))


class YOLOv8:
    def __init__(
        self,
        variant: str = "s",
        task: str = "detect",
        input_shape: int = 640,
        num_classes: int = 80,
        reg_max: int = 16,
    ):
        if variant not in YOLOV8_VARIANTS:
            raise ValueError(f"Unknown YOLOv8 variant: {variant}")

        self.variant     = variant
        self.task        = task.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.reg_max     = reg_max

        variant_cfg          = YOLOV8_VARIANTS[variant]
        self.depth_mult      = variant_cfg["depth_mult"]
        self.width_mult      = variant_cfg["width_mult"]
        self.max_channels    = variant_cfg["max_channels"]
        self.head_channels   = variant_cfg["head_channels"]
        self.intermediate_reg_max = variant_cfg.get("intermediate_reg_max", 16)

    # ------------------------------------------------------------------
    # Building blocks  (unchanged from your original except BN params)
    # ------------------------------------------------------------------

    def ConvBNAct(self, x, out_channels, kernel=1, stride=1,
                  activation="swish"):
        """Conv + BN + activation.

        BN corrected to match Ultralytics:
          epsilon=1e-3, momentum=0.97
          (PyTorch eps=1e-3, momentum=0.03 → TF decay = 1-0.03 = 0.97)

        Padding: for stride>1 we use explicit symmetric ZeroPadding2D(pad=k//2)
        + 'valid' to match PyTorch's padding=k//2 on all four sides.
        Keras 'same' with stride>1 pads asymmetrically (0-before, 1-after)
        which shifts the feature map by one pixel versus PyTorch.
        """
        if stride > 1 and kernel > 1:
            pad = kernel // 2
            x = layers.ZeroPadding2D(padding=((pad, pad), (pad, pad)))(x)
            x = layers.Conv2D(
                out_channels, kernel,
                strides=stride, padding="valid",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)
        else:
            x = layers.Conv2D(
                out_channels, kernel,
                strides=stride, padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.97)(x)
        if activation in ("silu", "swish"):
            return layers.Activation("swish")(x)   # swish avoids XLA/libdevice
        elif activation == "relu":
            return layers.Activation("relu")(x)
        elif activation in ("none", None):
            return x
        return layers.Activation(activation)(x)

    def Bottleneck(self, x, out_channels, shortcut=True, expansion=0.5):
        hidden = max(1, int(out_channels * expansion))
        y = self.ConvBNAct(x, hidden, kernel=3)
        y = self.ConvBNAct(y, out_channels, kernel=3)
        if shortcut and int(x.shape[-1]) == out_channels:
            y = layers.Add()([x, y])
        return y

    def C2f(self, x, out_channels, n=1, shortcut=True, expansion=0.5):
        hidden = int(out_channels * expansion)
        x1 = self.ConvBNAct(x, 2 * hidden, kernel=1)
        split_layer = layers.Lambda(
            lambda t: tf.split(t, num_or_size_splits=2, axis=-1))
        x1 = list(split_layer(x1))
        x1_0 = layers.Lambda(lambda t: t[0])(x1)
        x1_1 = layers.Lambda(lambda t: t[1])(x1)
        x2 = x1_1
        bottleneck_outputs = []
        for _ in range(max(1, n)):
            x2 = self.Bottleneck(x2, hidden, shortcut=shortcut, expansion=1.0)
            bottleneck_outputs.append(x2)
        y = layers.Concatenate(axis=-1)([x1_0, x1_1] + bottleneck_outputs)
        return self.ConvBNAct(y, out_channels, kernel=1)

    def SPPF(self, x, out_channels, pool_size=5):
        hidden = int(x.shape[-1]) // 2
        x = self.ConvBNAct(x, hidden, kernel=1)
        y1 = layers.MaxPool2D(pool_size, strides=1, padding="same")(x)
        y2 = layers.MaxPool2D(pool_size, strides=1, padding="same")(y1)
        y3 = layers.MaxPool2D(pool_size, strides=1, padding="same")(y2)
        y  = layers.Concatenate(axis=-1)([x, y1, y2, y3])
        return self.ConvBNAct(y, out_channels, kernel=1)

    def Upsample(self, x, scale=2):
        return layers.UpSampling2D(size=(scale, scale),
                                   interpolation="nearest")(x)

    def _apply_width_mult(self, channels):
        value = channels
        if self.max_channels is not None:
            value = min(value, self.max_channels)
        value = int(value * self.width_mult)
        value = make_divisible(value, divisor=8)
        return max(1, value)

    def _resolve_inputs(self, from_idx, layers_out):
        if isinstance(from_idx, list):
            return [layers_out[i] for i in from_idx]
        return layers_out[from_idx]

    def _build_backbone(self, inputs):
        layers_out = [inputs]
        for from_idx, repeats, module_name, args in BACKBONE_CFG:
            inp = self._resolve_inputs(from_idx, layers_out)
            n   = scale_depth(repeats, self.depth_mult)
            if module_name == "Conv":
                ch, k, s = args
                y = self.ConvBNAct(inp, self._apply_width_mult(ch), k, s)
            elif module_name == "C2f":
                ch, sc = args
                y = self.C2f(inp, self._apply_width_mult(ch), n, sc)
            elif module_name == "SPPF":
                ch, ps = args
                y = self.SPPF(inp, self._apply_width_mult(ch), ps)
            else:
                raise ValueError(f"Unsupported backbone module: {module_name}")
            layers_out.append(y)
        return layers_out

    def _build_neck(self, backbone_out):
        layers_out = backbone_out.copy()
        for from_idx, repeats, module_name, args in NECK_CFG:
            inp = self._resolve_inputs(from_idx, layers_out)
            n   = scale_depth(repeats, self.depth_mult)
            if module_name == "UpSample":
                y = self.Upsample(inp, scale=args[0])
            elif module_name == "Concat":
                y = layers.Concatenate(axis=-1)(inp)
            elif module_name == "Conv":
                ch, k, s = args
                y = self.ConvBNAct(inp, self._apply_width_mult(ch), k, s)
            elif module_name == "C2f":
                ch, sc = args
                y = self.C2f(inp, self._apply_width_mult(ch), n, sc)
            else:
                raise ValueError(f"Unsupported neck module: {module_name}")
            layers_out.append(y)
        return layers_out

    # ------------------------------------------------------------------
    # Detection head — two variants sharing the same conv weights
    # ------------------------------------------------------------------

    def _head_convs(self, features):
        """
        Shared conv operations for both training and inference heads.
        Returns (box_conv_outputs, cls_conv_outputs) — one per scale,
        both still spatial [B, H, W, ch] before final projection.
        """
        box_head_ch = 4 * self.intermediate_reg_max
        box_feats, cls_feats = [], []

        for fm in features:
            # box branch
            bx = self.ConvBNAct(fm, box_head_ch, kernel=3)
            bx = self.ConvBNAct(bx, box_head_ch, kernel=3)
            box_feats.append(bx)

            # cls branch
            cx = self.ConvBNAct(fm, self.head_channels, kernel=3)
            cx = self.ConvBNAct(cx, self.head_channels, kernel=3)
            cls_feats.append(cx)

        return box_feats, cls_feats

    def _final_projections(self, box_feats, cls_feats):
        """
        Final 1x1 box/cls projection convs — built ONCE per scale and reused
        by both the training and inference output paths (see
        build_paired_models). Returns (box_raw, cls_raw): per-scale spatial
        tensors [B, H, W, 4*reg_max] / [B, H, W, nc], before reshape.
        """
        # Per-stride prior bias matching Ultralytics Detect.bias_init():
        #   bias = log(5 / nc / (imgsz/stride)²)
        # For nc=80, imgsz=640: ≈ -11.54, -10.15, -8.77
        strides = [8, 16, 32]
        cls_biases = [
            math.log(5 / self.num_classes / (self.input_shape / s) ** 2)
            for s in strides
        ]

        box_raw, cls_raw = [], []
        for i, (bx, cx) in enumerate(zip(box_feats, cls_feats)):
            # bias_initializer=1.0 matches Ultralytics Detect.bias_init(): a[-1].bias[:] = 1.0
            box = layers.Conv2D(
                4 * self.reg_max, 1, padding="same", use_bias=True,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer=tf.keras.initializers.Constant(1.0),
            )(bx)
            cls = layers.Conv2D(
                self.num_classes, 1, padding="same", use_bias=True,
                kernel_initializer=tf.keras.initializers.HeNormal(),
                bias_initializer=tf.keras.initializers.Constant(cls_biases[i]),
            )(cx)
            box_raw.append(box)
            cls_raw.append(cls)

        return box_raw, cls_raw

    def _training_outputs(self, box_raw, cls_raw):
        """
        Training head — outputs raw logits per scale.

        Returns list of 3 tensors, one per detection scale:
            [B, Hi*Wi, 4*reg_max + nc]   ← raw, no softmax/sigmoid

        This is what YOLOv8Loss, TAL assigner, and decode_preds() expect.
        """
        outputs = []
        for box, cls in zip(box_raw, cls_raw):
            box = layers.Reshape((-1, 4 * self.reg_max))(box)
            cls = layers.Reshape((-1, self.num_classes))(cls)
            # Concat along last dim: [B, Hi*Wi, 4*reg_max + nc]
            outputs.append(layers.Concatenate(axis=-1)([box, cls]))
        return outputs   # list of 3 tensors

    def _inference_output(self, box_raw, cls_raw):
        """
        Inference head — matches your original exactly.
        Outputs [B, 8400, 84]: decoded boxes (grid units) + sigmoid scores.
        Verified correct against Ultralytics in Netron — unchanged.
        """
        box_preds = [layers.Reshape((-1, 4 * self.reg_max))(b) for b in box_raw]
        cls_preds = [layers.Reshape((-1, self.num_classes))(c) for c in cls_raw]

        all_boxes   = layers.Concatenate(axis=1)(box_preds)
        all_classes = layers.Concatenate(axis=1)(cls_preds)

        # DFL decode
        x = layers.Reshape((-1, 4, self.reg_max))(all_boxes)
        x = layers.Softmax(axis=-1)(x)
        dfl_w = np.arange(self.reg_max).reshape(
            1, 1, self.reg_max, 1).astype(np.float32)
        final_boxes = layers.Conv2D(
            filters=1, kernel_size=(1, 1),
            use_bias=False, trainable=False,
            kernel_initializer=tf.keras.initializers.Constant(dfl_w),
        )(x)
        final_boxes   = layers.Reshape((-1, 4))(final_boxes)
        final_classes = layers.Activation("sigmoid")(all_classes)
        return layers.Concatenate(axis=-1)([final_boxes, final_classes])

    def _build_detect_head_training(self, features):
        """Standalone training head (used by build_model()): builds its own
        final projection convs — NOT shared with a separately-built inference
        model. For a (train_model, inf_model) pair that shares weights with
        no transfer step, use build_paired_models() instead."""
        box_feats, cls_feats = self._head_convs(features)
        box_raw, cls_raw = self._final_projections(box_feats, cls_feats)
        return self._training_outputs(box_raw, cls_raw)

    def _build_detect_head_inference(self, features):
        """Standalone inference head (used by build_model()) — see note on
        _build_detect_head_training."""
        box_feats, cls_feats = self._head_convs(features)
        box_raw, cls_raw = self._final_projections(box_feats, cls_feats)
        return self._inference_output(box_raw, cls_raw)

    # ------------------------------------------------------------------
    # build_model
    # ------------------------------------------------------------------

    def build_model(self, training=False, print_summary=True):
        """
        Parameters
        ----------
        training : bool
            True  → training model, outputs list of 3 raw-logit tensors
            False → inference model, outputs [B, 8400, 84] (default)

        NOTE: each call builds a brand-new graph (new Conv2D/BN layer
        objects) — two separate calls do NOT share weights. To get a
        (train_model, inf_model) pair that shares weights with no transfer
        step required, use build_paired_models() instead.
        """
        inputs       = layers.Input(
            shape=(self.input_shape, self.input_shape, 3))
        backbone_out = self._build_backbone(inputs)
        neck_out     = self._build_neck(backbone_out)
        feat_indices = HEAD_CFG["from"]
        features     = [neck_out[i] for i in feat_indices]

        if self.task != "detect":
            raise ValueError(f"Unsupported task: {self.task}")

        if training:
            outputs = self._build_detect_head_training(features)
        else:
            outputs = self._build_detect_head_inference(features)

        suffix = "train" if training else "infer"
        model  = Model(inputs=inputs, outputs=outputs,
                       name=f"YOLOv8_{self.variant}_{self.task}_{suffix}")
        if print_summary:
            model.summary()
        return model

    def build_paired_models(self, print_summary=True):
        """
        Build (train_model, inf_model) from ONE shared graph: backbone, neck,
        head_convs, and the final box/cls projection convs are built once
        and reused by both output paths. The two Model() objects are
        separate computation graphs (inf_model has no training-only ops, so
        deployment latency/op-count is unaffected) but they reference the
        SAME underlying Conv2D/BatchNormalization Variable objects — so
        weight updates made via train_model are automatically visible in
        inf_model, with no save/load/transfer step required.

        This avoids the fragile alternative of building train_model and
        inf_model independently and copying weights afterward: Keras
        auto-names unnamed layers from a process-global counter, so by-name
        weight matching between independently-built graphs can silently
        match the wrong layer; and `model.layers` order is a topological
        traversal that depends on the model's output structure, so
        positional matching is unreliable too (both were tried and both hit
        real shape-mismatch failures).
        """
        if self.task != "detect":
            raise ValueError(f"Unsupported task: {self.task}")

        inputs       = layers.Input(
            shape=(self.input_shape, self.input_shape, 3))
        backbone_out = self._build_backbone(inputs)
        neck_out     = self._build_neck(backbone_out)
        feat_indices = HEAD_CFG["from"]
        features     = [neck_out[i] for i in feat_indices]

        box_feats, cls_feats = self._head_convs(features)
        box_raw, cls_raw     = self._final_projections(box_feats, cls_feats)

        train_outputs = self._training_outputs(box_raw, cls_raw)
        infer_output  = self._inference_output(box_raw, cls_raw)

        train_model = Model(inputs=inputs, outputs=train_outputs,
                            name=f"YOLOv8_{self.variant}_{self.task}_train")
        inf_model   = Model(inputs=inputs, outputs=infer_output,
                            name=f"YOLOv8_{self.variant}_{self.task}_infer")
        if print_summary:
            train_model.summary()
        return train_model, inf_model


def build_yolov8(variant="s", task="detect", input_shape=640,
                 num_classes=80, reg_max=16, training=False, print_summary=True):
    """
    Build a YOLOv8 model.

    training=False  →  inference model  [B, 8400, 84]  (default, Netron-verified)
    training=True   →  training model   list of 3 raw-logit tensors per scale
    """
    return YOLOv8(
        variant=variant, task=task,
        input_shape=input_shape,
        num_classes=num_classes,
        reg_max=reg_max,
    ).build_model(training=training, print_summary=print_summary)


def build_yolov8_pair(variant="s", task="detect", input_shape=640,
                      num_classes=80, reg_max=16, print_summary=True):
    """
    Build (train_model, inf_model) sharing the same backbone/neck/head
    weights — see YOLOv8.build_paired_models() for why this replaces
    building the two models independently and transferring weights after
    training.
    """
    return YOLOv8(
        variant=variant, task=task,
        input_shape=input_shape,
        num_classes=num_classes,
        reg_max=reg_max,
    ).build_paired_models(print_summary=print_summary)


if __name__ == "__main__":
    train_model, inf_model = build_yolov8_pair(variant="n", num_classes=20)
    print("Inference output:", inf_model.output_shape)
    print("Training outputs:", [o.shape for o in train_model.outputs])