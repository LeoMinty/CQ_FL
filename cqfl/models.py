"""Unified experiment model built from the original BitFL layers.

The experiment runner intentionally imports the existing repository modules
instead of maintaining a second ComplexConv implementation.  Consequently the
quantized methods use the original four-axis kernel and
``simplified_complex_matmul`` path from ``BitMyConv_noMul.py``.
"""

from __future__ import annotations

import tensorflow as tf

from Bit2Conv import ComplexMaxPool2D
from Bit2Linear import RealToComplex, TakeReal
from BitMyConv_noMul import ComplexBatchNormalization, ComplexConv2D


def build_model(
    input_shape,
    num_classes: int,
    method: str,
    model_profile: str = "standard",
):
    """Build one common architecture for all experiment-one methods.

    ``w2_fp32_adam`` and ``cqfl`` share the exact same 2-bit forward operator.
    Their local training-core differences are the backward quantization switch
    and the optimizer; the federated runner additionally gives CQ-FL the packed
    2-bit client-update boundary.  FP32 and SignSGD use the same complex
    architecture with the weight quantizer disabled, keeping parameter layouts
    comparable.
    """
    if method not in {"fedavg_fp32", "bitfl", "signsgd", "w2_fp32_adam", "cqfl"}:
        raise ValueError(f"unknown method: {method}")
    if model_profile == "standard":
        convolution_filters = (16, 32, 64)
        dense_units = (256, 128)
    elif model_profile in {"mnist_small", "dronerf_small"}:
        convolution_filters = (8, 16)
        dense_units = (64,)
    else:
        raise ValueError(f"unknown model profile: {model_profile}")

    quantized_weight = method in {"w2_fp32_adam", "cqfl"}
    quantized_gradient = method == "cqfl"

    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = inputs
    if len(input_shape) != 4 or input_shape[-1] != 2:
        x = RealToComplex(name="real_to_complex")(x)

    for filters in convolution_filters:
        x = ComplexConv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            use_quant=quantized_weight,
            quantize_backward=quantized_gradient,
            quant_weight_decay=0.01 if quantized_weight else 0.0,
            name=f"bitfl_complex_conv_{filters}",
        )(x)
        x = ComplexBatchNormalization(name=f"bitfl_complex_bn_{filters}")(x)
        x = ComplexMaxPool2D(pool_size=(2, 2), name=f"bitfl_complex_pool_{filters}")(x)

    x = TakeReal(name="take_real")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    for units in dense_units:
        x = tf.keras.layers.Dense(
            units, activation="relu", name=f"dense_{units}"
        )(x)
    outputs = tf.keras.layers.Dense(num_classes, name="logits")(x)
    model = tf.keras.Model(
        inputs, outputs, name=f"bitfl_{method}_{model_profile}"
    )
    model.bitfl_method = method
    model.bitfl_model_profile = model_profile
    return model
