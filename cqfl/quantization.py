"""2-bit message utilities compatible with the original BitFL phase codes.

TensorFlow stores the trainable shadow variable as FP32.  The functions below
simulate the paper's low-bit forward/communication representation and expose
the phase codes needed for exact communication accounting.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # Allows NumPy-side data preparation without TensorFlow.
    tf = None


def phase_codes_np(values: np.ndarray) -> np.ndarray:
    """Return codes 0:+R, 1:+I, 2:-R, 3:-I for [..., 2] values."""
    values = np.asarray(values)
    if values.shape[-1] != 2:
        raise ValueError("phase quantization expects a final [real, imag] axis")
    real, imag = values[..., 0], values[..., 1]
    angles = np.arctan2(imag, real).astype(np.float32)
    quarter_pi = np.float32(np.pi / 4.0)
    codes = np.full(real.shape, 3, dtype=np.uint8)
    codes[(angles >= -quarter_pi) & (angles < quarter_pi)] = 0
    codes[(angles >= quarter_pi) & (angles < 3.0 * quarter_pi)] = 1
    codes[(angles >= 3.0 * quarter_pi) | (angles < -3.0 * quarter_pi)] = 2
    return codes


def phase_dequantize_np(codes: np.ndarray, scale: float) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    real = ((codes == 0).astype(np.float32) - (codes == 2).astype(np.float32)) * scale
    imag = ((codes == 1).astype(np.float32) - (codes == 3).astype(np.float32)) * scale
    return np.stack([real, imag], axis=-1)


def phase_quantize_weight_np(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Original BitFL weight map: exact unit values in ``{+1,+i,-1,-i}``."""
    values = np.asarray(values, dtype=np.float32)
    codes = phase_codes_np(values)
    return phase_dequantize_np(codes, 1.0), codes


def phase_quantize_gradient_np(values: np.ndarray) -> Tuple[np.ndarray, np.float16, np.ndarray]:
    """Draft definition: 2-bit direction plus one FP16 global magnitude."""
    values = np.asarray(values, dtype=np.float32)
    codes = phase_codes_np(values)
    magnitudes = np.linalg.norm(values, axis=-1)
    scale = np.float16(np.mean(magnitudes, dtype=np.float64) if magnitudes.size else 0.0)
    return phase_dequantize_np(codes, float(scale)), scale, codes


def _require_tf() -> None:
    if tf is None:
        raise RuntimeError("TensorFlow is required for model training")


def _phase_masks_tf(values):
    real, imag = values[..., 0], values[..., 1]
    quarter_pi = tf.cast(np.pi / 4.0, values.dtype)
    angles = tf.math.atan2(imag, real)
    pos_real = tf.logical_and(angles >= -quarter_pi, angles < quarter_pi)
    pos_imag = tf.logical_and(angles >= quarter_pi, angles < 3.0 * quarter_pi)
    neg_real = tf.logical_or(angles >= 3.0 * quarter_pi, angles < -3.0 * quarter_pi)
    neg_imag = tf.logical_and(angles >= -3.0 * quarter_pi, angles < -quarter_pi)
    return real, imag, pos_real, pos_imag, neg_real, neg_imag


def phase_quantize_unit_tf(values):
    """TensorFlow unit-axis map used to quantize non-kernel complex gradients."""
    _require_tf()
    _real, _imag, pr, pi, nr, ni = _phase_masks_tf(values)
    q_real = tf.cast(pr, values.dtype) - tf.cast(nr, values.dtype)
    q_imag = tf.cast(pi, values.dtype) - tf.cast(ni, values.dtype)
    return tf.stack([q_real, q_imag], axis=-1)


def quantize_complex_delta_np(values: np.ndarray) -> Tuple[np.ndarray, int]:
    """Quantize an uploaded complex update and return reconstructed values/bytes."""
    reconstructed, _scale, codes = phase_quantize_gradient_np(values)
    # Four codes per byte plus one FP16 scale.
    payload_bytes = (codes.size + 3) // 4 + 2
    return reconstructed, payload_bytes
