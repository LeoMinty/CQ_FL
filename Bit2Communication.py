"""Packed 2-bit messages for the legacy BitFL federated boundary.

The original BitFL layers already implement local complex weight/gradient
phase quantization, but the legacy FL demos upload FP32 master weights.  This
module supplies the missing message codec used once per federated round:

* complex updates: four-axis phase code plus one FP32 tensor scale;
* real updates: the real-axis subset of the same phase alphabet
  (positive, zero, negative) plus one FP32 tensor scale.

The real codec deliberately occupies two bits per scalar even though one code
is unused.  This keeps one fixed-width 2-bit protocol for every trainable
tensor and gives an auditable 16x principal payload reduction from FP32.  The
functions really pack and unpack four codes per byte before reconstruction;
the returned byte count is therefore the serialized logical payload, not a
bit-width estimate applied to an uncompressed update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


COMPLEX_PHASE = "complex_phase"
REAL_AXIS = "real_axis"
PROTOCOL_VERSION = "cqfl_uplink_v2_real_axis_tensor_fp32_scale"


def _require_finite(values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError("2-bit updates must contain only finite values")


def pack_two_bit(codes: np.ndarray) -> np.ndarray:
    """Pack four values from ``[0, 3]`` into each uint8 byte."""
    raw = np.asarray(codes)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("2-bit codes must use an integer dtype")
    if np.any(raw < 0) or np.any(raw > 3):
        raise ValueError("2-bit codes must lie in [0, 3]")
    flat = raw.astype(np.uint8, copy=False).reshape(-1)
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return (
        flat[0::4]
        | (flat[1::4] << 2)
        | (flat[2::4] << 4)
        | (flat[3::4] << 6)
    )


def unpack_two_bit(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack exactly ``count`` values from a uint8 2-bit buffer."""
    if count < 0:
        raise ValueError("count must be non-negative")
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if packed.size * 4 < count:
        raise ValueError("packed buffer is too short for the requested count")
    output = np.empty(packed.size * 4, dtype=np.uint8)
    output[0::4] = packed & 0x03
    output[1::4] = (packed >> 2) & 0x03
    output[2::4] = (packed >> 4) & 0x03
    output[3::4] = (packed >> 6) & 0x03
    return output[:count]


def phase_codes_np(values: np.ndarray) -> np.ndarray:
    """Map ``[..., real, imag]`` pairs to ``+R,+I,-R,-I`` codes 0..3."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("complex phase coding expects a final [real, imag] axis")
    real, imag = values[..., 0], values[..., 1]
    angles = np.arctan2(imag, real).astype(np.float32)
    quarter_pi = np.float32(np.pi / 4.0)
    codes = np.full(real.shape, 3, dtype=np.uint8)
    codes[(angles >= -quarter_pi) & (angles < quarter_pi)] = 0
    codes[(angles >= quarter_pi) & (angles < 3.0 * quarter_pi)] = 1
    codes[(angles >= 3.0 * quarter_pi) | (angles < -3.0 * quarter_pi)] = 2
    return codes


@dataclass(frozen=True)
class Packed2BitTensor:
    packed: np.ndarray
    scale: np.float32
    shape: Tuple[int, ...]
    count: int
    scheme: str

    @property
    def nbytes(self) -> int:
        return int(self.packed.nbytes + np.asarray(self.scale).nbytes)


def encode_complex_update(values: np.ndarray) -> Packed2BitTensor:
    """Encode one complex client update with legacy four-axis semantics."""
    values = np.asarray(values, dtype=np.float32)
    _require_finite(values)
    codes = phase_codes_np(values)
    magnitudes = np.hypot(
        values[..., 0].astype(np.float64),
        values[..., 1].astype(np.float64),
    )
    scale_value = np.mean(magnitudes, dtype=np.float64) if magnitudes.size else 0.0
    if scale_value > np.finfo(np.float32).max:
        raise OverflowError("complex 2-bit update scale exceeds FP32 range")
    scale = np.float32(scale_value)
    return Packed2BitTensor(
        packed=pack_two_bit(codes),
        scale=scale,
        shape=tuple(values.shape),
        count=int(codes.size),
        scheme=COMPLEX_PHASE,
    )


def encode_real_update(values: np.ndarray) -> Packed2BitTensor:
    """Encode a real tensor using the real-axis subset of the phase alphabet.

    Code 0 is ``+scale``, code 2 is ``-scale``, and codes 1/3 decode to zero.
    Exact zeros use code 1.  Non-zero values retain their sign and share the
    tensor's FP32 mean-absolute-value scale, matching the global-scale style of
    the complex phase message.
    """
    values = np.asarray(values, dtype=np.float32)
    _require_finite(values)
    flat = values.reshape(-1)
    codes = np.full(flat.shape, 1, dtype=np.uint8)
    codes[flat > 0.0] = 0
    codes[flat < 0.0] = 2
    scale = np.float32(
        np.mean(np.abs(flat), dtype=np.float64) if flat.size else 0.0
    )
    return Packed2BitTensor(
        packed=pack_two_bit(codes),
        scale=scale,
        shape=tuple(values.shape),
        count=int(flat.size),
        scheme=REAL_AXIS,
    )


def decode_update(message: Packed2BitTensor) -> np.ndarray:
    """Decode a packed message to the FP32 tensor used by server aggregation."""
    codes = unpack_two_bit(message.packed, message.count)
    scale = np.float32(message.scale)
    if not np.isfinite(scale):
        raise ValueError("2-bit update scale must be finite")
    if message.scheme == COMPLEX_PHASE:
        real = (
            (codes == 0).astype(np.float32)
            - (codes == 2).astype(np.float32)
        ) * scale
        imag = (
            (codes == 1).astype(np.float32)
            - (codes == 3).astype(np.float32)
        ) * scale
        return np.stack([real, imag], axis=-1).reshape(message.shape)
    if message.scheme == REAL_AXIS:
        output = (
            (codes == 0).astype(np.float32)
            - (codes == 2).astype(np.float32)
        ) * scale
        return output.reshape(message.shape)
    raise ValueError(f"unknown 2-bit message scheme: {message.scheme}")


def quantize_complex_update_np(values: np.ndarray):
    """Encode, physically unpack, and reconstruct one complex update."""
    message = encode_complex_update(values)
    return decode_update(message), message.nbytes


def quantize_real_update_np(values: np.ndarray):
    """Encode, physically unpack, and reconstruct one real update."""
    message = encode_real_update(values)
    return decode_update(message), message.nbytes
