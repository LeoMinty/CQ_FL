"""Simplified BitFL uplink codec from Li et al., Computer Networks 2026.

This implements the BitFL efficiency baseline: tensor updates are
normalized to [-normalization_bound, normalization_bound], stochastically encoded to
one unbiased bit per scalar, physically packed, and decoded before aggregation.
Optional independent bit flips provide the paper's simplified HLDP perturbation
for sensitivity experiments; the default remains the non-private baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


LEGACY_PROTOCOL_VERSION = "bitfl_v1_stochastic_1bit_tensor_fp32_scale_topk50_ef"
PROTOCOL_VERSION = (
    "bitfl_v2_stochastic_1bit_tensor_fp32_scale_configurable_hldp_topk_ef"
)


def pack_one_bit(bits: np.ndarray) -> np.ndarray:
    raw = np.asarray(bits)
    if not np.issubdtype(raw.dtype, np.integer) and raw.dtype != np.bool_:
        raise TypeError("1-bit values must use an integer or boolean dtype")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("1-bit values must lie in {0, 1}")
    return np.packbits(raw.astype(np.uint8, copy=False).reshape(-1), bitorder="little")


def unpack_one_bit(packed: np.ndarray, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if packed.size * 8 < count:
        raise ValueError("packed buffer is too short for the requested count")
    return np.unpackbits(packed, bitorder="little")[:count]


@dataclass(frozen=True)
class PackedBitFLTensor:
    packed: np.ndarray
    scale: np.float32
    shape: Tuple[int, ...]
    count: int
    normalization_bound: float

    @property
    def nbytes(self) -> int:
        return int(self.packed.nbytes + np.asarray(self.scale).nbytes)


def encode_update(
    values: np.ndarray,
    rng: np.random.Generator,
    normalization_bound: float = 1.0,
    bit_flip_probability: float = 0.0,
) -> PackedBitFLTensor:
    """Apply Eq. (5) stochastic one-bit quantization to one tensor update."""
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("BitFL updates must contain only finite values")
    if not 0.0 < normalization_bound <= 1.0:
        raise ValueError("normalization_bound must lie in (0, 1]")
    if not 0.0 <= bit_flip_probability <= 0.5:
        raise ValueError("bit_flip_probability must lie in [0, 0.5]")
    flat = values.reshape(-1)
    maximum = float(np.max(np.abs(flat))) if flat.size else 0.0
    if maximum == 0.0:
        scale = np.float32(0.0)
        bits = np.zeros(flat.size, dtype=np.uint8)
    else:
        # normalized lies in [-normalization_bound, normalization_bound].  The
        # decoded magnitude cancels this factor, preserving E[decoded]=values.
        normalized = flat * np.float32(normalization_bound / maximum)
        probability_positive = (normalized + np.float32(1.0)) * np.float32(0.5)
        bits = (rng.random(flat.size) < probability_positive).astype(np.uint8)
        scale = np.float32(maximum / normalization_bound)
    if bit_flip_probability > 0.0 and bits.size:
        flips = rng.random(bits.size) < bit_flip_probability
        bits = np.bitwise_xor(bits, flips.astype(np.uint8))
    return PackedBitFLTensor(
        packed=pack_one_bit(bits),
        scale=scale,
        shape=tuple(values.shape),
        count=int(flat.size),
        normalization_bound=float(normalization_bound),
    )


def decode_update(message: PackedBitFLTensor) -> np.ndarray:
    bits = unpack_one_bit(message.packed, message.count)
    signs = bits.astype(np.float32) * np.float32(2.0) - np.float32(1.0)
    if message.scale == 0.0:
        signs.fill(0.0)
    return (signs * np.float32(message.scale)).reshape(message.shape)


def quantize_update_np(
    values: np.ndarray,
    rng: np.random.Generator,
    normalization_bound: float = 1.0,
    bit_flip_probability: float = 0.0,
):
    message = encode_update(
        values, rng, normalization_bound, bit_flip_probability
    )
    return decode_update(message), message.nbytes


def topk_error_feedback(
    aggregated: Sequence[np.ndarray],
    residual: Sequence[np.ndarray],
    fraction: float = 0.5,
):
    """Algorithm 3 top-k selection with cross-round error accumulation."""
    if len(aggregated) != len(residual):
        raise ValueError("BitFL update and residual layouts differ")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top-k fraction must lie in (0, 1]")
    shapes = [np.asarray(value).shape for value in aggregated]
    combined = np.concatenate(
        [
            (np.asarray(value, np.float32) + np.asarray(error, np.float32)).reshape(-1)
            for value, error in zip(aggregated, residual)
        ]
    )
    total = int(combined.size)
    keep = max(1, int(np.ceil(total * fraction))) if total else 0
    selected = np.zeros(total, dtype=bool)
    if keep >= total:
        selected[:] = True
    elif keep:
        indexes = np.argpartition(np.abs(combined), total - keep)[total - keep :]
        selected[indexes] = True
    applied_flat = np.where(selected, combined, np.float32(0.0))
    residual_flat = combined - applied_flat

    applied, next_residual = [], []
    offset = 0
    for shape in shapes:
        count = int(np.prod(shape))
        applied.append(applied_flat[offset : offset + count].reshape(shape))
        next_residual.append(residual_flat[offset : offset + count].reshape(shape))
        offset += count
    return applied, next_residual
