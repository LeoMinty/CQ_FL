"""Draft-aligned Complex-Aware 4-bit Adam.

Persistent state layout
-----------------------
* complex first moment: 2-bit phase + 4-bit block-quantized magnitude;
* complex second moment: component-wise Adam values with the original parameter
  shape, encoded with the zero-excluding 4-bit linear map;
* real-valued parameters: 4-bit signed first moment and zero-excluding 4-bit
  second moment;
* one FP16 scale per block of 64 values.

Only one parameter tensor is decompressed at a time.  This is an eager-mode
research implementation: it favours a transparent state format over speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .quantization import phase_codes_np

try:
    import tensorflow as tf
except ImportError:
    tf = None


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    return flat[0::2] | (flat[1::2] << 4)


def unpack_nibbles(packed: np.ndarray, count: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = packed & 0x0F
    out[1::2] = packed >> 4
    return out[:count]


def pack_two_bit(codes: np.ndarray) -> np.ndarray:
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)


def unpack_two_bit(packed: np.ndarray, count: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.empty(packed.size * 4, dtype=np.uint8)
    out[0::4] = packed & 0x03
    out[1::4] = (packed >> 2) & 0x03
    out[2::4] = (packed >> 4) & 0x03
    out[3::4] = (packed >> 6) & 0x03
    return out[:count]


def _unsigned_de4_codebook() -> np.ndarray:
    """Build the 4-bit dynamic-exponent mapping described in Appendix E.2."""
    values = [0.0]
    bits = 4
    for exponent_zeros in range(bits):
        fraction_bits = bits - exponent_zeros - 1
        bins = 2**fraction_bits
        edges = np.linspace(0.1, 1.0, bins + 1, dtype=np.float64)
        fractions = (edges[:-1] + edges[1:]) / 2.0
        values.extend((10.0 ** (-exponent_zeros) * fractions).tolist())
    values = np.asarray(sorted(values), dtype=np.float32)
    values /= values[-1]
    return values


DE4_UNSIGNED = _unsigned_de4_codebook()
# Four bits in the signed case leave three magnitude bits.  The duplicated zero
# is a legal unused code and keeps the packed representation exactly 4-bit.
_mag3 = DE4_UNSIGNED[
    np.rint(np.linspace(0, len(DE4_UNSIGNED) - 1, 8)).astype(np.int64)
]
_signed = np.concatenate([-_mag3[:0:-1], [0.0], _mag3[1:]])
DE4_SIGNED = np.insert(_signed, len(_signed) // 2, 0.0).astype(np.float32)[:16]
LINEAR4_NO_ZERO = (np.arange(1, 17, dtype=np.float32) / 16.0)


@dataclass
class Block4Tensor:
    packed: np.ndarray
    scales: np.ndarray
    shape: Tuple[int, ...]
    count: int
    block_size: int
    mapping: str

    @property
    def nbytes(self) -> int:
        return int(self.packed.nbytes + self.scales.nbytes)


@dataclass
class ComplexFirstMoment:
    phases: np.ndarray
    magnitudes: Block4Tensor
    shape: Tuple[int, ...]
    count: int

    @property
    def nbytes(self) -> int:
        return int(self.phases.nbytes + self.magnitudes.nbytes)


@dataclass
class MomentState:
    complex_parameter: bool
    first: object
    second: Block4Tensor

    @property
    def nbytes(self) -> int:
        return int(self.first.nbytes + self.second.nbytes)


def _nearest_codes(normalized: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    distances = np.abs(normalized[:, None] - codebook[None, :])
    return np.argmin(distances, axis=1).astype(np.uint8)


def quantize_block4(
    values: np.ndarray,
    block_size: int = 64,
    mapping: str = "unsigned_de",
) -> Block4Tensor:
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(-1)
    if mapping == "unsigned_de":
        codebook = DE4_UNSIGNED
    elif mapping == "signed_de":
        codebook = DE4_SIGNED
    elif mapping == "linear_no_zero":
        codebook = LINEAR4_NO_ZERO
    else:
        raise ValueError(f"unknown 4-bit mapping: {mapping}")

    codes = np.empty(flat.size, dtype=np.uint8)
    scales = np.empty((flat.size + block_size - 1) // block_size, dtype=np.float16)
    for block_index, start in enumerate(range(0, flat.size, block_size)):
        block = flat[start : start + block_size]
        scale = float(np.max(np.abs(block))) if block.size else 0.0
        scales[block_index] = np.float16(scale)
        if scale == 0.0:
            codes[start : start + block.size] = 0
            continue
        normalized = block / scale
        if mapping != "signed_de":
            normalized = np.clip(normalized, 0.0, 1.0)
        codes[start : start + block.size] = _nearest_codes(normalized, codebook)
    return Block4Tensor(
        packed=pack_nibbles(codes),
        scales=scales,
        shape=tuple(values.shape),
        count=int(flat.size),
        block_size=int(block_size),
        mapping=mapping,
    )


def dequantize_block4(state: Block4Tensor) -> np.ndarray:
    codebooks = {
        "unsigned_de": DE4_UNSIGNED,
        "signed_de": DE4_SIGNED,
        "linear_no_zero": LINEAR4_NO_ZERO,
    }
    codebook = codebooks[state.mapping]
    codes = unpack_nibbles(state.packed, state.count)
    out = np.empty(state.count, dtype=np.float32)
    for block_index, start in enumerate(range(0, state.count, state.block_size)):
        end = min(start + state.block_size, state.count)
        scale = np.float32(state.scales[block_index])
        out[start:end] = codebook[codes[start:end]] * scale
    return out.reshape(state.shape)


def quantize_complex_first(values: np.ndarray, block_size: int = 64) -> ComplexFirstMoment:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 2:
        raise ValueError("complex first moment expects a final [real, imag] axis")
    real, imag = values[..., 0], values[..., 1]
    phases = phase_codes_np(values)
    magnitudes = np.sqrt(real * real + imag * imag)
    return ComplexFirstMoment(
        phases=pack_two_bit(phases),
        magnitudes=quantize_block4(magnitudes, block_size, "unsigned_de"),
        shape=tuple(values.shape),
        count=int(phases.size),
    )


def dequantize_complex_first(state: ComplexFirstMoment) -> np.ndarray:
    phases = unpack_two_bit(state.phases, state.count).reshape(state.shape[:-1])
    magnitudes = dequantize_block4(state.magnitudes)
    real = ((phases == 0).astype(np.float32) - (phases == 2).astype(np.float32)) * magnitudes
    imag = ((phases == 1).astype(np.float32) - (phases == 3).astype(np.float32)) * magnitudes
    return np.stack([real, imag], axis=-1)


class CA4BitAdam:
    """Adam with compressed persistent moments, following the CQ-FL draft."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        block_size: int = 64,
    ) -> None:
        if tf is None:
            raise RuntimeError("TensorFlow is required for CA4BitAdam")
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.block_size = int(block_size)
        self.iterations = 0
        self._states: Dict[int, MomentState] = {}

    @staticmethod
    def _is_complex_parameter(variable) -> bool:
        shape = tuple(int(d) for d in variable.shape)
        return len(shape) >= 1 and shape[-1] == 2

    def _zero_state(self, variable) -> MomentState:
        shape = tuple(int(d) for d in variable.shape)
        complex_parameter = self._is_complex_parameter(variable)
        if complex_parameter:
            first = quantize_complex_first(np.zeros(shape, np.float32), self.block_size)
            second = quantize_block4(
                np.zeros(shape, np.float32), self.block_size, "linear_no_zero"
            )
        else:
            first = quantize_block4(np.zeros(shape, np.float32), self.block_size, "signed_de")
            second = quantize_block4(
                np.zeros(shape, np.float32), self.block_size, "linear_no_zero"
            )
        return MomentState(complex_parameter, first, second)

    def apply_gradients(self, grads_and_vars: Iterable[Tuple[object, object]]) -> None:
        pairs = [(g, v) for g, v in grads_and_vars if g is not None]
        if not pairs:
            return
        self.iterations += 1
        correction1 = 1.0 - self.beta1**self.iterations
        correction2 = 1.0 - self.beta2**self.iterations

        for gradient, variable in pairs:
            key = id(variable)
            state = self._states.setdefault(key, self._zero_state(variable))
            grad = np.asarray(gradient.numpy() if hasattr(gradient, "numpy") else gradient, np.float32)
            if state.complex_parameter:
                first = dequantize_complex_first(state.first)
                second = dequantize_block4(state.second)
                first = self.beta1 * first + (1.0 - self.beta1) * grad
                # Keep the exact Adam update: real and imaginary coordinates
                # retain independent second moments.  Only persistence changes.
                second = self.beta2 * second + (1.0 - self.beta2) * np.square(grad)
                first_hat = first / correction1
                second_hat = second / correction2
                update = first_hat / (np.sqrt(second_hat) + self.epsilon)
                variable.assign_sub(tf.convert_to_tensor(self.learning_rate * update, variable.dtype))
                state.first = quantize_complex_first(first, self.block_size)
                state.second = quantize_block4(second, self.block_size, "linear_no_zero")
            else:
                first = dequantize_block4(state.first)
                second = dequantize_block4(state.second)
                first = self.beta1 * first + (1.0 - self.beta1) * grad
                second = self.beta2 * second + (1.0 - self.beta2) * np.square(grad)
                first_hat = first / correction1
                second_hat = second / correction2
                update = first_hat / (np.sqrt(second_hat) + self.epsilon)
                variable.assign_sub(tf.convert_to_tensor(self.learning_rate * update, variable.dtype))
                state.first = quantize_block4(first, self.block_size, "signed_de")
                state.second = quantize_block4(second, self.block_size, "linear_no_zero")

    def state_bytes(self) -> int:
        return int(sum(state.nbytes for state in self._states.values()))

    def reset(self) -> None:
        self.iterations = 0
        self._states.clear()
