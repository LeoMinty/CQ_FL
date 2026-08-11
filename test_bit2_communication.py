"""Regression tests for the packed CQ-FL federated message boundary."""

from __future__ import annotations

import unittest

import numpy as np

from Bit2Communication import (
    COMPLEX_PHASE,
    REAL_AXIS,
    decode_update,
    encode_complex_update,
    encode_real_update,
    pack_two_bit,
    phase_codes_np,
    quantize_real_update_np,
    unpack_two_bit,
)
from cqfl.quantization import phase_codes_np as package_phase_codes_np


class TwoBitPackingTests(unittest.TestCase):
    def test_pack_round_trip_and_real_buffer_size(self):
        rng = np.random.default_rng(42)
        for count in (0, 1, 3, 4, 5, 257):
            codes = rng.integers(0, 4, size=count, dtype=np.uint8)
            packed = pack_two_bit(codes)
            self.assertEqual(packed.dtype, np.uint8)
            self.assertEqual(packed.nbytes, (count + 3) // 4)
            np.testing.assert_array_equal(unpack_two_bit(packed, count), codes)

    def test_pack_rejects_out_of_range_code(self):
        for codes in (
            np.array([-1, 0], dtype=np.int64),
            np.array([0, 4], dtype=np.uint8),
            np.array([0, 256], dtype=np.int64),
        ):
            with self.subTest(codes=codes):
                with self.assertRaises(ValueError):
                    pack_two_bit(codes)

    def test_pack_rejects_non_integer_codes(self):
        with self.assertRaises(TypeError):
            pack_two_bit(np.array([0.0, 1.0], dtype=np.float32))

    def test_unpack_rejects_short_buffer(self):
        with self.assertRaises(ValueError):
            unpack_two_bit(np.array([0], dtype=np.uint8), 5)


class TwoBitCodecTests(unittest.TestCase):
    def test_legacy_phase_boundaries(self):
        values = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 0.0],
                [-1.0, -1.0],
                [0.0, -1.0],
                [1.0, -1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            phase_codes_np(values),
            np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            phase_codes_np(values), package_phase_codes_np(values)
        )

    def test_complex_message_is_physically_packed(self):
        values = np.array(
            [[1.0, 0.1], [0.1, 2.0], [-3.0, 0.1], [0.1, -4.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        message = encode_complex_update(values)
        self.assertEqual(message.scheme, COMPLEX_PHASE)
        self.assertEqual(message.packed.nbytes, 2)
        self.assertEqual(message.nbytes, 6)  # 2 packed bytes + one FP32 scale.
        reconstructed = decode_update(message)
        self.assertEqual(reconstructed.shape, values.shape)
        self.assertEqual(reconstructed.dtype, np.float32)
        np.testing.assert_array_equal(
            unpack_two_bit(message.packed, message.count),
            np.array([0, 1, 2, 3, 1], dtype=np.uint8),
        )
        scale = np.float32(message.scale)
        np.testing.assert_array_equal(
            reconstructed,
            scale
            * np.array(
                [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 1]],
                dtype=np.float32,
            ),
        )

    def test_zero_complex_update_remains_zero(self):
        values = np.zeros((7, 2), dtype=np.float32)
        message = encode_complex_update(values)
        self.assertEqual(float(message.scale), 0.0)
        np.testing.assert_array_equal(decode_update(message), values)

    def test_mixed_complex_zero_follows_legacy_phase_rule(self):
        # With no fifth "zero" code, atan2(0, 0) selects +R.  A zero element
        # inside a non-zero tensor therefore reconstructs to the shared scale.
        values = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        message = encode_complex_update(values)
        reconstructed = decode_update(message)
        self.assertEqual(float(message.scale), 0.5)
        np.testing.assert_array_equal(
            reconstructed, np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
        )

    def test_real_axis_codes_preserve_sign_and_zero(self):
        values = np.array([[2.0, -1.0, 0.0, 0.25]], dtype=np.float32)
        message = encode_real_update(values)
        self.assertEqual(message.scheme, REAL_AXIS)
        self.assertEqual(message.nbytes, 5)  # 1 packed byte + one FP32 scale.
        np.testing.assert_array_equal(
            unpack_two_bit(message.packed, message.count),
            np.array([0, 2, 1, 0], dtype=np.uint8),
        )
        reconstructed = decode_update(message)
        self.assertEqual(reconstructed.shape, values.shape)
        self.assertGreater(reconstructed[0, 0], 0.0)
        self.assertLess(reconstructed[0, 1], 0.0)
        self.assertEqual(reconstructed[0, 2], 0.0)
        self.assertGreater(reconstructed[0, 3], 0.0)

    def test_dense_payload_uses_actual_two_bit_buffer(self):
        values = np.linspace(-1.0, 1.0, 1001, dtype=np.float32).reshape(7, 143)
        reconstructed, payload = quantize_real_update_np(values)
        self.assertEqual(reconstructed.shape, values.shape)
        self.assertEqual(payload, (values.size + 3) // 4 + 4)
        self.assertGreater(values.nbytes / payload, 15.0)

    def test_fp32_communication_scale_preserves_tiny_update(self):
        values = np.array([1e-9], dtype=np.float32)
        message = encode_real_update(values)
        self.assertGreater(float(message.scale), 0.0)
        np.testing.assert_array_equal(decode_update(message), values)

    def test_non_finite_and_unrepresentable_scales_are_rejected(self):
        with self.assertRaises(ValueError):
            encode_real_update(np.array([np.nan], dtype=np.float32))
        maximum = np.finfo(np.float32).max
        with self.assertRaises(OverflowError):
            encode_complex_update(
                np.array([[maximum, maximum]], dtype=np.float32)
            )

    def test_weighted_decoded_updates_differ_from_fp32_updates(self):
        base = np.array([0.25, -0.5, 1.0, 0.0], dtype=np.float32)
        first = np.array([1.0, -2.0, 0.0, 0.5], dtype=np.float32)
        second = np.array([-0.5, 1.0, 2.0, 0.0], dtype=np.float32)
        first_message = encode_real_update(first)
        second_message = encode_real_update(second)
        decoded_first = decode_update(first_message)
        decoded_second = decode_update(second_message)
        expected = base + decoded_first * 0.25 + decoded_second * 0.75
        actual = base + sum(
            update * weight
            for update, weight in ((decoded_first, 0.25), (decoded_second, 0.75))
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, base + first * 0.25 + second * 0.75))


if __name__ == "__main__":
    unittest.main()
