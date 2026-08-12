import unittest

import numpy as np

from BitFLCommunication import (
    decode_update,
    encode_update,
    pack_one_bit,
    topk_error_feedback,
    unpack_one_bit,
)


class BitFLPackingTests(unittest.TestCase):
    def test_pack_round_trip_and_physical_size(self):
        bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        packed = pack_one_bit(bits)
        self.assertEqual(packed.nbytes, 2)
        np.testing.assert_array_equal(unpack_one_bit(packed, bits.size), bits)

    def test_tensor_message_is_one_bit_plus_fp32_scale(self):
        values = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
        message = encode_update(values, np.random.default_rng(7), 1.0)
        self.assertEqual(message.packed.nbytes, 2)
        self.assertEqual(np.asarray(message.scale).nbytes, 4)
        self.assertEqual(message.nbytes, 6)
        self.assertEqual(decode_update(message).shape, values.shape)

    def test_zero_tensor_stays_zero(self):
        values = np.zeros((3, 4), dtype=np.float32)
        message = encode_update(values, np.random.default_rng(9), 1.0)
        np.testing.assert_array_equal(decode_update(message), values)


class BitFLAlgorithmTests(unittest.TestCase):
    def test_stochastic_codec_is_unbiased_in_expectation(self):
        values = np.array([-1.0, -0.4, 0.0, 0.25, 0.8], dtype=np.float32)
        rng = np.random.default_rng(1234)
        draws = np.stack(
            [decode_update(encode_update(values, rng, 1.0)) for _ in range(20_000)]
        )
        np.testing.assert_allclose(draws.mean(axis=0), values, atol=0.025, rtol=0.0)

    def test_topk_residual_conserves_the_full_update(self):
        update = [
            np.array([1.0, -4.0, 2.0], dtype=np.float32),
            np.array([3.0, -0.5, 5.0], dtype=np.float32),
        ]
        residual = [np.zeros_like(value) for value in update]
        applied, next_residual = topk_error_feedback(update, residual, 0.5)
        flat_applied = np.concatenate([value.reshape(-1) for value in applied])
        self.assertEqual(np.count_nonzero(flat_applied), 3)
        for source, kept, error in zip(update, applied, next_residual):
            np.testing.assert_array_equal(kept + error, source)

        zeros = [np.zeros_like(value) for value in update]
        applied_next, residual_next = topk_error_feedback(
            zeros, next_residual, 1.0
        )
        for previous_error, kept, error in zip(
            next_residual, applied_next, residual_next
        ):
            np.testing.assert_array_equal(kept, previous_error)
            np.testing.assert_array_equal(error, np.zeros_like(error))


if __name__ == "__main__":
    unittest.main()
