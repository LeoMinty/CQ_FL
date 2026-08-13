"""TensorFlow-side wiring tests for the CQ-FL federated boundary.

The module is skipped on preprocessing-only machines without TensorFlow and is
intended to run on the experiment server before a training pilot.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # Local Windows preprocessing environment may omit TF.
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed in this environment")
class FederatedWiringTests(unittest.TestCase):
    def test_upload_dispatch_and_weighted_aggregation(self):
        from Bit2Communication import (
            quantize_complex_update_np,
            quantize_real_update_np,
        )
        from cqfl.federated import _quantized_upload, _weighted_mean

        complex_delta = np.array(
            [[1.0, 0.1], [-0.1, -2.0], [0.0, 3.0]], dtype=np.float32
        )
        real_delta = np.array([1.0, -2.0, 0.0, 0.5], dtype=np.float32)

        actual_complex, complex_bytes, complex_kind = _quantized_upload(
            complex_delta, "cqfl", True
        )
        expected_complex, expected_complex_bytes = quantize_complex_update_np(
            complex_delta
        )
        np.testing.assert_array_equal(actual_complex, expected_complex)
        self.assertEqual(complex_bytes, expected_complex_bytes)
        self.assertEqual(complex_kind, "complex_2bit")

        actual_real, real_bytes, real_kind = _quantized_upload(
            real_delta, "cqfl", False
        )
        expected_real, expected_real_bytes = quantize_real_update_np(real_delta)
        np.testing.assert_array_equal(actual_real, expected_real)
        self.assertEqual(real_bytes, expected_real_bytes)
        self.assertEqual(real_kind, "real_2bit")

        second_real = np.array([-0.25, 0.5, 1.0, -2.0], dtype=np.float32)
        decoded_second, _bytes, _kind = _quantized_upload(
            second_real, "cqfl", False
        )
        aggregated = _weighted_mean(
            [[actual_real], [decoded_second]], [1, 3], tensor_index=0
        )
        np.testing.assert_array_equal(
            aggregated, actual_real * 0.25 + decoded_second * 0.75
        )

        for method in ("fedavg_fp32", "w2_fp32_adam"):
            reconstructed, payload, kind = _quantized_upload(
                real_delta, method, False
            )
            np.testing.assert_array_equal(reconstructed, real_delta)
            self.assertEqual(payload, real_delta.nbytes)
            self.assertEqual(kind, "fp32")

    def test_cqfl_uplink_error_feedback_conserves_update(self):
        from cqfl.federated import _quantized_upload

        raw = np.array([0.7, -0.2, 0.0, 1.3], dtype=np.float32)
        previous_residual = np.array([0.1, 0.05, -0.03, 0.2], dtype=np.float32)
        corrected = raw + previous_residual
        reconstructed, _payload, kind = _quantized_upload(
            corrected, "cqfl", False
        )
        next_residual = corrected - reconstructed
        np.testing.assert_allclose(
            reconstructed + next_residual, corrected, atol=1e-7, rtol=0.0
        )
        self.assertEqual(kind, "real_2bit")

    def test_binary_dense_head_is_not_misclassified_as_complex(self):
        from cqfl.config import ExperimentConfig
        from cqfl.federated import BitFLOffloadTrainer, _bitfl_variable_masks
        from cqfl.models import build_model

        model = build_model((8, 8, 1, 2), 2, "cqfl", "standard")
        _ = model(tf.zeros((1, 8, 8, 1, 2), dtype=tf.float32), training=False)
        _kernel_mask, complex_mask = _bitfl_variable_masks(model)

        logits = model.get_layer("logits")
        logits_ids = {id(logits.kernel), id(logits.bias)}
        logits_indexes = [
            index
            for index, variable in enumerate(model.trainable_variables)
            if id(variable) in logits_ids
        ]
        self.assertEqual(len(logits_indexes), 2)
        self.assertTrue(all(not complex_mask[index] for index in logits_indexes))

        config = ExperimentConfig(dataset="ravdess", method="cqfl")
        trainer = BitFLOffloadTrainer(model, "cqfl", config)
        gradients = [tf.ones_like(variable) for variable in model.trainable_variables]
        prepared = trainer._prepare_gradients(gradients)
        for index in logits_indexes:
            np.testing.assert_array_equal(
                prepared[index].numpy(), gradients[index].numpy()
            )

    def test_bitfl_uses_fp32_local_model_and_adam(self):
        from BitMyConv_noMul import ComplexConv2D
        from cqfl.config import ExperimentConfig
        from cqfl.federated import BitFLOffloadTrainer
        from cqfl.models import build_model

        model = build_model((8, 8, 1, 2), 2, "bitfl", "standard")
        _ = model(tf.zeros((1, 8, 8, 1, 2), dtype=tf.float32), training=False)
        complex_layers = [
            layer for layer in model.layers if isinstance(layer, ComplexConv2D)
        ]
        self.assertTrue(complex_layers)
        self.assertTrue(all(not layer.use_quant for layer in complex_layers))
        self.assertTrue(all(not layer.quantize_backward for layer in complex_layers))

        config = ExperimentConfig(dataset="ravdess", method="bitfl")
        trainer = BitFLOffloadTrainer(model, "bitfl", config)
        self.assertIsInstance(trainer.optimizer, tf.keras.optimizers.Adam)

    def test_dronerf_small_is_shared_and_smaller(self):
        from cqfl.models import build_model

        input_shape = (64, 32, 1, 2)
        standard = build_model(input_shape, 4, "fedavg_fp32", "standard")
        small_counts = []
        small_shapes = []
        for method in (
            "fedavg_fp32",
            "bitfl",
            "signsgd",
            "w2_fp32_adam",
            "cqfl",
        ):
            model = build_model(input_shape, 4, method, "dronerf_small")
            model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
            small_counts.append(model.count_params())
            small_shapes.append([tuple(variable.shape) for variable in model.weights])
        standard(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
        self.assertTrue(all(count == small_counts[0] for count in small_counts))
        self.assertTrue(all(shapes == small_shapes[0] for shapes in small_shapes))
        self.assertLess(small_counts[0], standard.count_params())


if __name__ == "__main__":
    unittest.main()
