import unittest

from analyze_storage import (
    VariableSpec,
    block4_bytes,
    ca4_moment_bytes,
    packed_2bit_weight_bytes,
    select_experiment5_rows,
    summarize_storage,
)


class StorageFormulaTests(unittest.TestCase):
    def test_block4_includes_packed_values_and_fp32_scales(self):
        self.assertEqual(block4_bytes(64, 64), 36)
        self.assertEqual(block4_bytes(65, 64), 41)

    def test_real_ca4_has_two_four_bit_moments(self):
        spec = VariableSpec("real", (65,), 65, False)
        self.assertEqual(ca4_moment_bytes(spec, 64), (41, 41))

    def test_complex_ca4_matches_runtime_layout(self):
        spec = VariableSpec("complex", (5, 2), 10, True)
        # m: ceil(5/4) phase + ceil(5/2) magnitude + one scale.
        # v: ceil(10/2) values + one scale.
        self.assertEqual(ca4_moment_bytes(spec, 64), (2 + 3 + 4, 5 + 4))

    def test_two_bit_weights_use_logical_complex_elements(self):
        complex_spec = VariableSpec("complex", (5, 2), 10, True)
        real_spec = VariableSpec("real", (10,), 10, False)
        self.assertEqual(packed_2bit_weight_bytes(complex_spec), 2 + 4)
        self.assertEqual(packed_2bit_weight_bytes(real_spec), 3 + 4)

    def test_prototype_and_packed_target_are_kept_separate(self):
        specs = [
            VariableSpec("complex", (4, 2), 8, True),
            VariableSpec("real", (8,), 8, False),
        ]
        rows, details = summarize_storage("toy", "standard", specs, 64)
        lookup = {(row.method, row.representation): row for row in rows}
        self.assertEqual(details["trainable_scalar_parameters"], 16)
        self.assertEqual(lookup[("cqfl", "prototype")].weight_bytes, 64)
        self.assertLess(
            lookup[("cqfl", "packed_target")].weight_bytes,
            lookup[("cqfl", "prototype")].weight_bytes,
        )
        self.assertEqual(
            lookup[("cqfl", "prototype")].optimizer_bytes,
            lookup[("cqfl", "packed_target")].optimizer_bytes,
        )

    def test_experiment5_primary_cqfl_value_uses_packed_weights(self):
        specs = [
            VariableSpec("complex", (4, 2), 8, True),
            VariableSpec("real", (8,), 8, False),
        ]
        rows, _details = summarize_storage("toy", "standard", specs, 64)
        selected = select_experiment5_rows(rows)
        cqfl = next(row for row in selected if row.method == "cqfl")
        self.assertEqual(cqfl.representation, "packed_target")


if __name__ == "__main__":
    unittest.main()
