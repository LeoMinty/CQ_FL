import unittest

import numpy as np

from cqfl.config import ExperimentConfig
from cqfl.data import _group_train_validation_test_indices


class ValidationSplitTests(unittest.TestCase):
    def test_dronerf_split_keeps_physical_groups_disjoint_and_stratified(self):
        groups = np.repeat(np.arange(80), 3)
        group_labels = np.repeat(np.arange(4), 20)
        labels = np.repeat(group_labels, 3)

        train, validation, test = _group_train_validation_test_indices(
            groups, labels, seed=42, validation_ratio=0.1
        )

        train_groups = set(groups[train])
        validation_groups = set(groups[validation])
        test_groups = set(groups[test])
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)
        self.assertEqual(train_groups | validation_groups | test_groups, set(groups))
        self.assertEqual(set(labels[validation]), {0, 1, 2, 3})
        self.assertEqual(set(labels[test]), {0, 1, 2, 3})

    def test_split_is_deterministic(self):
        groups = np.repeat(np.arange(40), 2)
        labels = np.repeat(np.repeat(np.arange(4), 10), 2)
        first = _group_train_validation_test_indices(groups, labels, 123, 0.1)
        second = _group_train_validation_test_indices(groups, labels, 123, 0.1)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_cqfl_validation_requires_restore_best(self):
        with self.assertRaisesRegex(ValueError, "requires cqfl_restore_best"):
            ExperimentConfig(
                dataset="dronerf",
                method="cqfl",
                validation_ratio=0.1,
            ).resolved()

    def test_validation_configuration_resolves(self):
        config = ExperimentConfig(
            dataset="dronerf",
            method="cqfl",
            validation_ratio=0.1,
            cqfl_restore_best=True,
        ).resolved()
        self.assertEqual(config.validation_ratio, 0.1)


if __name__ == "__main__":
    unittest.main()
