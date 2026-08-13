import csv
import json
import tempfile
import unittest
from pathlib import Path

from Bit2Communication import PROTOCOL_VERSION as CQFL_PROTOCOL
from BitFLCommunication import PROTOCOL_VERSION as BITFL_PROTOCOL
from cqfl.config import METHOD_NAMES
from plot_experiment1 import _config_signature, collect
from plot_experiment4 import _read_cumulative_uplink


class FiveMethodResultTests(unittest.TestCase):
    def _write_run(self, root: Path, method: str, seed: int) -> Path:
        run = root / "ravdess" / method / f"seed_{seed}_test"
        run.mkdir(parents=True)
        config = {
            "dataset": "ravdess",
            "method": method,
            "seed": seed,
            "clients": 2,
            "rounds": 2,
            "local_epochs": 2,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "block_size": 64,
            "max_train_samples": 0,
            "max_test_samples": 0,
            "model_profile": "standard",
            "validation_ratio": 0.0,
            "bitfl_normalization_bound": 1.0,
            "bitfl_topk_fraction": 0.5,
            "bitfl_bit_flip_probability": 0.0,
            "bitfl_error_feedback": True,
            "cqfl_uplink_error_feedback": False,
            "cqfl_restore_best": False,
            "cqfl_reduce_lr_patience": 0,
            "cqfl_reduce_lr_factor": 0.5,
            "cqfl_min_learning_rate": 1e-5,
            "cqfl_early_stopping_patience": 0,
            "cqfl_early_stopping_min_delta": 0.0,
        }
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        fields = [
            "round",
            "test_accuracy",
            "uplink_protocol",
            "uplink_bytes",
            "uplink_trainable_bytes",
            "uplink_complex_2bit_bytes",
            "uplink_real_2bit_bytes",
            "uplink_bitfl_1bit_bytes",
            "uplink_non_trainable_bytes",
        ]
        with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for round_index in (1, 2):
                if method == "cqfl":
                    trainable, complex_bytes, real_bytes, bitfl_bytes = 30, 10, 20, 0
                    protocol = CQFL_PROTOCOL
                elif method == "bitfl":
                    trainable, complex_bytes, real_bytes, bitfl_bytes = 15, 0, 0, 15
                    protocol = BITFL_PROTOCOL
                else:
                    trainable, complex_bytes, real_bytes, bitfl_bytes = 40, 0, 0, 0
                    protocol = method
                writer.writerow(
                    {
                        "round": round_index,
                        "test_accuracy": 0.1 * round_index,
                        "uplink_protocol": protocol,
                        "uplink_bytes": trainable + 5,
                        "uplink_trainable_bytes": trainable,
                        "uplink_complex_2bit_bytes": complex_bytes,
                        "uplink_real_2bit_bytes": real_bytes,
                        "uplink_bitfl_1bit_bytes": bitfl_bytes,
                        "uplink_non_trainable_bytes": 5,
                    }
                )
        return run / "metrics.csv"

    def test_all_five_methods_are_collectable_and_bitfl_bytes_close(self):
        self.assertEqual(len(METHOD_NAMES), 5)
        seeds = (42, 123, 2024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for method in METHOD_NAMES:
                paths[method] = [self._write_run(root, method, seed) for seed in seeds]
                curves, configs, selected = collect(root, "ravdess", method, seeds)
                self.assertEqual(curves.shape, (3, 2))
                self.assertEqual(len(configs), 3)
                self.assertEqual(selected, paths[method])
            cumulative = _read_cumulative_uplink(paths["bitfl"][0], "bitfl")
            self.assertEqual(cumulative.tolist(), [20, 40])

    def test_bitfl_only_controls_do_not_make_other_methods_incomparable(self):
        fedavg = {
            "dataset": "ravdess",
            "rounds": 50,
            "bitfl_normalization_bound": 1.0,
            "bitfl_topk_fraction": 0.5,
        }
        bitfl = {
            "dataset": "ravdess",
            "rounds": 50,
            "bitfl_normalization_bound": 0.5,
            "bitfl_topk_fraction": 0.2,
        }
        self.assertEqual(_config_signature(fedavg), _config_signature(bitfl))

    def test_bitfl_only_controls_must_match_between_bitfl_seeds(self):
        seeds = (42, 123, 2024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [self._write_run(root, "bitfl", seed) for seed in seeds]
            config_path = paths[-1].with_name("config.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["bitfl_topk_fraction"] = 0.2
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "inconsistent bitfl-specific configurations"
            ):
                collect(root, "ravdess", "bitfl", seeds)


if __name__ == "__main__":
    unittest.main()
