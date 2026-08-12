import csv
import json
import tempfile
import unittest
from pathlib import Path

from Bit2Communication import PROTOCOL_VERSION as CQFL_PROTOCOL
from BitFLCommunication import PROTOCOL_VERSION as BITFL_PROTOCOL
from cqfl.config import METHOD_NAMES
from plot_experiment1 import collect
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
            "bitfl_normalization_bound": 1.0,
            "bitfl_topk_fraction": 0.5,
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


if __name__ == "__main__":
    unittest.main()
