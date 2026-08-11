"""Validate one completed CQ-FL run's 2-bit uplink accounting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from Bit2Communication import PROTOCOL_VERSION


EXPECTED_UPLINK = {
    ("ravdess", "standard", 2): 3_178_468,
    ("dronerf", "standard", 5): 736_565,
    ("mnist", "standard", 10): 532_990,
    ("mnist", "mnist_small", 10): 135_070,
}

SPLIT_FIELDS = (
    "uplink_trainable_bytes",
    "uplink_complex_2bit_bytes",
    "uplink_real_2bit_bytes",
    "uplink_non_trainable_bytes",
)


def validate(run_path: Path) -> None:
    run_path = run_path.resolve()
    if run_path.is_file():
        metrics_path = run_path
        run_path = run_path.parent
    else:
        metrics_path = run_path / "metrics.csv"
    config_path = run_path / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"expected config.json and metrics.csv below {run_path}"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("method") != "cqfl":
        raise ValueError(f"not a CQ-FL run: method={config.get('method')!r}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {metrics_path}")
    required = {
        "uplink_protocol",
        "uplink_bytes",
        "test_loss",
        "test_accuracy",
        *SPLIT_FIELDS,
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"legacy or incomplete CQ-FL metrics; missing {sorted(missing)}"
        )

    totals = []
    for row_number, row in enumerate(rows, start=1):
        if row["uplink_protocol"] != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol at row {row_number}: "
                f"{row['uplink_protocol']!r}; expected {PROTOCOL_VERSION!r}"
            )
        total = int(row["uplink_bytes"])
        trainable = int(row["uplink_trainable_bytes"])
        complex_bytes = int(row["uplink_complex_2bit_bytes"])
        real_bytes = int(row["uplink_real_2bit_bytes"])
        non_trainable = int(row["uplink_non_trainable_bytes"])
        if trainable != complex_bytes + real_bytes:
            raise ValueError(f"trainable split mismatch at row {row_number}")
        if total != trainable + non_trainable:
            raise ValueError(f"total split mismatch at row {row_number}")
        if complex_bytes <= 0 or real_bytes <= 0:
            raise ValueError(f"missing complex/real 2-bit payload at row {row_number}")
        values = (float(row["test_loss"]), float(row["test_accuracy"]))
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite metric at row {row_number}: {values}")
        totals.append(total)

    if len(set(totals)) != 1:
        raise ValueError(f"uplink payload changed across rounds: {sorted(set(totals))}")
    key = (
        config.get("dataset"),
        config.get("model_profile", "standard"),
        int(config.get("clients", 0)),
    )
    expected = EXPECTED_UPLINK.get(key)
    if expected is not None and totals[0] != expected:
        raise ValueError(
            f"unexpected uplink for {key}: got {totals[0]:,} B, "
            f"expected {expected:,} B from the current model layout"
        )

    first = rows[0]
    print(f"validated: {run_path}")
    print(f"rounds: {len(rows)}")
    print(f"uplink/round: {totals[0]:,} B")
    print(f"  complex 2-bit: {int(first['uplink_complex_2bit_bytes']):,} B")
    print(f"  real 2-bit: {int(first['uplink_real_2bit_bytes']):,} B")
    print(f"  FP32 non-trainable: {int(first['uplink_non_trainable_bytes']):,} B")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=Path, help="CQ-FL run directory or metrics.csv")
    args = parser.parse_args()
    validate(args.run_path)


if __name__ == "__main__":
    main()
