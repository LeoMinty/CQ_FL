"""Validate repeated experiment-one runs and draw accuracy-round curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Bit2Communication import PROTOCOL_VERSION
from cqfl.config import METHOD_NAMES


LABELS = {
    "fedavg_fp32": "FedAvg (FP32)",
    "signsgd": "SignSGD",
    "w2_fp32_adam": "2-bit W + FP32 Adam",
    "cqfl": "CQ-FL",
}

# ``method``, ``seed`` and ``output_root`` are deliberately excluded: method
# must differ between curves, seed must differ between repetitions, and the
# output location has no effect on an experiment.  Every field below must be
# identical for all curves included in one figure.
COMPARABLE_CONFIG_FIELDS = (
    "dataset",
    "data_path",
    "clients",
    "rounds",
    "local_epochs",
    "batch_size",
    "learning_rate",
    "block_size",
    "max_train_samples",
    "max_test_samples",
    "model_profile",
)


def _read_run(metrics_path: Path) -> Tuple[dict, np.ndarray]:
    config_path = metrics_path.with_name("config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json beside {metrics_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    with metrics_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {metrics_path}")
    required_columns = {"round", "test_accuracy"}
    if config.get("method") == "cqfl":
        # Full-trainable 2-bit CQ-FL changes the decoded global update, not
        # merely its communication accounting.  Reject older CQ-FL files so
        # an accuracy plot cannot mix the pre-codec algorithm with this one.
        required_columns.update(
            {
                "uplink_trainable_bytes",
                "uplink_complex_2bit_bytes",
                "uplink_real_2bit_bytes",
                "uplink_non_trainable_bytes",
                "uplink_protocol",
            }
        )
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        if config.get("method") == "cqfl" and "uplink_real_2bit_bytes" in missing_columns:
            raise ValueError(
                f"{metrics_path} is a legacy CQ-FL run from before the "
                f"full-trainable 2-bit codec; rerun CQ-FL"
            )
        raise ValueError(
            f"{metrics_path} is missing columns: {sorted(missing_columns)}"
        )
    if config.get("method") == "cqfl":
        protocols = {row["uplink_protocol"] for row in rows}
        if protocols != {PROTOCOL_VERSION}:
            raise ValueError(
                f"unsupported CQ-FL uplink protocol in {metrics_path}: "
                f"{sorted(protocols)!r}; expected {PROTOCOL_VERSION!r}"
            )

    recorded_rounds = [int(row["round"]) for row in rows]
    expected_count = int(config["rounds"])
    expected_rounds = list(range(1, expected_count + 1))
    if recorded_rounds != expected_rounds:
        raise ValueError(
            f"incomplete or non-contiguous run in {metrics_path}: "
            f"expected rounds 1..{expected_count}, got {recorded_rounds[:3]}"
            f"...{recorded_rounds[-3:]}"
        )

    curve = np.asarray(
        [float(row["test_accuracy"]) for row in rows], dtype=np.float64
    )
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"non-finite test accuracy in {metrics_path}")
    return config, curve


def _config_signature(config: dict) -> Dict[str, object]:
    return {field: config.get(field) for field in COMPARABLE_CONFIG_FIELDS}


def _describe_config_difference(reference: dict, candidate: dict) -> str:
    differences = []
    for field in COMPARABLE_CONFIG_FIELDS:
        if reference.get(field) != candidate.get(field):
            differences.append(
                f"{field}: {reference.get(field)!r} != {candidate.get(field)!r}"
            )
    return "; ".join(differences)


def collect(
    root: Path,
    dataset: str,
    method: str,
    requested_seeds: Iterable[int],
) -> Tuple[np.ndarray, List[dict], List[Path]]:
    requested_seeds = tuple(int(seed) for seed in requested_seeds)
    requested_set = set(requested_seeds)
    paths = sorted((root / dataset / method).glob("seed_*_*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no metrics found for {dataset}/{method} below {root}"
        )

    by_seed: Dict[int, Tuple[np.ndarray, dict, Path]] = {}
    extra_runs = []
    for path in paths:
        config, curve = _read_run(path)
        if config.get("dataset") != dataset or config.get("method") != method:
            raise ValueError(
                f"config identity does not match its directory: {path}"
            )
        seed = int(config["seed"])
        if seed not in requested_set:
            extra_runs.append((seed, path))
            continue
        if seed in by_seed:
            previous = by_seed[seed][2]
            raise ValueError(
                f"duplicate seed {seed} for {dataset}/{method}: "
                f"{previous} and {path}. Move the obsolete run out of {root}."
            )
        by_seed[seed] = (curve, config, path)

    if extra_runs:
        details = ", ".join(f"seed {seed}: {path}" for seed, path in extra_runs)
        raise ValueError(
            f"unexpected runs for {dataset}/{method}; requested only "
            f"{list(requested_seeds)}: {details}"
        )
    missing = [seed for seed in requested_seeds if seed not in by_seed]
    if missing:
        raise FileNotFoundError(
            f"missing seeds for {dataset}/{method}: {missing}"
        )

    curves = np.stack([by_seed[seed][0] for seed in requested_seeds])
    configs = [by_seed[seed][1] for seed in requested_seeds]
    selected_paths = [by_seed[seed][2] for seed in requested_seeds]
    return curves, configs, selected_paths


def _write_summary(
    path: Path,
    dataset: str,
    seeds: List[int],
    curves_by_method: Dict[str, np.ndarray],
) -> None:
    fields = (
        "dataset",
        "method",
        "seeds",
        "rounds",
        "final_mean",
        "final_std",
        "last10_mean",
        "last10_std",
        "mean_best_accuracy",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_NAMES:
            curves = curves_by_method[method]
            last_window = min(10, curves.shape[1])
            final_values = curves[:, -1]
            last_values = curves[:, -last_window:].mean(axis=1)
            writer.writerow(
                {
                    "dataset": dataset,
                    "method": method,
                    "seeds": ",".join(map(str, seeds)),
                    "rounds": curves.shape[1],
                    "final_mean": float(final_values.mean()),
                    "final_std": float(final_values.std()),
                    "last10_mean": float(last_values.mean()),
                    "last10_std": float(last_values.std()),
                    "mean_best_accuracy": float(curves.max(axis=1).mean()),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["ravdess", "dronerf", "mnist"]
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/experiment1")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"--seeds contains duplicates: {args.seeds}")

    output = args.output or (
        args.results_root / f"{args.dataset}_accuracy_vs_round.pdf"
    )
    summary_output = args.summary_output or (
        args.results_root / f"{args.dataset}_accuracy_summary.csv"
    )

    curves_by_method: Dict[str, np.ndarray] = {}
    reference_config = None
    reference_path = None
    for method in METHOD_NAMES:
        curves, configs, paths = collect(
            args.results_root, args.dataset, method, args.seeds
        )
        curves_by_method[method] = curves
        for config, path in zip(configs, paths):
            if reference_config is None:
                reference_config, reference_path = config, path
                continue
            if _config_signature(config) != _config_signature(reference_config):
                difference = _describe_config_difference(reference_config, config)
                raise ValueError(
                    f"incomparable configurations: {reference_path} vs {path}: "
                    f"{difference}"
                )

    plt.figure(figsize=(7.2, 4.8))
    for method in METHOD_NAMES:
        curves = curves_by_method[method]
        rounds = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        plt.plot(rounds, mean, label=LABELS[method])
        if curves.shape[0] > 1:
            plt.fill_between(rounds, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close()

    _write_summary(
        summary_output, args.dataset, list(args.seeds), curves_by_method
    )
    print(
        f"validated methods={list(METHOD_NAMES)}, seeds={args.seeds}, "
        f"rounds={next(iter(curves_by_method.values())).shape[1]}"
    )
    print(f"saved {output}")
    print(f"saved {summary_output}")


if __name__ == "__main__":
    main()
