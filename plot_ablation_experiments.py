"""Validate repeated ablation runs and plot mean accuracy with seed variation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Bit2Communication import PROTOCOL_VERSION
from cqfl.config import ABLATION_VARIANTS


LABELS = {
    2: {"cpmq": "CPMQ", "independent": "Independent Re/Im 4-bit"},
    3: {"phase_ste": "Phase-domain STE", "standard_ste": "Standard STE"},
}
COLORS = {"cpmq": "#d62728", "independent": "#1f77b4", "phase_ste": "#d62728", "standard_ste": "#2ca02c"}
CONTROL_FIELDS = {
    2: ("cqfl_complex_first_moment",),
    3: ("cqfl_phase_ste",),
}
IGNORED_FIELDS = {"output_root", "seed"}


def _read_run(path: Path) -> Tuple[dict, List[dict]]:
    with path.with_name("config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rounds = list(range(1, int(config["rounds"]) + 1))
    recorded_rounds = [int(row["round"]) for row in rows]
    if recorded_rounds != expected_rounds:
        raise ValueError(f"incomplete run: {path}")
    protocols = {row["uplink_protocol"] for row in rows}
    if protocols != {PROTOCOL_VERSION}:
        raise ValueError(f"unsupported CQ-FL protocol in {path}: {protocols}")
    accuracy = np.asarray([float(row["test_accuracy"]) for row in rows])
    if not np.all(np.isfinite(accuracy)):
        raise ValueError(f"non-finite accuracy in {path}")
    return config, rows


def _base_signature(config: dict, experiment: int) -> dict:
    excluded = IGNORED_FIELDS.union(CONTROL_FIELDS[experiment])
    return {key: value for key, value in config.items() if key not in excluded}


def _collect(
    root: Path, experiment: int, dataset: str, variant: str, seeds: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    paths = sorted((root / f"experiment{experiment}" / variant / dataset / "cqfl").glob("seed_*_*/metrics.csv"))
    by_seed = {}
    for path in paths:
        config, rows = _read_run(path)
        seed = int(config["seed"])
        if seed not in seeds:
            raise ValueError(f"unexpected seed {seed}: {path}")
        if seed in by_seed:
            raise ValueError(f"duplicate seed {seed}: {by_seed[seed][3]} and {path}")
        expected = ABLATION_VARIANTS[experiment][variant]
        for field, value in expected.items():
            if config.get(field) != value:
                raise ValueError(f"{variant} has {field}={config.get(field)!r}, expected {value!r}: {path}")
        by_seed[seed] = (
            np.asarray([float(row["test_accuracy"]) for row in rows]),
            np.asarray([float(row["uplink_bytes"]) for row in rows]).cumsum(),
            np.asarray([float(row["optimizer_state_bytes"]) for row in rows]),
            path,
            config,
        )
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise FileNotFoundError(f"missing {variant} seeds: {missing}")
    reference = by_seed[seeds[0]][4]
    for seed in seeds[1:]:
        if _base_signature(by_seed[seed][4], experiment) != _base_signature(reference, experiment):
            raise ValueError(f"inconsistent repeated-run config for {variant}, seed {seed}")
    stacked = tuple(
        np.stack([by_seed[seed][index] for seed in seeds]) for index in range(3)
    )
    return (*stacked, reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True, choices=[2, 3])
    parser.add_argument("--dataset", default="ravdess", choices=["ravdess", "dronerf", "mnist"])
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"duplicate seeds: {args.seeds}")

    collected: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = {}
    reference_shape = None
    reference_config_signature = None
    for variant in ABLATION_VARIANTS[args.experiment]:
        collected[variant] = _collect(
            args.results_root, args.experiment, args.dataset, variant, args.seeds
        )
        # The changed control field is the sole allowed config difference.
        shape = collected[variant][0].shape
        if reference_shape is not None and shape != reference_shape:
            raise ValueError("ablation variants have different seed/round counts")
        reference_shape = shape
        config_signature = _base_signature(collected[variant][3], args.experiment)
        if (
            reference_config_signature is not None
            and config_signature != reference_config_signature
        ):
            raise ValueError(
                "ablation variants differ in fields other than the intended control"
            )
        reference_config_signature = config_signature

    output = args.output or args.results_root / f"experiment{args.experiment}" / f"{args.dataset}_accuracy_vs_round.pdf"
    summary_output = args.summary_output or args.results_root / f"experiment{args.experiment}" / f"{args.dataset}_summary.csv"
    plt.figure(figsize=(7.2, 4.8))
    for variant, (accuracy, _, _, _) in collected.items():
        rounds = np.arange(1, accuracy.shape[1] + 1)
        mean, std = accuracy.mean(axis=0), accuracy.std(axis=0)
        plt.plot(rounds, mean, label=LABELS[args.experiment][variant], color=COLORS[variant])
        if accuracy.shape[0] > 1:
            plt.fill_between(rounds, mean - std, mean + std, color=COLORS[variant], alpha=0.15)
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close()

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "dataset", "variant", "seeds", "rounds", "final_accuracy_mean", "final_accuracy_std", "last10_accuracy_mean", "mean_best_accuracy", "cumulative_uplink_mib_mean", "final_optimizer_state_mib_mean"])
        writer.writeheader()
        for variant, (accuracy, uplink, optimizer, _) in collected.items():
            last = min(10, accuracy.shape[1])
            writer.writerow({
                "experiment": args.experiment,
                "dataset": args.dataset,
                "variant": variant,
                "seeds": ",".join(map(str, args.seeds)),
                "rounds": accuracy.shape[1],
                "final_accuracy_mean": float(accuracy[:, -1].mean()),
                "final_accuracy_std": float(accuracy[:, -1].std()),
                "last10_accuracy_mean": float(accuracy[:, -last:].mean(axis=1).mean()),
                "mean_best_accuracy": float(accuracy.max(axis=1).mean()),
                "cumulative_uplink_mib_mean": float(uplink[:, -1].mean() / 2**20),
                "final_optimizer_state_mib_mean": float(optimizer[:, -1].mean() / 2**20),
            })
    print(f"saved {output}")
    print(f"saved {summary_output}")


if __name__ == "__main__":
    main()
