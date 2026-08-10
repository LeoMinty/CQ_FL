"""Aggregate repeated experiment-one runs and draw accuracy-round curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cqfl.config import METHOD_NAMES


LABELS = {
    "fedavg_fp32": "FedAvg (FP32)",
    "signsgd": "SignSGD",
    "w2_fp32_adam": "2-bit W + FP32 Adam",
    "cqfl": "CQ-FL",
}


def read_curve(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([float(row["test_accuracy"]) for row in rows], dtype=np.float64)


def collect(root: Path, dataset: str, method: str):
    curves = []
    for path in sorted((root / dataset / method).glob("seed_*_*/metrics.csv")):
        curves.append(read_curve(path))
    if not curves:
        return None
    shortest = min(map(len, curves))
    return np.stack([curve[:shortest] for curve in curves])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ravdess", "dronerf", "mnist"])
    parser.add_argument("--results-root", type=Path, default=Path("results/experiment1"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.results_root / f"{args.dataset}_accuracy_vs_round.pdf"

    plt.figure(figsize=(7.2, 4.8))
    found = False
    for method in METHOD_NAMES:
        curves = collect(args.results_root, args.dataset, method)
        if curves is None:
            continue
        found = True
        rounds = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        plt.plot(rounds, mean, label=LABELS[method])
        if curves.shape[0] > 1:
            plt.fill_between(rounds, mean - std, mean + std, alpha=0.15)
    if not found:
        raise FileNotFoundError(f"no experiment-one metrics found for {args.dataset}")
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
