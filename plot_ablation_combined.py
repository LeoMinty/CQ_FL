"""Plot experiments 2 and 3 as one shared-baseline ablation study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_ablation_experiments import _collect


VARIANTS = (
    ("full_cqfl", 2, "cpmq", "Full CQ-FL", "#d62728"),
    (
        "without_cpmq",
        2,
        "independent",
        "w/o CPMQ (independent Re/Im)",
        "#1f77b4",
    ),
    (
        "without_phase_ste",
        3,
        "standard_ste",
        "w/o phase-domain STE",
        "#2ca02c",
    ),
)

_COMBINED_CONTROL_FIELDS = {
    "output_root",
    "seed",
    "cqfl_complex_first_moment",
    "cqfl_phase_ste",
}


def _combined_signature(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in _COMBINED_CONTROL_FIELDS
    }


def collect_combined(
    results_root: Path, dataset: str, seeds: List[int]
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    collected = {}
    reference_shape = None
    reference_signature = None
    for key, experiment, source_variant, _label, _color in VARIANTS:
        result = _collect(
            results_root,
            experiment,
            dataset,
            source_variant,
            seeds,
        )
        accuracy, _uplink, _optimizer, config = result
        if reference_shape is not None and accuracy.shape != reference_shape:
            raise ValueError(
                "combined ablation variants have different seed/round counts"
            )
        signature = _combined_signature(config)
        if reference_signature is not None and signature != reference_signature:
            differing = sorted(
                key
                for key in set(reference_signature).union(signature)
                if reference_signature.get(key) != signature.get(key)
            )
            raise ValueError(
                "combined ablation variants differ outside the intended controls: "
                f"{differing}"
            )
        reference_shape = accuracy.shape
        reference_signature = signature
        collected[key] = result
    return collected


def _write_summary(
    path: Path,
    dataset: str,
    seeds: List[int],
    collected: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
) -> None:
    baseline_accuracy = collected["full_cqfl"][0]
    baseline_final = float(baseline_accuracy[:, -1].mean())
    baseline_last = min(10, baseline_accuracy.shape[1])
    baseline_last10 = float(
        baseline_accuracy[:, -baseline_last:].mean(axis=1).mean()
    )
    fields = [
        "dataset",
        "variant",
        "label",
        "source_experiment",
        "source_variant",
        "seeds",
        "rounds",
        "final_accuracy_mean",
        "final_accuracy_std",
        "final_delta_vs_full_pp",
        "last10_accuracy_mean",
        "last10_accuracy_std",
        "last10_delta_vs_full_pp",
        "mean_best_accuracy",
        "cumulative_uplink_mib_mean",
        "final_optimizer_state_mib_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, experiment, source_variant, label, _color in VARIANTS:
            accuracy, uplink, optimizer, _config = collected[key]
            last = min(10, accuracy.shape[1])
            final_mean = float(accuracy[:, -1].mean())
            last10_per_seed = accuracy[:, -last:].mean(axis=1)
            last10_mean = float(last10_per_seed.mean())
            writer.writerow(
                {
                    "dataset": dataset,
                    "variant": key,
                    "label": label,
                    "source_experiment": experiment,
                    "source_variant": source_variant,
                    "seeds": ",".join(map(str, seeds)),
                    "rounds": accuracy.shape[1],
                    "final_accuracy_mean": final_mean,
                    "final_accuracy_std": float(accuracy[:, -1].std()),
                    "final_delta_vs_full_pp": 100.0
                    * (final_mean - baseline_final),
                    "last10_accuracy_mean": last10_mean,
                    "last10_accuracy_std": float(last10_per_seed.std()),
                    "last10_delta_vs_full_pp": 100.0
                    * (last10_mean - baseline_last10),
                    "mean_best_accuracy": float(
                        accuracy.max(axis=1).mean()
                    ),
                    "cumulative_uplink_mib_mean": float(
                        uplink[:, -1].mean() / 2**20
                    ),
                    "final_optimizer_state_mib_mean": float(
                        optimizer[:, -1].mean() / 2**20
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined CPMQ and phase-domain STE ablation plot"
    )
    parser.add_argument(
        "--dataset",
        default="ravdess",
        choices=["ravdess", "dronerf", "mnist"],
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 2024]
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"duplicate seeds: {args.seeds}")

    collected = collect_combined(
        args.results_root, args.dataset, list(args.seeds)
    )
    output = args.output or (
        args.results_root
        / "ablation_combined"
        / f"{args.dataset}_ablation_accuracy.pdf"
    )
    summary_output = args.summary_output or (
        args.results_root
        / "ablation_combined"
        / f"{args.dataset}_ablation_summary.csv"
    )

    figure, (curve_axis, bar_axis) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.3),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    bar_means = []
    bar_stds = []
    bar_labels = []
    bar_colors = []
    for key, _experiment, _source_variant, label, color in VARIANTS:
        accuracy = collected[key][0]
        rounds = np.arange(1, accuracy.shape[1] + 1)
        mean = accuracy.mean(axis=0)
        std = accuracy.std(axis=0)
        curve_axis.plot(rounds, mean, label=label, color=color, linewidth=2.0)
        if accuracy.shape[0] > 1:
            curve_axis.fill_between(
                rounds,
                mean - std,
                mean + std,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        bar_means.append(float(accuracy[:, -1].mean()))
        bar_stds.append(float(accuracy[:, -1].std()))
        bar_labels.append(
            {
                "full_cqfl": "Full\nCQ-FL",
                "without_cpmq": "w/o\nCPMQ",
                "without_phase_ste": "w/o phase\nSTE",
            }[key]
        )
        bar_colors.append(color)

    curve_axis.set_xlabel("Communication round")
    curve_axis.set_ylabel("Test accuracy")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(fontsize=8.5)
    curve_axis.set_title("(a) Convergence")

    positions = np.arange(len(bar_means))
    bar_axis.bar(
        positions,
        bar_means,
        yerr=bar_stds,
        capsize=4,
        color=bar_colors,
        alpha=0.88,
        width=0.68,
    )
    bar_axis.set_xticks(positions, bar_labels)
    bar_axis.set_ylabel("Final test accuracy")
    bar_axis.grid(axis="y", alpha=0.25)
    bar_axis.set_title("(b) Final accuracy")
    upper = min(1.0, max(bar_means) + max(bar_stds + [0.0]) + 0.03)
    if upper > 0.0:
        bar_axis.set_ylim(0.0, upper)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    _write_summary(
        summary_output,
        args.dataset,
        list(args.seeds),
        collected,
    )
    print(
        "validated shared baseline=experiment2/cpmq, "
        "ablations=[experiment2/independent, experiment3/standard_ste]"
    )
    print(f"saved {output}")
    print(f"saved {summary_output}")


if __name__ == "__main__":
    main()
