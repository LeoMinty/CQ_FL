"""Validate repeated experiment-one runs and draw accuracy-round curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np

from Bit2Communication import PROTOCOL_VERSION
from BitFLCommunication import (
    LEGACY_PROTOCOL_VERSION as BITFL_LEGACY_PROTOCOL_VERSION,
    PROTOCOL_VERSION as BITFL_PROTOCOL_VERSION,
)
from cqfl.config import METHOD_NAMES


LABELS = {
    "fedavg_fp32": "FedAvg (FP32)",
    "bitfl": "BitFL (1-bit)",
    "signsgd": "SignSGD",
    "w2_fp32_adam": "2-bit W + FP32 Adam",
    "cqfl": "CQ-FL (ours)",
}

COLORS = {
    "fedavg_fp32": "#1f77b4",
    "bitfl": "#9467bd",
    "signsgd": "#ff7f0e",
    "w2_fp32_adam": "#2ca02c",
    "cqfl": "#d62728",
}

DATASETS = ("ravdess", "dronerf", "mnist")
DATASET_TITLES = {
    "ravdess": "RAVDESS",
    "dronerf": "DroneRF",
    "mnist": "MNIST",
}
DEFAULT_TRIPTYCH_ROOTS = {
    "ravdess": Path("results/experiment1_five_v1_final"),
    "dronerf": Path("results/experiment1_five_cqfl_optimized_50round_final"),
    "mnist": Path("results/experiment1_five_mnist_tuned_final"),
}
STYLES = {
    "fedavg_fp32": {"linestyle": "-", "marker": "o"},
    "bitfl": {"linestyle": ":", "marker": "s"},
    "signsgd": {"linestyle": "--", "marker": "^"},
    "w2_fp32_adam": {"linestyle": "-.", "marker": "D"},
    "cqfl": {"linestyle": "-", "marker": "*"},
}


def configure_paper_plot_style() -> None:
    """Use fonts that remain at least 9 pt in a full-width paper figure."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Times",
                "cmr10",
            ],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.formatter.use_mathtext": True,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    selected_font = font_manager.findfont(
        font_manager.FontProperties(family="serif")
    )
    print(f"paper plot font: {selected_font}")

# ``method``, ``seed`` and ``output_root`` are deliberately excluded: method
# must differ between curves, seed must differ between repetitions, and the
# output location has no effect on an experiment.  Every field below must be
# identical for all curves included in one figure. BitFL-only controls are
# checked separately within the three BitFL seeds and are intentionally absent
# here because they do not affect the other four methods.
COMPARABLE_CONFIG_FIELDS = (
    "dataset",
    "data_path",
    "clients",
    "rounds",
    "local_epochs",
    #"batch_size",
    "block_size",
    "max_train_samples",
    "max_test_samples",
    "model_profile",
)

METHOD_SPECIFIC_CONFIG_FIELDS = {
    "fedavg_fp32": ("learning_rate",),
    "bitfl": (
        "learning_rate",
        "bitfl_normalization_bound",
        "bitfl_topk_fraction",
        "bitfl_bit_flip_probability",
        "bitfl_error_feedback",
    ),
    "signsgd": ("learning_rate",),
    "w2_fp32_adam": ("learning_rate",),
    "cqfl": (
        "learning_rate",
        "cqfl_uplink_error_feedback",
        "cqfl_restore_best",
        "cqfl_reduce_lr_patience",
        "cqfl_reduce_lr_factor",
        "cqfl_min_learning_rate",
        "cqfl_early_stopping_patience",
        "cqfl_early_stopping_min_delta",
    ),
}

COMPARABLE_CONFIG_DEFAULTS = {
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
    if config.get("method") == "bitfl":
        required_columns.update(
            {
                "uplink_trainable_bytes",
                "uplink_bitfl_1bit_bytes",
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
    if config.get("method") == "bitfl":
        protocols = {row["uplink_protocol"] for row in rows}
        allowed_protocols = {BITFL_PROTOCOL_VERSION}
        if (
            "bitfl_bit_flip_probability" not in config
            and "bitfl_error_feedback" not in config
        ):
            allowed_protocols.add(BITFL_LEGACY_PROTOCOL_VERSION)
        if len(protocols) != 1 or not protocols.issubset(allowed_protocols):
            raise ValueError(
                f"unsupported BitFL uplink protocol in {metrics_path}: "
                f"{sorted(protocols)!r}; expected one of {sorted(allowed_protocols)!r}"
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


def _config_signature(config: dict, max_rounds: int = 0) -> Dict[str, object]:
    signature = {
        field: config.get(field, COMPARABLE_CONFIG_DEFAULTS.get(field))
        for field in COMPARABLE_CONFIG_FIELDS
    }
    if max_rounds:
        signature["rounds"] = max_rounds
    return signature


def _describe_config_difference(
    reference: dict, candidate: dict, max_rounds: int = 0
) -> str:
    differences = []
    for field in COMPARABLE_CONFIG_FIELDS:
        reference_value = reference.get(field, COMPARABLE_CONFIG_DEFAULTS.get(field))
        candidate_value = candidate.get(field, COMPARABLE_CONFIG_DEFAULTS.get(field))
        if field == "rounds" and max_rounds:
            reference_value = candidate_value = max_rounds
        if reference_value != candidate_value:
            differences.append(
                f"{field}: {reference_value!r} != {candidate_value!r}"
            )
    return "; ".join(differences)


def _method_config_signature(config: dict) -> Dict[str, object]:
    fields = METHOD_SPECIFIC_CONFIG_FIELDS.get(config.get("method"), ())
    return {
        field: config.get(field, COMPARABLE_CONFIG_DEFAULTS.get(field))
        for field in fields
    }


def _describe_method_config_difference(reference: dict, candidate: dict) -> str:
    fields = METHOD_SPECIFIC_CONFIG_FIELDS.get(reference.get("method"), ())
    differences = []
    for field in fields:
        reference_value = reference.get(field, COMPARABLE_CONFIG_DEFAULTS.get(field))
        candidate_value = candidate.get(field, COMPARABLE_CONFIG_DEFAULTS.get(field))
        if reference_value != candidate_value:
            differences.append(
                f"{field}: {reference_value!r} != {candidate_value!r}"
            )
    return "; ".join(differences)


def collect(
    root: Path,
    dataset: str,
    method: str,
    requested_seeds: Iterable[int],
    max_rounds: int = 0,
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
        if max_rounds:
            if len(curve) < max_rounds:
                raise ValueError(
                    f"run has only {len(curve)} rounds but --max-rounds "
                    f"requires {max_rounds}: {path}"
                )
            curve = curve[:max_rounds]
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

    method_reference = by_seed[requested_seeds[0]][1]
    method_reference_path = by_seed[requested_seeds[0]][2]
    for seed in requested_seeds[1:]:
        candidate = by_seed[seed][1]
        if _method_config_signature(candidate) != _method_config_signature(
            method_reference
        ):
            difference = _describe_method_config_difference(
                method_reference, candidate
            )
            raise ValueError(
                f"inconsistent {method}-specific configurations: "
                f"{method_reference_path} vs {by_seed[seed][2]}: {difference}"
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
        "--dataset", required=True, choices=[*DATASETS, "all"]
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/experiment1")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument(
        "--ravdess-results-root",
        type=Path,
        default=DEFAULT_TRIPTYCH_ROOTS["ravdess"],
    )
    parser.add_argument(
        "--dronerf-results-root",
        type=Path,
        default=DEFAULT_TRIPTYCH_ROOTS["dronerf"],
    )
    parser.add_argument(
        "--mnist-results-root",
        type=Path,
        default=DEFAULT_TRIPTYCH_ROOTS["mnist"],
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="plot only rounds 1..N and allow source runs configured for more rounds",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"--seeds contains duplicates: {args.seeds}")
    if args.max_rounds < 0:
        raise ValueError("--max-rounds must be non-negative")

    roots = {
        "ravdess": args.ravdess_results_root,
        "dronerf": args.dronerf_results_root,
        "mnist": args.mnist_results_root,
    }
    if args.dataset == "all":
        missing_roots = [name for name, root in roots.items() if root is None]
        if missing_roots:
            raise ValueError(
                "--dataset all requires separate result roots for all datasets; "
                f"missing {missing_roots}"
            )
        if args.summary_output is not None:
            raise ValueError("--summary-output is only supported for one dataset")
        datasets = list(DATASETS)
        output = args.output or Path("experiment1_accuracy_triptych.pdf")
    else:
        datasets = [args.dataset]
        roots[args.dataset] = roots[args.dataset] or args.results_root
        output = args.output or (
            roots[args.dataset] / f"{args.dataset}_accuracy_vs_round.pdf"
        )

    curves_by_dataset: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset in datasets:
        curves_by_method: Dict[str, np.ndarray] = {}
        reference_config = None
        reference_path = None
        for method in METHOD_NAMES:
            curves, configs, paths = collect(
                roots[dataset], dataset, method, args.seeds, args.max_rounds
            )
            curves_by_method[method] = curves
            for config, path in zip(configs, paths):
                if reference_config is None:
                    reference_config, reference_path = config, path
                    continue
                if _config_signature(config, args.max_rounds) != _config_signature(
                    reference_config, args.max_rounds
                ):
                    difference = _describe_config_difference(
                        reference_config, config, args.max_rounds
                    )
                    raise ValueError(
                        f"incomparable configurations: {reference_path} vs {path}: "
                        f"{difference}"
                    )
        curves_by_dataset[dataset] = curves_by_method
        summary_output = (
            args.summary_output
            if len(datasets) == 1 and args.summary_output is not None
            else roots[dataset] / f"{dataset}_accuracy_summary.csv"
        )
        _write_summary(summary_output, dataset, list(args.seeds), curves_by_method)

    configure_paper_plot_style()
    if len(datasets) == 3:
        fig, grid_axes = plt.subplots(2, 2, figsize=(3.39, 2.40))
        axes = [grid_axes[0, 0], grid_axes[0, 1], grid_axes[1, 0]]
        legend_axis = grid_axes[1, 1]
        legend_axis.axis("off")
    else:
        fig, axis = plt.subplots(figsize=(3.39, 2.55))
        axes = [axis]
        legend_axis = None

    legend_handles = None
    legend_labels = None
    for panel_index, (axis, dataset) in enumerate(zip(axes, datasets)):
        for method in METHOD_NAMES:
            curves = curves_by_dataset[dataset][method]
            rounds = np.arange(1, curves.shape[1] + 1)
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            marker_step = max(1, len(rounds) // 8)
            axis.plot(
                rounds,
                mean,
                label=LABELS[method],
                color=COLORS[method],
                linewidth=1.25,
                markersize=3.0,
                markevery=marker_step,
                **STYLES[method],
            )
            if curves.shape[0] > 1:
                axis.fill_between(
                    rounds,
                    np.clip(mean - std, 0.0, 1.0),
                    np.clip(mean + std, 0.0, 1.0),
                    color=COLORS[method],
                    alpha=0.12,
                )
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(axis="both", pad=1.5)
        if len(datasets) == 3:
            axis.text(
                0.97,
                0.06,
                f"({chr(ord('a') + panel_index)}) {DATASET_TITLES[dataset]}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.82,
                    "edgecolor": "none",
                    "pad": 0.25,
                },
            )
        else:
            axis.set_title(DATASET_TITLES[dataset])
            axis.set_ylabel("Test accuracy")
        legend_handles, legend_labels = axis.get_legend_handles_labels()

    if len(datasets) == 3:
        fig.text(0.52, 0.01, "Communication round", ha="center", va="bottom")
        fig.text(0.006, 0.54, "Test accuracy", ha="left", va="center", rotation=90)
        legend_axis.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=1,
            handlelength=1.4,
            handletextpad=0.35,
            borderpad=0.30,
            labelspacing=0.25,
        )
        fig.subplots_adjust(
            left=0.155,
            right=0.995,
            top=0.995,
            bottom=0.13,
            wspace=0.30,
            hspace=0.18,
        )
    else:
        axes[0].set_xlabel("Communication round")
        axes[0].legend()
        fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(
        f"validated datasets={datasets}, methods={list(METHOD_NAMES)}, "
        f"seeds={args.seeds}"
    )
    print(f"saved {output}")
    print(f"saved {summary_output}")


if __name__ == "__main__":
    main()
