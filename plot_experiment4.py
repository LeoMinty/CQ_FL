"""Draw experiment 4: test accuracy versus cumulative uplink payload.

This script reuses the strict experiment-one run selection so communication
figures cannot silently mix duplicate seeds, incomplete runs, or different
training configurations.  ``uplink_bytes`` is the simulator's packed logical
payload summed over all participating clients; it is not measured network
traffic or wall-clock communication latency.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from BitFLCommunication import PROTOCOL_VERSION as BITFL_PROTOCOL_VERSION
from cqfl.config import METHOD_NAMES
from plot_experiment1 import (
    LABELS,
    _config_signature,
    _describe_config_difference,
    collect,
)


UNIT_DIVISORS = {
    "bytes": (1.0, "bytes"),
    "kib": (1024.0, "KiB"),
    "mib": (1024.0**2, "MiB"),
    "gib": (1024.0**3, "GiB"),
}

STYLES = {
    "fedavg_fp32": {"color": "#1f77b4", "linestyle": "-"},
    "bitfl": {"color": "#9467bd", "linestyle": ":"},
    "signsgd": {"color": "#ff7f0e", "linestyle": "--"},
    "w2_fp32_adam": {"color": "#2ca02c", "linestyle": "-."},
    "cqfl": {"color": "#d62728", "linestyle": "-"},
}


def _read_cumulative_uplink(metrics_path: Path, method: str) -> np.ndarray:
    with metrics_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {metrics_path}")
    if "uplink_bytes" not in rows[0]:
        raise ValueError(
            f"{metrics_path} has no uplink_bytes column; this run cannot be "
            "used for experiment 4"
        )

    # CQ-FL results produced before the full-trainable 2-bit codec only
    # compressed tensors whose final dimension happened to be two.  Reject
    # those legacy files instead of silently drawing an invalid near-FP32
    # communication curve.  The split fields also make the byte accounting
    # auditable: packed complex + packed real + FP32 non-trainable state must
    # equal the reported total in every round.
    if method == "cqfl":
        split_fields = {
            "uplink_trainable_bytes",
            "uplink_complex_2bit_bytes",
            "uplink_real_2bit_bytes",
            "uplink_non_trainable_bytes",
        }
        missing = split_fields.difference(rows[0])
        if missing:
            raise ValueError(
                f"{metrics_path} is a legacy CQ-FL run without full-trainable "
                f"2-bit accounting (missing {sorted(missing)}); rerun CQ-FL"
            )
        for row_index, row in enumerate(rows, start=1):
            try:
                total = int(row["uplink_bytes"])
                trainable = int(row["uplink_trainable_bytes"])
                complex_bytes = int(row["uplink_complex_2bit_bytes"])
                real_bytes = int(row["uplink_real_2bit_bytes"])
                non_trainable = int(row["uplink_non_trainable_bytes"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid CQ-FL uplink split in {metrics_path}, row {row_index}"
                ) from error
            if trainable != complex_bytes + real_bytes:
                raise ValueError(
                    f"CQ-FL trainable byte split does not add up in "
                    f"{metrics_path}, row {row_index}"
                )
            if total != trainable + non_trainable:
                raise ValueError(
                    f"CQ-FL total uplink byte split does not add up in "
                    f"{metrics_path}, row {row_index}"
                )
            if complex_bytes <= 0 or real_bytes <= 0:
                raise ValueError(
                    f"CQ-FL did not encode both complex and real trainable "
                    f"updates in {metrics_path}, row {row_index}"
                )

    if method == "bitfl":
        required = {
            "uplink_trainable_bytes",
            "uplink_bitfl_1bit_bytes",
            "uplink_non_trainable_bytes",
            "uplink_protocol",
        }
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(
                f"{metrics_path} is missing BitFL accounting fields: {sorted(missing)}"
            )
        for row_index, row in enumerate(rows, start=1):
            try:
                total = int(row["uplink_bytes"])
                trainable = int(row["uplink_trainable_bytes"])
                one_bit = int(row["uplink_bitfl_1bit_bytes"])
                non_trainable = int(row["uplink_non_trainable_bytes"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid BitFL byte split in {metrics_path}, row {row_index}"
                ) from error
            if row["uplink_protocol"] != BITFL_PROTOCOL_VERSION:
                raise ValueError(
                    f"unsupported BitFL protocol in {metrics_path}, row {row_index}"
                )
            if trainable != one_bit or total != one_bit + non_trainable:
                raise ValueError(
                    f"BitFL byte split does not add up in {metrics_path}, row {row_index}"
                )
            if one_bit <= 0:
                raise ValueError(
                    f"BitFL has no packed 1-bit payload in {metrics_path}, row {row_index}"
                )

    try:
        per_round = np.asarray(
            [int(row["uplink_bytes"]) for row in rows], dtype=np.int64
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid uplink_bytes in {metrics_path}") from error
    if np.any(per_round < 0):
        raise ValueError(f"negative uplink_bytes in {metrics_path}")
    return np.cumsum(per_round, dtype=np.int64)


def _collect_communication(
    root: Path,
    dataset: str,
    method: str,
    seeds: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[Path]]:
    accuracy, configs, paths = collect(root, dataset, method, seeds)
    cumulative = np.stack(
        [_read_cumulative_uplink(path, method) for path in paths]
    )
    if cumulative.shape != accuracy.shape:
        raise ValueError(
            f"accuracy/uplink shape mismatch for {dataset}/{method}: "
            f"{accuracy.shape} vs {cumulative.shape}"
        )

    # With one fixed model and method, packed payload size is determined by
    # tensor shapes and must not depend on the random seed.  Rejecting a
    # mismatch avoids averaging accuracies at different communication budgets.
    for seed_index in range(1, cumulative.shape[0]):
        if not np.array_equal(cumulative[0], cumulative[seed_index]):
            raise ValueError(
                f"cumulative uplink differs between seeds for "
                f"{dataset}/{method}; inspect the selected runs"
            )
    return accuracy, cumulative, configs, paths


def _bytes_to_target(
    accuracy: np.ndarray,
    cumulative: np.ndarray,
    target: float,
) -> np.ndarray:
    reached = np.full(accuracy.shape[0], np.nan, dtype=np.float64)
    for seed_index, curve in enumerate(accuracy):
        positions = np.flatnonzero(curve >= target)
        if positions.size:
            reached[seed_index] = float(cumulative[seed_index, positions[0]])
    return reached


def _write_summary(
    path: Path,
    dataset: str,
    seeds: List[int],
    accuracy_by_method: Dict[str, np.ndarray],
    cumulative_by_method: Dict[str, np.ndarray],
    target_accuracy: Optional[float],
) -> None:
    fields = (
        "dataset",
        "method",
        "seeds",
        "rounds",
        "final_cumulative_uplink_bytes_mean",
        "final_cumulative_uplink_bytes_std",
        "uplink_compression_vs_fedavg",
        "final_accuracy_mean",
        "final_accuracy_std",
        "target_accuracy",
        "seeds_reaching_target",
        "mean_uplink_bytes_to_target",
        "std_uplink_bytes_to_target",
    )
    fedavg_final = cumulative_by_method["fedavg_fp32"][:, -1].mean()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_NAMES:
            accuracy = accuracy_by_method[method]
            cumulative = cumulative_by_method[method]
            final_bytes = cumulative[:, -1].astype(np.float64)
            if target_accuracy is None:
                reached = np.asarray([], dtype=np.float64)
                target_value = ""
            else:
                target_values = _bytes_to_target(
                    accuracy, cumulative, target_accuracy
                )
                reached = target_values[np.isfinite(target_values)]
                target_value = target_accuracy
            writer.writerow(
                {
                    "dataset": dataset,
                    "method": method,
                    "seeds": ",".join(map(str, seeds)),
                    "rounds": accuracy.shape[1],
                    "final_cumulative_uplink_bytes_mean": float(final_bytes.mean()),
                    "final_cumulative_uplink_bytes_std": float(final_bytes.std()),
                    "uplink_compression_vs_fedavg": float(
                        fedavg_final / final_bytes.mean()
                    ),
                    "final_accuracy_mean": float(accuracy[:, -1].mean()),
                    "final_accuracy_std": float(accuracy[:, -1].std()),
                    "target_accuracy": target_value,
                    "seeds_reaching_target": int(reached.size),
                    "mean_uplink_bytes_to_target": (
                        float(reached.mean()) if reached.size else ""
                    ),
                    "std_uplink_bytes_to_target": (
                        float(reached.std()) if reached.size else ""
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: accuracy versus cumulative uplink payload"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["ravdess", "dronerf", "mnist"]
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/experiment1")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--unit", choices=UNIT_DIVISORS, default="mib")
    parser.add_argument("--xscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"--seeds contains duplicates: {args.seeds}")
    if args.target_accuracy is not None and not 0.0 <= args.target_accuracy <= 1.0:
        raise ValueError("--target-accuracy must lie in [0, 1]")

    output = args.output or (
        args.results_root
        / f"{args.dataset}_accuracy_vs_cumulative_uplink.pdf"
    )
    summary_output = args.summary_output or (
        args.results_root / f"{args.dataset}_communication_summary.csv"
    )

    accuracy_by_method: Dict[str, np.ndarray] = {}
    cumulative_by_method: Dict[str, np.ndarray] = {}
    reference_config = None
    reference_path = None
    for method in METHOD_NAMES:
        accuracy, cumulative, configs, paths = _collect_communication(
            args.results_root, args.dataset, method, args.seeds
        )
        accuracy_by_method[method] = accuracy
        cumulative_by_method[method] = cumulative
        for config, path in zip(configs, paths):
            if reference_config is None:
                reference_config, reference_path = config, path
                continue
            if _config_signature(config) != _config_signature(reference_config):
                difference = _describe_config_difference(reference_config, config)
                raise ValueError(
                    f"incomparable configurations: {reference_path} vs "
                    f"{path}: {difference}"
                )

    divisor, unit_label = UNIT_DIVISORS[args.unit]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for method in METHOD_NAMES:
        accuracy = accuracy_by_method[method]
        cumulative = cumulative_by_method[method]
        x = cumulative[0].astype(np.float64) / divisor
        mean = accuracy.mean(axis=0)
        std = accuracy.std(axis=0)
        marker_step = max(1, len(x) // 10)
        axis.plot(
            x,
            mean,
            label=LABELS[method],
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            markevery=marker_step,
            **STYLES[method],
        )
        if accuracy.shape[0] > 1:
            axis.fill_between(
                x,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=STYLES[method]["color"],
                alpha=0.12,
            )

    if args.target_accuracy is not None:
        axis.axhline(
            args.target_accuracy,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=f"Target accuracy = {args.target_accuracy:.3f}",
        )
    axis.set_xscale(args.xscale)
    axis.set_xlabel(f"Cumulative uplink payload ({unit_label})")
    axis.set_ylabel("Test accuracy")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    _write_summary(
        summary_output,
        args.dataset,
        list(args.seeds),
        accuracy_by_method,
        cumulative_by_method,
        args.target_accuracy,
    )
    print(
        f"validated methods={list(METHOD_NAMES)}, seeds={args.seeds}, "
        f"rounds={next(iter(accuracy_by_method.values())).shape[1]}"
    )
    print("uplink bytes are packed logical payload summed over all clients")
    print(f"saved {output}")
    print(f"saved {summary_output}")


if __name__ == "__main__":
    main()
