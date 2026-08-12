"""Analyze persistent training-state storage for CQ-FL experiment 5.

This report deliberately separates two different claims:

* ``prototype``: persistent state retained by the current TensorFlow trainer,
  including its FP32 master weights;
* ``packed_target``: the algorithmic deployment format in which every
  trainable tensor is physically packed to two bits plus one FP32 tensor scale.

Transient gradients, activations, decoded work buffers, Python objects and
non-trainable BatchNorm moving statistics are outside both definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


METHODS: Tuple[str, ...] = (
    "fedavg_fp32",
    "bitfl",
    "signsgd",
    "w2_fp32_adam",
    "cqfl",
)

DATASET_MODELS = {
    "ravdess": ((128, 192, 1, 2), 8, "standard"),
    "dronerf": ((64, 32, 1, 2), 4, "standard"),
    "mnist": ((28, 28, 1), 10, "mnist_small"),
}


@dataclass(frozen=True)
class VariableSpec:
    name: str
    shape: Tuple[int, ...]
    scalar_count: int
    complex_parameter: bool


@dataclass(frozen=True)
class StorageRow:
    dataset: str
    model_profile: str
    method: str
    representation: str
    trainable_scalar_parameters: int
    weight_bytes: int
    optimizer_bytes: int
    total_bytes: int
    bytes_per_parameter: float


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def packed_2bit_weight_bytes(spec: VariableSpec) -> int:
    """Two-bit buffer plus one FP32 scale for one trainable tensor."""
    if spec.complex_parameter:
        if not spec.shape or spec.shape[-1] != 2:
            raise ValueError(f"invalid complex shape for {spec.name}: {spec.shape}")
        logical_values = spec.scalar_count // 2
    else:
        logical_values = spec.scalar_count
    return _ceil_div(logical_values, 4) + 4


def block4_bytes(count: int, block_size: int) -> int:
    """Four-bit values plus one FP32 scale per block."""
    return _ceil_div(count, 2) + 4 * _ceil_div(count, block_size)


def ca4_moment_bytes(spec: VariableSpec, block_size: int) -> Tuple[int, int]:
    """Return the exact persistent first- and second-moment byte counts."""
    if spec.complex_parameter:
        complex_count = spec.scalar_count // 2
        first = (
            _ceil_div(complex_count, 4)
            + _ceil_div(complex_count, 2)
            + 4 * _ceil_div(complex_count, block_size)
        )
        second = block4_bytes(spec.scalar_count, block_size)
    else:
        first = block4_bytes(spec.scalar_count, block_size)
        second = block4_bytes(spec.scalar_count, block_size)
    return first, second


def summarize_storage(
    dataset: str,
    model_profile: str,
    specs: Sequence[VariableSpec],
    block_size: int,
) -> Tuple[List[StorageRow], dict]:
    parameter_count = sum(spec.scalar_count for spec in specs)
    fp32_weights = 4 * parameter_count
    fp32_adam = 8 * parameter_count
    packed_weights = sum(packed_2bit_weight_bytes(spec) for spec in specs)
    moments = [ca4_moment_bytes(spec, block_size) for spec in specs]
    ca4_first = sum(first for first, _ in moments)
    ca4_second = sum(second for _, second in moments)
    ca4_total = ca4_first + ca4_second

    definitions = {
        "fedavg_fp32": {
            "prototype": (fp32_weights, fp32_adam),
            "packed_target": (fp32_weights, fp32_adam),
        },
        "bitfl": {
            "prototype": (fp32_weights, fp32_adam),
            "packed_target": (fp32_weights, fp32_adam),
        },
        "signsgd": {
            "prototype": (fp32_weights, 0),
            "packed_target": (fp32_weights, 0),
        },
        "w2_fp32_adam": {
            "prototype": (fp32_weights, fp32_adam),
            "packed_target": (packed_weights, fp32_adam),
        },
        "cqfl": {
            "prototype": (fp32_weights, ca4_total),
            "packed_target": (packed_weights, ca4_total),
        },
    }

    rows: List[StorageRow] = []
    for method in METHODS:
        for representation in ("prototype", "packed_target"):
            weight_bytes, optimizer_bytes = definitions[method][representation]
            total_bytes = weight_bytes + optimizer_bytes
            rows.append(
                StorageRow(
                    dataset=dataset,
                    model_profile=model_profile,
                    method=method,
                    representation=representation,
                    trainable_scalar_parameters=parameter_count,
                    weight_bytes=weight_bytes,
                    optimizer_bytes=optimizer_bytes,
                    total_bytes=total_bytes,
                    bytes_per_parameter=total_bytes / parameter_count,
                )
            )

    details = {
        "dataset": dataset,
        "model_profile": model_profile,
        "trainable_scalar_parameters": parameter_count,
        "trainable_tensor_count": len(specs),
        "complex_tensor_count": sum(spec.complex_parameter for spec in specs),
        "real_tensor_count": sum(not spec.complex_parameter for spec in specs),
        "fp32_weight_bytes": fp32_weights,
        "fp32_adam_moment_bytes": fp32_adam,
        "packed_2bit_weight_bytes": packed_weights,
        "ca4_first_moment_bytes": ca4_first,
        "ca4_second_moment_bytes": ca4_second,
        "ca4_total_moment_bytes": ca4_total,
        "block_size": block_size,
        "variables": [asdict(spec) for spec in specs],
    }
    return rows, details


def build_variable_specs(
    input_shape: Sequence[int], num_classes: int, model_profile: str
) -> List[VariableSpec]:
    """Build the real experiment model and reuse its explicit complex mask."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to construct the experiment models. "
            "Run this script in the same environment used for training."
        ) from error

    from cqfl.federated import _bitfl_variable_masks
    from cqfl.models import build_model

    model = build_model(input_shape, num_classes, "fedavg_fp32", model_profile)
    model(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
    _, complex_mask = _bitfl_variable_masks(model)
    if len(complex_mask) != len(model.trainable_variables):
        raise RuntimeError("complex mask does not match the trainable-variable layout")

    specs = []
    for variable, is_complex in zip(model.trainable_variables, complex_mask):
        shape = tuple(int(dimension) for dimension in variable.shape)
        scalar_count = math.prod(shape)
        if is_complex and (not shape or shape[-1] != 2):
            raise RuntimeError(
                f"explicitly complex variable lacks a final real/imag axis: "
                f"{variable.name} {shape}"
            )
        specs.append(
            VariableSpec(
                name=variable.name,
                shape=shape,
                scalar_count=scalar_count,
                complex_parameter=bool(is_complex),
            )
        )
    return specs


def _write_csv(path: Path, rows: Iterable[StorageRow]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def select_experiment5_rows(rows: Sequence[StorageRow]) -> List[StorageRow]:
    """Select the single storage definition reported by experiment 5.

    Quantized-weight methods use the packed persistent-weight definition;
    FP32/SignSGD baselines retain their native FP32 persistent weights.
    """
    representation = {
        "fedavg_fp32": "prototype",
        "bitfl": "prototype",
        "signsgd": "prototype",
        "w2_fp32_adam": "packed_target",
        "cqfl": "packed_target",
    }
    selected = []
    for dataset in dict.fromkeys(row.dataset for row in rows):
        dataset_rows = [row for row in rows if row.dataset == dataset]
        for method in METHODS:
            selected.append(
                next(
                    row
                    for row in dataset_rows
                    if row.method == method
                    and row.representation == representation[method]
                )
            )
    return selected


def _write_markdown(path: Path, rows: Sequence[StorageRow]) -> None:
    labels = {
        "fedavg_fp32": "FedAvg (FP32)",
        "bitfl": "BitFL",
        "signsgd": "SignSGD",
        "w2_fp32_adam": "2-bit W + FP32 Adam",
        "cqfl": "CQ-FL",
    }
    lines = [
        "# Experiment 5: persistent training-state storage",
        "",
        "This primary experiment table uses physically packed 2-bit persistent "
        "weights for the two quantized-weight methods and FP32 persistent weights "
        "for the FP32/SignSGD baselines.",
        "",
        "Transient gradients, activations, decoded work buffers, Python objects "
        "and non-trainable BatchNorm moving state are excluded.",
        "",
    ]
    for dataset in dict.fromkeys(row.dataset for row in rows):
        selected = [row for row in rows if row.dataset == dataset]
        profile = selected[0].model_profile
        parameter_count = selected[0].trainable_scalar_parameters
        lines.extend(
            [
                f"## {dataset} ({profile})",
                "",
                f"Trainable scalar parameters: `{parameter_count:,}`",
                "",
                "| Method | Weight MiB | Optimizer MiB | Total MiB | B/param |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                f"| {labels[row.method]} | "
                f"{row.weight_bytes / 2**20:.3f} | "
                f"{row.optimizer_bytes / 2**20:.3f} | "
                f"{row.total_bytes / 2**20:.3f} | "
                f"{row.bytes_per_parameter:.6f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(path: Path, rows: Sequence[StorageRow]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = list(dict.fromkeys(row.dataset for row in rows))
    figure, axes = plt.subplots(
        1, len(datasets), figsize=(6.2 * len(datasets), 4.8), squeeze=False
    )
    x = np.arange(len(METHODS))
    for axis, dataset in zip(axes[0], datasets):
        selected = [row for row in rows if row.dataset == dataset]
        values = [
            next(
                row.bytes_per_parameter
                for row in selected
                if row.method == method
            )
            for method in METHODS
        ]
        bars = axis.bar(x, values, width=0.62)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
        axis.set_title(dataset)
        axis.set_xticks(
            x, ("FedAvg", "BitFL", "SignSGD", "2-bit W", "CQ-FL"), rotation=20
        )
        axis.set_ylabel("Persistent-state bytes / trainable parameter")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_MODELS),
        default=list(DATASET_MODELS),
    )
    parser.add_argument(
        "--mnist-profile",
        choices=("standard", "mnist_small"),
        default="mnist_small",
        help="MNIST profile to report; formal experiments must use the same profile.",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--output-dir", default="results/experiment5_storage")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("block size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[StorageRow] = []
    all_details = []
    for dataset in args.datasets:
        input_shape, num_classes, default_profile = DATASET_MODELS[dataset]
        model_profile = args.mnist_profile if dataset == "mnist" else default_profile
        specs = build_variable_specs(input_shape, num_classes, model_profile)
        rows, details = summarize_storage(
            dataset, model_profile, specs, args.block_size
        )
        all_rows.extend(rows)
        all_details.append(details)
        packed_cqfl = next(
            row for row in rows
            if row.method == "cqfl" and row.representation == "packed_target"
        )
        print(
            f"{dataset}/{model_profile}: "
            f"P={packed_cqfl.trainable_scalar_parameters:,}, "
            f"CQ-FL={packed_cqfl.bytes_per_parameter:.6f} B/param"
        )

    selected_rows = select_experiment5_rows(all_rows)
    _write_csv(output_dir / "storage_summary.csv", selected_rows)
    _write_csv(output_dir / "storage_all_definitions.csv", all_rows)
    (output_dir / "storage_details.json").write_text(
        json.dumps(all_details, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(output_dir / "storage_summary.md", selected_rows)
    if not args.no_plot:
        _write_plot(output_dir / "storage_bars.png", selected_rows)
    print(f"completed: {output_dir}")


if __name__ == "__main__":
    main()
