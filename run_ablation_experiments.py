"""Run the draft's controlled experiments 2 and 3 on RAVDESS by default."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from cqfl.config import ABLATION_VARIANTS, MODEL_PROFILES, ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled CQ-FL ablations for experiments 2 and 3"
    )
    parser.add_argument("--experiment", type=int, required=True, choices=[2, 3])
    parser.add_argument(
        "--variant",
        default="all",
        help="all, or one experiment-specific variant shown with --list-variants",
    )
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument(
        "--dataset", default="ravdess", choices=["ravdess", "dronerf", "mnist"]
    )
    parser.add_argument("--data-path", default="")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--local-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--model-profile", choices=MODEL_PROFILES, default="standard")
    parser.add_argument("--cqfl-uplink-error-feedback", action="store_true")
    parser.add_argument("--cqfl-restore-best", action="store_true")
    parser.add_argument("--cqfl-reduce-lr-patience", type=int, default=0)
    parser.add_argument("--cqfl-reduce-lr-factor", type=float, default=0.5)
    parser.add_argument("--cqfl-min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--cqfl-early-stopping-patience", type=int, default=0)
    parser.add_argument("--cqfl-early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available = ABLATION_VARIANTS[args.experiment]
    if args.list_variants:
        print(" ".join(available))
        return
    if args.variant == "all":
        variants = tuple(available)
    elif args.variant in available:
        variants = (args.variant,)
    else:
        raise ValueError(
            f"experiment {args.experiment} variants are {list(available)}; "
            f"got {args.variant!r}"
        )

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required to run the ablation training; use the "
            "same server environment as experiment 1"
        ) from exc
    from cqfl.federated import run

    for variant in variants:
        controls = available[variant]
        config = ExperimentConfig(
            dataset=args.dataset,
            method="cqfl",
            data_path=args.data_path,
            output_root=str(
                args.output_root / f"experiment{args.experiment}" / variant
            ),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            block_size=args.block_size,
            seed=args.seed,
            clients=args.clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            model_profile=args.model_profile,
            cqfl_uplink_error_feedback=args.cqfl_uplink_error_feedback,
            cqfl_restore_best=args.cqfl_restore_best,
            cqfl_reduce_lr_patience=args.cqfl_reduce_lr_patience,
            cqfl_reduce_lr_factor=args.cqfl_reduce_lr_factor,
            cqfl_min_learning_rate=args.cqfl_min_learning_rate,
            cqfl_early_stopping_patience=args.cqfl_early_stopping_patience,
            cqfl_early_stopping_min_delta=args.cqfl_early_stopping_min_delta,
            **controls,
        )
        output = run(config)
        print(f"completed experiment {args.experiment}/{variant}: {output}")
        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == "__main__":
    main()
