"""CLI for the draft's experiment one: accuracy versus communication rounds."""

from __future__ import annotations

import argparse
import gc

import tensorflow as tf

from cqfl.config import METHOD_NAMES, MODEL_PROFILES, ExperimentConfig
from cqfl.federated import run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ravdess", "dronerf", "mnist"])
    parser.add_argument("--method", default="all", choices=[*METHOD_NAMES, "all"])
    parser.add_argument("--data-path", default="")
    parser.add_argument("--output-root", default="results/experiment1")
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
    parser.add_argument("--bitfl-normalization-bound", type=float, default=1.0)
    parser.add_argument("--bitfl-topk-fraction", type=float, default=0.5)
    parser.add_argument("--bitfl-bit-flip-probability", type=float, default=0.0)
    parser.add_argument(
        "--bitfl-disable-error-feedback",
        dest="bitfl_error_feedback",
        action="store_false",
        help="discard the unselected BitFL top-k residual instead of carrying it across rounds",
    )
    parser.set_defaults(bitfl_error_feedback=True)
    parser.add_argument(
        "--cqfl-uplink-error-feedback",
        action="store_true",
        help="carry each client's 2-bit uplink quantization residual into the next round",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    methods = METHOD_NAMES if args.method == "all" else (args.method,)
    for method in methods:
        config = ExperimentConfig(
            dataset=args.dataset,
            method=method,
            data_path=args.data_path,
            output_root=args.output_root,
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
            bitfl_normalization_bound=args.bitfl_normalization_bound,
            bitfl_topk_fraction=args.bitfl_topk_fraction,
            bitfl_bit_flip_probability=args.bitfl_bit_flip_probability,
            bitfl_error_feedback=args.bitfl_error_feedback,
            cqfl_uplink_error_feedback=args.cqfl_uplink_error_feedback,
        )
        output = run(config)
        print(f"completed: {output}")
        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == "__main__":
    main()
