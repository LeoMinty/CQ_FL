"""CLI for the draft's experiment one: accuracy versus communication rounds."""

from __future__ import annotations

import argparse
import gc

import tensorflow as tf

from cqfl.config import METHOD_NAMES, ExperimentConfig
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
        )
        output = run(config)
        print(f"completed: {output}")
        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == "__main__":
    main()
