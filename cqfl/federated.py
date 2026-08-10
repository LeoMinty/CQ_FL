"""Federated experiment-one runner integrated with the original BitFL core."""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from BitMyConv_noMul import ComplexConv2D

from .ca4bit import CA4BitAdam
from .config import ExperimentConfig
from .data import DatasetBundle, load_dataset
from .models import build_model
from .quantization import (
    phase_quantize_unit_tf,
    phase_quantize_weight_np,
    quantize_complex_delta_np,
)


def set_determinism(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _is_complex_array(array: np.ndarray) -> bool:
    return array.ndim >= 1 and array.shape[-1] == 2


def _is_complex_variable(variable) -> bool:
    shape = tuple(int(dimension) for dimension in variable.shape)
    return len(shape) >= 1 and shape[-1] == 2


def _quantized_broadcast(
    weights: Sequence[np.ndarray],
    method: str,
    quantize_mask: Sequence[bool],
):
    """Encode the server-to-client state using the original BitFL phase map."""
    output, payload = [], 0
    if len(weights) != len(quantize_mask):
        raise ValueError("broadcast quantization mask does not match model state")
    for value, is_bitfl_kernel in zip(weights, quantize_mask):
        value = np.asarray(value, dtype=np.float32)
        if method in {"w2_fp32_adam", "cqfl"} and is_bitfl_kernel:
            quantized, codes = phase_quantize_weight_np(value)
            output.append(quantized)
            payload += (codes.size + 3) // 4
        else:
            output.append(value.copy())
            payload += value.nbytes
    return output, int(payload)


def _quantized_upload(delta: np.ndarray, method: str):
    """Encode a FedAvg client update; CQ-FL compresses complex tensors to 2-bit."""
    delta = np.asarray(delta, dtype=np.float32)
    if method == "cqfl" and _is_complex_array(delta):
        return quantize_complex_delta_np(delta)
    if method == "signsgd":
        scale = float(np.mean(np.abs(delta), dtype=np.float64)) if delta.size else 0.0
        reconstructed = np.sign(delta).astype(np.float32) * scale
        return reconstructed, (delta.size + 7) // 8 + 4
    return delta, int(delta.nbytes)


class BitFLOffloadTrainer:
    """Original BitFL shadow-weight path with a selectable optimizer.

    Trainable FP32 master weights and optimizer states live on CPU.  The Keras
    model is the compute shadow and uses the original BitFL quantized Conv
    operator.  CA-4bit therefore replaces Adam state persistence without
    replacing the existing 2-bit forward/backward implementation.
    """

    def __init__(self, model, method: str, config: ExperimentConfig):
        self.model = model
        self.method = method
        self.learning_rate = float(config.learning_rate)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.device("/CPU:0"):
            self.master_weights = [
                tf.Variable(variable.numpy(), dtype=tf.float32, trainable=True)
                for variable in model.trainable_variables
            ]
            if method == "cqfl":
                self.optimizer = CA4BitAdam(
                    learning_rate=config.learning_rate,
                    block_size=config.block_size,
                )
            elif method == "signsgd":
                self.optimizer = None
            else:
                self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    def _sync_master_to_compute(self) -> None:
        for compute, master in zip(self.model.trainable_variables, self.master_weights):
            compute.assign(master)

    def set_state(
        self,
        trainable_weights: Sequence[np.ndarray],
        non_trainable_weights: Sequence[np.ndarray],
    ) -> None:
        if len(trainable_weights) != len(self.master_weights):
            raise ValueError("trainable state does not match the BitFL model")
        if len(non_trainable_weights) != len(self.model.non_trainable_variables):
            raise ValueError("non-trainable state does not match the BitFL model")
        with tf.device("/CPU:0"):
            for master, value in zip(self.master_weights, trainable_weights):
                master.assign(value)
        self._sync_master_to_compute()
        for variable, value in zip(self.model.non_trainable_variables, non_trainable_weights):
            variable.assign(value)

    def get_state(self):
        trainable = [np.asarray(weight.numpy(), dtype=np.float32) for weight in self.master_weights]
        non_trainable = [
            np.asarray(weight.numpy(), dtype=np.float32)
            for weight in self.model.non_trainable_variables
        ]
        return trainable, non_trainable

    def _prepare_gradients(self, gradients):
        prepared = []
        for gradient, variable in zip(gradients, self.model.trainable_variables):
            if gradient is None:
                prepared.append(None)
                continue
            # The original layer already quantizes Conv kernels through its
            # custom gradient.  Applying the same unit map here also covers
            # complex biases while leaving real Dense/BN tensors untouched.
            if self.method == "cqfl" and _is_complex_variable(variable):
                gradient = phase_quantize_unit_tf(gradient)
            prepared.append(gradient)
        return prepared

    def train(self, x, y, epochs: int, batch_size: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        losses = []
        self._sync_master_to_compute()
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size]
                xb = tf.convert_to_tensor(x[index], dtype=tf.float32)
                yb = tf.convert_to_tensor(y[index], dtype=tf.int64)
                with tf.GradientTape() as tape:
                    logits = self.model(xb, training=True)
                    loss = self.loss_fn(yb, logits)
                    if self.model.losses:
                        loss += tf.add_n(self.model.losses)
                gradients = self._prepare_gradients(
                    tape.gradient(loss, self.model.trainable_variables)
                )
                pairs = []
                with tf.device("/CPU:0"):
                    for gradient, master in zip(gradients, self.master_weights):
                        if gradient is not None:
                            cpu_gradient = tf.convert_to_tensor(gradient.numpy(), dtype=master.dtype)
                            pairs.append((cpu_gradient, master))
                    if self.method == "signsgd":
                        for gradient, master in pairs:
                            scale = tf.reduce_mean(tf.abs(gradient))
                            master.assign_sub(self.learning_rate * tf.sign(gradient) * scale)
                    else:
                        self.optimizer.apply_gradients(pairs)
                self._sync_master_to_compute()
                losses.append(float(loss.numpy()))
        return float(np.mean(losses)) if losses else float("nan")

    @property
    def optimizer_state_bytes(self) -> int:
        if self.method == "cqfl":
            return self.optimizer.state_bytes()
        if self.optimizer is None:
            return 0
        variables = self.optimizer.variables
        variables = variables() if callable(variables) else variables
        total = 0
        for variable in variables:
            dtype = tf.as_dtype(variable.dtype)
            total += int(np.prod(variable.shape)) * dtype.size
        return int(total)


def _assign_model_state(model, trainable, non_trainable) -> None:
    for variable, value in zip(model.trainable_variables, trainable):
        variable.assign(value)
    for variable, value in zip(model.non_trainable_variables, non_trainable):
        variable.assign(value)


def evaluate(model, trainable, non_trainable, x, y, batch_size: int) -> Tuple[float, float]:
    _assign_model_state(model, trainable, non_trainable)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    correct = 0
    total_loss = 0.0
    for start in range(0, len(y), batch_size):
        xb = tf.convert_to_tensor(x[start : start + batch_size], dtype=tf.float32)
        yb = tf.convert_to_tensor(y[start : start + batch_size], dtype=tf.int64)
        logits = model(xb, training=False)
        total_loss += float(loss_fn(yb, logits).numpy()) * len(yb)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
        correct += int(tf.reduce_sum(tf.cast(predictions == yb, tf.int32)))
    return total_loss / len(y), correct / len(y)


def _apply_limits(bundle: DatasetBundle, config: ExperimentConfig) -> DatasetBundle:
    if config.max_train_samples:
        per_client = max(1, config.max_train_samples // len(bundle.clients))
        bundle.clients = [(x[:per_client], y[:per_client]) for x, y in bundle.clients]
    if config.max_test_samples:
        bundle.x_test = bundle.x_test[: config.max_test_samples]
        bundle.y_test = bundle.y_test[: config.max_test_samples]
    return bundle


def _weighted_mean(deltas, sample_counts, tensor_index: int):
    total = float(sum(sample_counts))
    return sum(
        client[tensor_index] * (sample_counts[cid] / total)
        for cid, client in enumerate(deltas)
    )


def run(config: ExperimentConfig) -> Path:
    config = config.resolved()
    set_determinism(config.seed)
    bundle = _apply_limits(
        load_dataset(config.dataset, config.data_path, config.clients, config.seed), config
    )

    global_model = build_model(bundle.input_shape, bundle.num_classes, config.method)
    _ = global_model(tf.zeros((1, *bundle.input_shape), tf.float32), training=False)
    global_trainable = [np.asarray(v.numpy(), np.float32) for v in global_model.trainable_variables]
    global_non_trainable = [
        np.asarray(v.numpy(), np.float32) for v in global_model.non_trainable_variables
    ]
    bitfl_kernel_ids = {
        id(layer.kernel_fp)
        for layer in global_model.layers
        if isinstance(layer, ComplexConv2D)
    }
    bitfl_kernel_mask = [
        id(variable) in bitfl_kernel_ids for variable in global_model.trainable_variables
    ]

    clients: List[BitFLOffloadTrainer] = []
    for client_id in range(config.clients):
        set_determinism(config.seed + client_id)
        model = build_model(bundle.input_shape, bundle.num_classes, config.method)
        _ = model(tf.zeros((1, *bundle.input_shape), tf.float32), training=False)
        clients.append(BitFLOffloadTrainer(model, config.method, config))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(config.output_root) / config.dataset / config.method / f"seed_{config.seed}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)

    fields = [
        "round",
        "train_loss",
        "test_loss",
        "test_accuracy",
        "uplink_bytes",
        "downlink_bytes",
        "optimizer_state_bytes",
    ]
    history = []
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for round_index in range(1, config.rounds + 1):
            broadcast_trainable, trainable_downlink = _quantized_broadcast(
                global_trainable, config.method, bitfl_kernel_mask
            )
            # BN moving statistics and other non-trainable state stay FP32.
            broadcast_non_trainable = [value.copy() for value in global_non_trainable]
            non_trainable_downlink = sum(value.nbytes for value in broadcast_non_trainable)

            trainable_deltas, non_trainable_deltas = [], []
            sample_counts, losses = [], []
            uplink_bytes = 0
            for client_id, (trainer, (x_client, y_client)) in enumerate(zip(clients, bundle.clients)):
                trainer.set_state(broadcast_trainable, broadcast_non_trainable)
                losses.append(
                    trainer.train(
                        x_client,
                        y_client,
                        config.local_epochs,
                        config.batch_size,
                        config.seed + round_index * 10_000 + client_id,
                    )
                )
                local_trainable, local_non_trainable = trainer.get_state()
                encoded_trainable = []
                for local, base in zip(local_trainable, broadcast_trainable):
                    reconstructed, payload = _quantized_upload(local - base, config.method)
                    encoded_trainable.append(reconstructed)
                    uplink_bytes += payload
                encoded_non_trainable = []
                for local, base in zip(local_non_trainable, broadcast_non_trainable):
                    delta = local - base
                    encoded_non_trainable.append(delta)
                    uplink_bytes += delta.nbytes
                trainable_deltas.append(encoded_trainable)
                non_trainable_deltas.append(encoded_non_trainable)
                sample_counts.append(len(y_client))

            global_trainable = [
                base + _weighted_mean(trainable_deltas, sample_counts, index)
                for index, base in enumerate(broadcast_trainable)
            ]
            global_non_trainable = [
                base + _weighted_mean(non_trainable_deltas, sample_counts, index)
                for index, base in enumerate(broadcast_non_trainable)
            ]
            test_loss, test_accuracy = evaluate(
                global_model,
                global_trainable,
                global_non_trainable,
                bundle.x_test,
                bundle.y_test,
                config.batch_size,
            )
            row = {
                "round": round_index,
                "train_loss": float(np.mean(losses)),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "uplink_bytes": int(uplink_bytes),
                "downlink_bytes": int(
                    (trainable_downlink + non_trainable_downlink) * config.clients
                ),
                "optimizer_state_bytes": int(
                    sum(client.optimizer_state_bytes for client in clients)
                ),
            }
            writer.writerow(row)
            handle.flush()
            history.append(row)
            print(
                f"[{config.dataset}/{config.method}] round {round_index:03d}/{config.rounds}: "
                f"loss={test_loss:.4f}, acc={test_accuracy:.4f}"
            )

    summary = {
        "config": asdict(config),
        "implementation": {
            "core": "BitMyConv_noMul.ComplexConv2D",
            "weight_quantizer": "original unit {+1,+i,-1,-i}",
            "federated_update": "FedAvg client delta",
            "optimizer": "CA4BitAdam" if config.method == "cqfl" else config.method,
        },
        "best_test_accuracy": max(row["test_accuracy"] for row in history),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "parameter_count": int(global_model.count_params()),
        "notes": [
            "All four methods use one complex architecture and parameter layout.",
            "2-bit W and CQ-FL share the original BitFL no-multiply forward operator.",
            "CA-4bit preserves component-wise Adam second-moment equations.",
            "CQ-FL uploads a phase-quantized FedAvg client update, not a per-batch raw gradient.",
            "Logical low-bit payloads are packed; TensorFlow compute shadows remain FP32.",
        ],
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return output
