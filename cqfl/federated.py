"""Federated experiment-one runner integrated with the original BitFL core."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from BitMyConv_noMul import ComplexConv2D
from Bit2Communication import (
    PROTOCOL_VERSION,
    quantize_complex_update_np,
    quantize_real_update_np,
)
from BitFLCommunication import (
    PROTOCOL_VERSION as BITFL_PROTOCOL_VERSION,
    quantize_update_np as quantize_bitfl_update_np,
    topk_error_feedback as bitfl_topk_error_feedback,
)

from .ca4bit import CA4BitAdam
from .config import ExperimentConfig
from .data import DatasetBundle, load_dataset
from .models import build_model
from .quantization import (
    phase_quantize_unit_tf,
    phase_quantize_weight_np,
)


def set_determinism(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _bitfl_variable_masks(model):
    """Return explicit kernel/complex masks in model trainable-variable order."""
    bitfl_layers = [
        layer for layer in model.layers if isinstance(layer, ComplexConv2D)
    ]
    kernel_ids = {id(layer.kernel_fp) for layer in bitfl_layers}
    complex_ids = set(kernel_ids)
    complex_ids.update(
        id(layer.bias)
        for layer in bitfl_layers
        if getattr(layer, "bias", None) is not None
    )
    kernel_mask = [id(variable) in kernel_ids for variable in model.trainable_variables]
    complex_mask = [id(variable) in complex_ids for variable in model.trainable_variables]
    if sum(kernel_mask) != len(kernel_ids) or sum(complex_mask) != len(complex_ids):
        raise RuntimeError("failed to identify all BitFL variables in model state")
    for variable, is_complex in zip(model.trainable_variables, complex_mask):
        if is_complex and (
            len(variable.shape) < 1 or int(variable.shape[-1]) != 2
        ):
            raise RuntimeError(
                f"BitFL complex variable lacks a final real/imag axis: {variable.name}"
            )
    return kernel_mask, complex_mask


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


def _quantized_upload(
    delta: np.ndarray,
    method: str,
    is_complex: bool,
    rng: np.random.Generator = None,
    bitfl_normalization_bound: float = 1.0,
    bitfl_bit_flip_probability: float = 0.0,
):
    """Encode the message that is actually reconstructed by the server.

    The legacy BitFL demos upload FP32 master weights and contain no low-bit
    federated codec.  CQ-FL supplies that missing boundary here: every
    trainable client delta is physically packed to two bits per scalar/complex
    element and decoded before weighted FedAvg aggregation.  Complex tensors
    use the original four-axis phase alphabet; real tensors use its real-axis
    subset.  Non-trainable state is handled separately in FP32.
    """
    delta = np.asarray(delta, dtype=np.float32)
    if method == "bitfl":
        if rng is None:
            raise ValueError("BitFL stochastic quantization requires an explicit RNG")
        reconstructed, payload = quantize_bitfl_update_np(
            delta,
            rng,
            bitfl_normalization_bound,
            bitfl_bit_flip_probability,
        )
        return reconstructed, payload, "bitfl_1bit"
    if method == "cqfl":
        if is_complex:
            reconstructed, payload = quantize_complex_update_np(delta)
            return reconstructed, payload, "complex_2bit"
        reconstructed, payload = quantize_real_update_np(delta)
        return reconstructed, payload, "real_2bit"
    if method == "signsgd":
        scale = float(np.mean(np.abs(delta), dtype=np.float64)) if delta.size else 0.0
        reconstructed = np.sign(delta).astype(np.float32) * scale
        return reconstructed, (delta.size + 7) // 8 + 4, "sign_1bit"
    return delta, int(delta.nbytes), "fp32"


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
        _kernel_mask, self.complex_trainable_mask = _bitfl_variable_masks(model)
        if len(self.complex_trainable_mask) != len(model.trainable_variables):
            raise RuntimeError("complex gradient mask does not match trainable variables")
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
                # The optimizer is deliberately executed on CPU in this
                # offload trainer.  XLA-compiling one update function for each
                # client adds substantial tracing cost (especially for the 10
                # MNIST clients) and produced repeated _update_step_xla traces.
                # Disabling JIT changes execution only, not the Adam equations.
                self.optimizer = tf.keras.optimizers.Adam(
                    config.learning_rate, jit_compile=False
                )

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
        if len(gradients) != len(self.complex_trainable_mask):
            raise ValueError("gradient list does not match the BitFL model")
        for gradient, is_complex in zip(gradients, self.complex_trainable_mask):
            if gradient is None:
                prepared.append(None)
                continue
            # The original layer already quantizes Conv kernels through its
            # custom gradient.  Applying the same unit map here also covers
            # complex biases while leaving real Dense/BN tensors untouched.
            if self.method == "cqfl" and is_complex:
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
                            gradient_array = np.asarray(gradient.numpy(), dtype=np.float32)
                            # CA4BitAdam performs its state update with NumPy.
                            # Passing the array directly avoids converting the
                            # same gradient Tensor GPU -> CPU Tensor -> NumPy.
                            if self.method == "cqfl":
                                cpu_gradient = gradient_array
                            else:
                                cpu_gradient = tf.convert_to_tensor(
                                    gradient_array, dtype=master.dtype
                                )
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

    global_model = build_model(
        bundle.input_shape,
        bundle.num_classes,
        config.method,
        config.model_profile,
    )
    _ = global_model(tf.zeros((1, *bundle.input_shape), tf.float32), training=False)
    global_trainable = [np.asarray(v.numpy(), np.float32) for v in global_model.trainable_variables]
    global_non_trainable = [
        np.asarray(v.numpy(), np.float32) for v in global_model.non_trainable_variables
    ]
    # Explicit masks are safer than classifying arbitrary tensors by a final
    # dimension of length two: a real Dense layer may legitimately have two
    # outputs.  ComplexConv kernels and biases are the only complex variables
    # in the shared architecture; all other trainable tensors use the real
    # 2-bit message codec for CQ-FL uploads.
    bitfl_kernel_mask, bitfl_complex_mask = _bitfl_variable_masks(global_model)
    if len(bitfl_complex_mask) != len(global_trainable):
        raise RuntimeError("complex upload mask does not match trainable state")
    if config.method == "cqfl" and (
        not any(bitfl_complex_mask) or all(bitfl_complex_mask)
    ):
        raise RuntimeError(
            "CQ-FL requires both complex and real trainable tensors for its "
            "two-codec uplink protocol"
        )

    clients: List[BitFLOffloadTrainer] = []
    for client_id in range(config.clients):
        set_determinism(config.seed + client_id)
        model = build_model(
            bundle.input_shape,
            bundle.num_classes,
            config.method,
            config.model_profile,
        )
        _ = model(tf.zeros((1, *bundle.input_shape), tf.float32), training=False)
        trainer = BitFLOffloadTrainer(model, config.method, config)
        client_shapes = [
            tuple(int(dimension) for dimension in variable.shape)
            for variable in model.trainable_variables
        ]
        global_shapes = [tuple(value.shape) for value in global_trainable]
        if client_shapes != global_shapes:
            raise RuntimeError("client/global trainable variable layouts differ")
        if trainer.complex_trainable_mask != bitfl_complex_mask:
            raise RuntimeError("client/global complex variable masks differ")
        clients.append(trainer)

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
        "uplink_protocol",
        "uplink_bytes",
        "uplink_trainable_bytes",
        "uplink_complex_2bit_bytes",
        "uplink_real_2bit_bytes",
        "uplink_bitfl_1bit_bytes",
        "uplink_non_trainable_bytes",
        "downlink_bytes",
        "optimizer_state_bytes",
        "round_seconds",
    ]
    history = []
    bitfl_error = [np.zeros_like(value) for value in global_trainable]
    cqfl_uplink_error = [
        [np.zeros_like(value) for value in global_trainable]
        for _ in range(config.clients)
    ]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for round_index in range(1, config.rounds + 1):
            round_started = time.perf_counter()
            broadcast_trainable, trainable_downlink = _quantized_broadcast(
                global_trainable, config.method, bitfl_kernel_mask
            )
            # BN moving statistics and other non-trainable state stay FP32.
            broadcast_non_trainable = [value.copy() for value in global_non_trainable]
            non_trainable_downlink = sum(value.nbytes for value in broadcast_non_trainable)

            trainable_deltas, non_trainable_deltas = [], []
            sample_counts, losses = [], []
            uplink_bytes = 0
            uplink_trainable_bytes = 0
            uplink_complex_2bit_bytes = 0
            uplink_real_2bit_bytes = 0
            uplink_bitfl_1bit_bytes = 0
            uplink_non_trainable_bytes = 0
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
                bitfl_rng = np.random.default_rng(
                    config.seed + round_index * 1_000_003 + client_id * 10_007
                )
                for tensor_index, (local, base, is_complex) in enumerate(zip(
                    local_trainable,
                    broadcast_trainable,
                    bitfl_complex_mask,
                )):
                    raw_delta = local - base
                    upload_delta = raw_delta
                    if config.method == "cqfl" and config.cqfl_uplink_error_feedback:
                        upload_delta = (
                            raw_delta + cqfl_uplink_error[client_id][tensor_index]
                        )
                    reconstructed, payload, encoding = _quantized_upload(
                        upload_delta,
                        config.method,
                        is_complex,
                        rng=bitfl_rng,
                        bitfl_normalization_bound=config.bitfl_normalization_bound,
                        bitfl_bit_flip_probability=config.bitfl_bit_flip_probability,
                    )
                    if config.method == "cqfl" and config.cqfl_uplink_error_feedback:
                        cqfl_uplink_error[client_id][tensor_index] = (
                            upload_delta - reconstructed
                        ).astype(np.float32, copy=False)
                    encoded_trainable.append(reconstructed)
                    uplink_bytes += payload
                    uplink_trainable_bytes += payload
                    if encoding == "complex_2bit":
                        uplink_complex_2bit_bytes += payload
                    elif encoding == "real_2bit":
                        uplink_real_2bit_bytes += payload
                    elif encoding == "bitfl_1bit":
                        uplink_bitfl_1bit_bytes += payload
                encoded_non_trainable = []
                for local, base in zip(local_non_trainable, broadcast_non_trainable):
                    delta = local - base
                    encoded_non_trainable.append(delta)
                    uplink_bytes += delta.nbytes
                    uplink_non_trainable_bytes += delta.nbytes
                trainable_deltas.append(encoded_trainable)
                non_trainable_deltas.append(encoded_non_trainable)
                sample_counts.append(len(y_client))

            aggregated_trainable = [
                _weighted_mean(trainable_deltas, sample_counts, index)
                for index in range(len(broadcast_trainable))
            ]
            if config.method == "bitfl":
                residual = (
                    bitfl_error
                    if config.bitfl_error_feedback
                    else [np.zeros_like(value) for value in aggregated_trainable]
                )
                aggregated_trainable, next_bitfl_error = bitfl_topk_error_feedback(
                    aggregated_trainable,
                    residual,
                    config.bitfl_topk_fraction,
                )
                bitfl_error = (
                    next_bitfl_error
                    if config.bitfl_error_feedback
                    else [np.zeros_like(value) for value in aggregated_trainable]
                )
            global_trainable = [
                base + update
                for base, update in zip(broadcast_trainable, aggregated_trainable)
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
                "uplink_protocol": (
                    PROTOCOL_VERSION
                    if config.method == "cqfl"
                    else BITFL_PROTOCOL_VERSION
                    if config.method == "bitfl"
                    else config.method
                ),
                "uplink_bytes": int(uplink_bytes),
                "uplink_trainable_bytes": int(uplink_trainable_bytes),
                "uplink_complex_2bit_bytes": int(uplink_complex_2bit_bytes),
                "uplink_real_2bit_bytes": int(uplink_real_2bit_bytes),
                "uplink_bitfl_1bit_bytes": int(uplink_bitfl_1bit_bytes),
                "uplink_non_trainable_bytes": int(uplink_non_trainable_bytes),
                "downlink_bytes": int(
                    (trainable_downlink + non_trainable_downlink) * config.clients
                ),
                "optimizer_state_bytes": int(
                    sum(client.optimizer_state_bytes for client in clients)
                ),
                "round_seconds": time.perf_counter() - round_started,
            }
            writer.writerow(row)
            handle.flush()
            history.append(row)
            print(
                f"[{config.dataset}/{config.method}] round {round_index:03d}/{config.rounds}: "
                f"loss={test_loss:.4f}, acc={test_accuracy:.4f}, "
                f"time={row['round_seconds']:.1f}s"
            )

    summary = {
        "config": asdict(config),
        "implementation": {
            "core": "BitMyConv_noMul.ComplexConv2D",
            "weight_quantizer": "original unit {+1,+i,-1,-i}",
            "federated_update": "FedAvg client delta",
            "uplink_protocol_version": (
                PROTOCOL_VERSION
                if config.method == "cqfl"
                else BITFL_PROTOCOL_VERSION
                if config.method == "bitfl"
                else config.method
            ),
            "complex_uplink": (
                "packed 2-bit phase code + one FP32 tensor scale"
                if config.method == "cqfl"
                else "not used by this method"
            ),
            "real_uplink": (
                "packed 2-bit real-axis code + one FP32 tensor scale"
                if config.method == "cqfl"
                else "not used by this method"
            ),
            "non_trainable_uplink": "FP32",
            "optimizer": "CA4BitAdam" if config.method == "cqfl" else config.method,
            "cqfl_uplink_error_feedback": (
                "per-client residual across communication rounds"
                if config.method == "cqfl" and config.cqfl_uplink_error_feedback
                else "disabled"
            ),
            "bitfl_baseline": (
                {
                    "privacy_perturbation": (
                        "disabled (BitFL without DP)"
                        if config.bitfl_bit_flip_probability == 0.0
                        else "independent post-quantization bit flip"
                    ),
                    "bit_flip_probability": config.bitfl_bit_flip_probability,
                    "stochastic_quantization": "Eq. (5), physically packed 1-bit",
                    "normalization_bound": config.bitfl_normalization_bound,
                    "paper_compression_rate_0_8": "not used; paper does not define it reproducibly",
                    "topk_fraction": config.bitfl_topk_fraction,
                    "error_feedback": (
                        "server residual across communication rounds"
                        if config.bitfl_error_feedback
                        else "disabled; unselected top-k residual is discarded"
                    ),
                }
                if config.method == "bitfl"
                else "not used by this method"
            ),
        },
        "best_test_accuracy": max(row["test_accuracy"] for row in history),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "parameter_count": int(global_model.count_params()),
        "notes": [
            "All five methods use one complex architecture and parameter layout.",
            (
                "BitFL uses FP32 local Adam, stochastic packed 1-bit uplink, "
                f"top-k={config.bitfl_topk_fraction:g}, "
                f"bit-flip={config.bitfl_bit_flip_probability:g}, and "
                f"error-feedback={config.bitfl_error_feedback}."
                if config.method == "bitfl"
                else "BitFL-only controls are not used by this method."
            ),
            "2-bit W and CQ-FL share the original BitFL no-multiply forward operator.",
            "CA-4bit preserves component-wise Adam second-moment equations.",
            "The legacy demos upload FP32 masters; Bit2Communication supplies the missing packed federated codec.",
            "CQ-FL uploads every trainable FedAvg client delta at 2-bit; complex and real tensors use separate decoders.",
            "BatchNorm moving statistics and other non-trainable client state remain FP32.",
            "Logical low-bit payloads are packed; TensorFlow compute shadows remain FP32.",
        ],
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return output
