"""Dataset loading and deterministic client partitions for experiment one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class DatasetBundle:
    clients: List[Tuple[np.ndarray, np.ndarray]]
    x_test: np.ndarray
    y_test: np.ndarray
    input_shape: Tuple[int, ...]
    num_classes: int


def _limit(x, y, maximum: int):
    return (x, y) if not maximum else (x[:maximum], y[:maximum])


def _standardize_from_train(x_train: np.ndarray, x_test: np.ndarray):
    axes = tuple(range(x_train.ndim - 1))
    mean = np.mean(x_train, axis=axes, keepdims=True, dtype=np.float64)
    std = np.std(x_train, axis=axes, keepdims=True, dtype=np.float64)
    return (
        ((x_train - mean) / (std + 1e-6)).astype(np.float32),
        ((x_test - mean) / (std + 1e-6)).astype(np.float32),
    )


def _group_train_test_indices(groups: np.ndarray, labels: np.ndarray, seed: int, test_ratio=0.2):
    """Split physical recording groups, never derived windows, to avoid leakage."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    unique_groups = np.unique(groups)
    group_labels = np.asarray([labels[np.flatnonzero(groups == g)[0]] for g in unique_groups])
    test_groups = []
    for label in np.unique(group_labels):
        candidates = unique_groups[group_labels == label].copy()
        rng.shuffle(candidates)
        count = max(1, int(round(len(candidates) * test_ratio)))
        test_groups.extend(candidates[:count].tolist())
    is_test = np.isin(groups, np.asarray(test_groups))
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def _assign_groups_to_clients(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_clients: int,
    seed: int,
):
    """Label-stratified round-robin allocation of whole recording groups."""
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(num_clients)]
    unique_groups = np.unique(groups)
    labels_per_group = [np.unique(y[groups == group]) for group in unique_groups]
    if any(len(values) > 1 for values in labels_per_group):
        # RAVDESS actors contain all emotions: allocate whole speakers directly.
        candidates = unique_groups.copy()
        rng.shuffle(candidates)
        for offset, group in enumerate(candidates):
            buckets[offset % num_clients].append(group)
    else:
        group_labels = np.asarray([values[0] for values in labels_per_group])
        for label in np.unique(group_labels):
            candidates = unique_groups[group_labels == label].copy()
            rng.shuffle(candidates)
            for offset, group in enumerate(candidates):
                buckets[offset % num_clients].append(group)
    clients = []
    for bucket in buckets:
        index = np.flatnonzero(np.isin(groups, np.asarray(bucket)))
        clients.append((x[index], y[index]))
    return clients


def load_ravdess(path: str, num_clients: int, seed: int) -> DatasetBundle:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"RAVDESS file not found: {source}. Run prepare_ravdess.py with CLASS_TYPE=3 first."
        )
    data = np.load(source, allow_pickle=False)
    x = np.asarray(data["Xc"], dtype=np.float32)
    y = np.asarray(data["Y"], dtype=np.int64)
    if int(data["n_classes"]) != 8:
        raise ValueError("experiment one requires RAVDESS c3 (8 emotion classes)")
    if "actor" in data:
        groups = np.asarray(data["actor"], dtype=np.int64)
    else:
        # Compatibility with the old preprocessor, which writes actors in order.
        groups = np.repeat(np.arange(24), int(np.ceil(len(y) / 24)))[: len(y)]
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for actor in np.unique(groups):
        for label in np.unique(y[groups == actor]):
            index = np.flatnonzero((groups == actor) & (y == label))
            rng.shuffle(index)
            test_count = max(1, int(round(len(index) * 0.2)))
            test_parts.append(index[:test_count])
            train_parts.append(index[test_count:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    x_train, x_test = _standardize_from_train(x[train_idx], x[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]
    clients = _assign_groups_to_clients(
        x_train, y_train, groups[train_idx], num_clients, seed
    )
    return DatasetBundle(clients, x_test, y_test, tuple(x.shape[1:]), 8)


def load_dronerf(path: str, num_clients: int, seed: int) -> DatasetBundle:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"DroneRF file not found: {source}. Run prepare_dronerf.py on the raw CSV directory first."
        )
    data = np.load(source, allow_pickle=False)
    required = {"X", "Y", "groups"}
    if not required.issubset(data.files):
        raise ValueError(f"DroneRF npz must contain {sorted(required)}")
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["Y"], dtype=np.int64)
    groups = np.asarray(data["groups"])
    if x.ndim != 5 or x.shape[-1] != 2:
        raise ValueError("DroneRF X must have shape [N,H,W,1,2]")
    train_idx, test_idx = _group_train_test_indices(groups, y, seed)
    x_train, x_test = _standardize_from_train(x[train_idx], x[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]
    clients = _assign_groups_to_clients(
        x_train, y_train, groups[train_idx], num_clients, seed
    )
    return DatasetBundle(clients, x_test, y_test, tuple(x.shape[1:]), 4)


def _mnist_two_shards_per_client(x, y, num_clients: int, seed: int):
    """Classic shard Non-IID split: every client receives exactly two shards."""
    rng = np.random.default_rng(seed)
    sorted_index = np.argsort(y, kind="stable")
    shard_count = 2 * num_clients
    shards = [arr for arr in np.array_split(sorted_index, shard_count) if len(arr)]
    rng.shuffle(shards)
    clients = []
    for client_id in range(num_clients):
        index = np.concatenate(shards[2 * client_id : 2 * client_id + 2])
        rng.shuffle(index)
        clients.append((x[index], y[index]))
    return clients


def load_mnist(num_clients: int, seed: int) -> DatasetBundle:
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train[..., None].astype(np.float32) / 255.0)
    x_test = (x_test[..., None].astype(np.float32) / 255.0)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    clients = _mnist_two_shards_per_client(x_train, y_train, num_clients, seed)
    return DatasetBundle(clients, x_test, y_test, tuple(x_train.shape[1:]), 10)


def load_dataset(name: str, path: str, num_clients: int, seed: int) -> DatasetBundle:
    if name == "mnist":
        return load_mnist(num_clients, seed)
    if name == "ravdess":
        return load_ravdess(path, num_clients, seed)
    if name == "dronerf":
        return load_dronerf(path, num_clients, seed)
    raise ValueError(f"unknown dataset: {name}")
