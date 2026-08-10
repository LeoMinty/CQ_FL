"""Single source of truth for the experiment configuration in the draft."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


METHOD_NAMES: Tuple[str, ...] = (
    "fedavg_fp32",
    "signsgd",
    "w2_fp32_adam",
    "cqfl",
)


@dataclass(frozen=True)
class DatasetConfig:
    clients: int
    rounds: int
    local_epochs: int
    classes: int
    default_path: str = ""


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "ravdess": DatasetConfig(
        clients=2,
        rounds=50,
        local_epochs=2,
        classes=8,
        default_path="ravdess_processed/ravdess_c3_stft.npz",
    ),
    "dronerf": DatasetConfig(
        clients=5,
        rounds=100,
        local_epochs=2,
        classes=4,
        default_path="dronerf_processed/dronerf_drf2_complex.npz",
    ),
    "mnist": DatasetConfig(
        clients=10,
        rounds=50,
        local_epochs=2,
        classes=10,
    ),
}


@dataclass
class ExperimentConfig:
    dataset: str
    method: str
    data_path: str = ""
    output_root: str = "results/experiment1"
    batch_size: int = 32
    learning_rate: float = 3e-4
    block_size: int = 64
    seed: int = 42
    clients: int = 0
    rounds: int = 0
    local_epochs: int = 0
    max_train_samples: int = 0
    max_test_samples: int = 0

    def resolved(self) -> "ExperimentConfig":
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(f"unknown dataset: {self.dataset}")
        if self.method not in METHOD_NAMES:
            raise ValueError(f"unknown method: {self.method}")
        ds = DATASET_CONFIGS[self.dataset]
        self.clients = self.clients or ds.clients
        self.rounds = self.rounds or ds.rounds
        self.local_epochs = self.local_epochs or ds.local_epochs
        self.data_path = self.data_path or ds.default_path
        if self.data_path:
            self.data_path = str(Path(self.data_path))
        return self
