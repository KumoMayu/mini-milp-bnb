from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: str
    output_dir: str = "research/ml_branching/trained_models/learning_branching"
    require_audit_pass: bool = True
    required_dataset_id: str | None = None
    min_candidate_count: int = 2
    balance_train_by_family: bool = True
    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    max_epochs: int = 30
    early_stopping_patience: int = 8
    min_delta: float = 1e-6
    gradient_clip: float = 1.0
    device: str = "cpu"
    hyperparameter_grid: dict | None = None


def load_training_config(path: str | Path) -> TrainingConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig(**data)
