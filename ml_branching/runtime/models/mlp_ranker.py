from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.0


class MLPRanker(nn.Module):
    """Shared candidate scorer.

    Input shape is [num_candidates, feature_dim], output shape is
    [num_candidates]. The module never assumes a fixed number of candidates.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        if config.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if config.activation != "relu":
            raise ValueError('only activation="relu" is currently supported')
        layers: list[nn.Module] = []
        dim = int(config.input_dim)
        for _ in range(int(config.num_layers)):
            layers.append(nn.Linear(dim, int(config.hidden_dim)))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(float(config.dropout)))
            dim = int(config.hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [num_candidates, feature_dim]")
        return self.network(features).squeeze(-1)

    def predict(self, features) -> torch.Tensor:
        tensor = torch.as_tensor(features, dtype=torch.float32)
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            return self.forward(tensor.to(device)).detach().cpu()

    def config_dict(self) -> dict:
        return asdict(self.config)

    @classmethod
    def from_config_dict(cls, data: dict) -> "MLPRanker":
        return cls(MLPConfig(**data))

    def save_checkpoint(self, path: str | Path, payload: dict) -> None:
        payload = dict(payload)
        payload["model_state_dict"] = self.state_dict()
        payload["model_config"] = self.config_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
        return torch.load(path, map_location=map_location, weights_only=False)
