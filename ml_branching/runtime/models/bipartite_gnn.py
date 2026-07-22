from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class BipartiteGNNConfig:
    variable_dim: int
    constraint_dim: int
    edge_dim: int
    global_dim: int
    hidden_dim: int = 64
    message_rounds: int = 2
    dropout: float = 0.0


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.Linear(hidden_dim, output_dim), nn.ReLU()])
    return nn.Sequential(*layers)


class BipartiteGNN(nn.Module):
    def __init__(self, config: BipartiteGNNConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_dim)
        self.variable_encoder = _mlp(config.variable_dim, h, h, config.dropout)
        self.constraint_encoder = _mlp(config.constraint_dim, h, h, config.dropout)
        self.edge_encoder = nn.Sequential(nn.Linear(config.edge_dim, h), nn.ReLU())
        self.var_to_con = _mlp(2 * h, h, h, config.dropout)
        self.con_to_var = _mlp(2 * h, h, h, config.dropout)
        self.constraint_update = _mlp(2 * h, h, h, config.dropout)
        self.variable_update = _mlp(2 * h, h, h, config.dropout)
        self.scorer = nn.Sequential(
            nn.Linear(h + config.global_dim, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        var_h = self.variable_encoder(batch.variable_features)
        con_h = self.constraint_encoder(batch.constraint_features)
        edge_h = self.edge_encoder(batch.edge_features)
        var_idx = batch.edge_indices[0]
        con_idx = batch.edge_indices[1]

        for _ in range(int(self.config.message_rounds)):
            c_messages = self.var_to_con(torch.cat([var_h[var_idx], edge_h], dim=1))
            c_agg = torch.zeros_like(con_h)
            c_count = torch.zeros((con_h.shape[0], 1), dtype=con_h.dtype, device=con_h.device)
            c_agg.index_add_(0, con_idx, c_messages)
            c_count.index_add_(0, con_idx, torch.ones((len(con_idx), 1), dtype=con_h.dtype, device=con_h.device))
            c_agg = c_agg / c_count.clamp_min(1.0)
            con_h = self.constraint_update(torch.cat([con_h, c_agg], dim=1))

            v_messages = self.con_to_var(torch.cat([con_h[con_idx], edge_h], dim=1))
            v_agg = torch.zeros_like(var_h)
            v_count = torch.zeros((var_h.shape[0], 1), dtype=var_h.dtype, device=var_h.device)
            v_agg.index_add_(0, var_idx, v_messages)
            v_count.index_add_(0, var_idx, torch.ones((len(var_idx), 1), dtype=var_h.dtype, device=var_h.device))
            v_agg = v_agg / v_count.clamp_min(1.0)
            var_h = self.variable_update(torch.cat([var_h, v_agg], dim=1))

        candidate_h = var_h[batch.candidate_indices]
        candidate_global = batch.global_features[batch.candidate_graph_ids]
        return self.scorer(torch.cat([candidate_h, candidate_global], dim=1)).squeeze(-1)

    def config_dict(self) -> dict:
        return asdict(self.config)

    @classmethod
    def from_config_dict(cls, data: dict) -> "BipartiteGNN":
        return cls(BipartiteGNNConfig(**data))

    def save_checkpoint(self, path: str | Path, payload: dict) -> None:
        payload = dict(payload)
        payload["model_state_dict"] = self.state_dict()
        payload["model_config"] = self.config_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
        return torch.load(path, map_location=map_location, weights_only=False)


__all__ = ["BipartiteGNN", "BipartiteGNNConfig"]
