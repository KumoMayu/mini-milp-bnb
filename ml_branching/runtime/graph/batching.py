from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .bipartite_state import BipartiteGraphState


@dataclass(frozen=True)
class GraphBatch:
    variable_features: torch.Tensor
    constraint_features: torch.Tensor
    edge_indices: torch.Tensor
    edge_features: torch.Tensor
    global_features: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_graph_ids: torch.Tensor
    graph_variable_offsets: torch.Tensor
    num_graphs: int

    def to(self, device: str | torch.device) -> "GraphBatch":
        return GraphBatch(
            variable_features=self.variable_features.to(device),
            constraint_features=self.constraint_features.to(device),
            edge_indices=self.edge_indices.to(device),
            edge_features=self.edge_features.to(device),
            global_features=self.global_features.to(device),
            candidate_indices=self.candidate_indices.to(device),
            candidate_graph_ids=self.candidate_graph_ids.to(device),
            graph_variable_offsets=self.graph_variable_offsets.to(device),
            num_graphs=self.num_graphs,
        )


def batch_graph_states(states: list[BipartiteGraphState], dtype=torch.float32) -> GraphBatch:
    if not states:
        raise ValueError("cannot batch an empty list of graph states")
    variable_parts = []
    constraint_parts = []
    edge_parts = []
    edge_feature_parts = []
    global_parts = []
    candidate_parts = []
    candidate_graph_ids = []
    variable_offsets = []
    var_offset = 0
    con_offset = 0
    for graph_id, state in enumerate(states):
        n_var = state.variable_features.shape[0]
        n_con = state.constraint_features.shape[0]
        variable_offsets.append(var_offset)
        variable_parts.append(state.variable_features)
        constraint_parts.append(state.constraint_features)
        shifted_edges = state.edge_indices.copy()
        shifted_edges[0, :] += var_offset
        shifted_edges[1, :] += con_offset
        edge_parts.append(shifted_edges)
        edge_feature_parts.append(state.edge_features)
        global_parts.append(state.global_features)
        candidate_parts.append(state.candidate_positions + var_offset)
        candidate_graph_ids.extend([graph_id] * len(state.candidate_positions))
        var_offset += n_var
        con_offset += n_con

    return GraphBatch(
        variable_features=torch.as_tensor(np.vstack(variable_parts), dtype=dtype),
        constraint_features=torch.as_tensor(np.vstack(constraint_parts), dtype=dtype),
        edge_indices=torch.as_tensor(np.hstack(edge_parts), dtype=torch.long),
        edge_features=torch.as_tensor(np.vstack(edge_feature_parts), dtype=dtype),
        global_features=torch.as_tensor(np.vstack(global_parts), dtype=dtype),
        candidate_indices=torch.as_tensor(np.concatenate(candidate_parts), dtype=torch.long),
        candidate_graph_ids=torch.as_tensor(candidate_graph_ids, dtype=torch.long),
        graph_variable_offsets=torch.as_tensor(variable_offsets, dtype=torch.long),
        num_graphs=len(states),
    )


def single_graph_batch(state: BipartiteGraphState, dtype=torch.float32) -> GraphBatch:
    return batch_graph_states([state], dtype=dtype)


__all__ = ["GraphBatch", "batch_graph_states", "single_graph_batch"]
