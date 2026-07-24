from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import (
    CONSTRAINT_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    VARIABLE_FEATURE_NAMES,
)


@dataclass(frozen=True)
class BipartiteGraphState:
    variable_features: np.ndarray
    constraint_features: np.ndarray
    edge_indices: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray
    candidate_indices: np.ndarray
    candidate_positions: np.ndarray
    metadata: dict


def _safe_scale(values: np.ndarray, floor: float = 1.0) -> float:
    if values.size == 0:
        return floor
    return max(floor, float(np.max(np.abs(values))))


def _metadata_value(metadata: dict, *keys, default=None):
    for key in keys:
        if key in metadata:
            return metadata[key]
    return default


def sample_to_bipartite_state(sample: dict, tolerance: float = 1e-8) -> BipartiteGraphState:
    arrays = sample["arrays"]
    metadata = sample["metadata"]
    c = np.asarray(arrays["internal_objective_coefficients"], dtype=float)
    G = np.asarray(arrays["G"], dtype=float)
    h = np.asarray(arrays["h"], dtype=float)
    node_lb = np.asarray(arrays["node_lb"], dtype=float)
    node_ub = np.asarray(arrays["node_ub"], dtype=float)
    lp_solution = np.asarray(arrays["lp_solution"], dtype=float)
    binary_indices = np.asarray(arrays.get("binary_variable_indices", []), dtype=int)
    candidate_indices = np.asarray(arrays["candidate_indices"], dtype=int)

    n_vars = len(c)
    n_rows = len(h)
    binary_mask = np.zeros(n_vars, dtype=float)
    binary_mask[binary_indices] = 1.0
    candidate_mask = np.zeros(n_vars, dtype=float)
    candidate_mask[candidate_indices] = 1.0
    width = np.maximum(node_ub - node_lb, 0.0)
    finite_width = np.maximum(width, 1e-9)
    c_scale = _safe_scale(c)
    bound_scale = _safe_scale(np.concatenate([node_lb, node_ub, lp_solution]))

    fractional_distance = np.abs(lp_solution - np.round(lp_solution))
    is_fixed = (width <= tolerance).astype(float)
    distance_to_lb = np.clip((lp_solution - node_lb) / finite_width, 0.0, 1.0)
    distance_to_ub = np.clip((node_ub - lp_solution) / finite_width, 0.0, 1.0)
    variable_features = np.column_stack(
        [
            c / c_scale,
            lp_solution / bound_scale,
            node_lb / bound_scale,
            node_ub / bound_scale,
            width / max(1.0, float(np.max(width)) if len(width) else 1.0),
            binary_mask,
            candidate_mask,
            fractional_distance,
            is_fixed,
            distance_to_lb,
            distance_to_ub,
        ]
    ).astype(float)

    activity = G @ lp_solution
    slack = h - activity
    row_l1 = np.sum(np.abs(G), axis=1)
    row_l2 = np.linalg.norm(G, axis=1)
    row_nnz = np.count_nonzero(np.abs(G) > 1e-12, axis=1)
    row_scale = np.maximum.reduce([np.ones(n_rows), np.abs(h), np.abs(activity), row_l1])
    near_active = (np.abs(slack) <= 1e-6 * np.maximum(1.0, np.abs(h))).astype(float)
    constraint_features = np.column_stack(
        [
            h / row_scale,
            activity / row_scale,
            slack / row_scale,
            row_l1 / max(1.0, float(np.max(row_l1)) if len(row_l1) else 1.0),
            row_l2 / max(1.0, float(np.max(row_l2)) if len(row_l2) else 1.0),
            row_nnz / max(1.0, float(n_vars)),
            near_active,
        ]
    ).astype(float)

    row_idx, col_idx = np.nonzero(np.abs(G) > 1e-12)
    edge_indices = np.vstack([col_idx.astype(int), row_idx.astype(int)])
    coeff = G[row_idx, col_idx]
    coeff_scale = np.maximum(1.0, row_l2[row_idx])
    edge_features = (coeff / coeff_scale).reshape(-1, 1).astype(float)

    parent_bound = float(_metadata_value(metadata, "parent_internal_lp_bound", "parent_lp_bound", default=0.0))
    incumbent = metadata.get("incumbent_internal_value")
    incumbent_exists = 0.0 if incumbent is None else 1.0
    normalized_gap = 0.0 if incumbent is None else max(0.0, parent_bound - float(incumbent)) / max(1.0, abs(float(incumbent)))
    fixed_binary_count = int(np.count_nonzero(is_fixed[binary_indices])) if len(binary_indices) else 0
    global_features = np.asarray(
        [
            float(metadata.get("node_depth", 0)) / max(1.0, float(len(binary_indices))),
            incumbent_exists,
            normalized_gap,
            float(len(candidate_indices)) / max(1.0, float(len(binary_indices))),
            float(fixed_binary_count) / max(1.0, float(len(binary_indices))),
        ],
        dtype=float,
    )
    candidate_positions = candidate_indices.astype(int)
    if not np.all(np.isfinite(variable_features)):
        raise ValueError("variable graph features contain nan or inf")
    if not np.all(np.isfinite(constraint_features)):
        raise ValueError("constraint graph features contain nan or inf")
    if not np.all(np.isfinite(edge_features)):
        raise ValueError("edge graph features contain nan or inf")
    return BipartiteGraphState(
        variable_features=variable_features,
        constraint_features=constraint_features,
        edge_indices=edge_indices,
        edge_features=edge_features,
        global_features=global_features,
        candidate_indices=candidate_indices.astype(int),
        candidate_positions=candidate_positions,
        metadata={
            "schema_version": GRAPH_SCHEMA_VERSION,
            "variable_feature_names": VARIABLE_FEATURE_NAMES,
            "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "global_feature_names": GLOBAL_FEATURE_NAMES,
            "source_metadata": metadata,
        },
    )


def runtime_context_to_sample(context) -> dict:
    if context.problem is None:
        raise ValueError("graph branching requires a MILPProblem in BranchingContext")
    return {
        "arrays": {
            "internal_objective_coefficients": context.problem.internal_c,
            "G": context.problem.G,
            "h": context.problem.h,
            "node_lb": context.node_lb,
            "node_ub": context.node_ub,
            "lp_solution": context.lp_result.x,
            "binary_variable_indices": np.asarray(context.problem.binary_indices, dtype=int),
            "candidate_indices": np.asarray(context.candidate_indices, dtype=int),
        },
        "metadata": {
            "node_depth": context.node_depth,
            "parent_internal_lp_bound": float(context.current_node_internal_bound),
            "incumbent_internal_value": context.incumbent_internal_value,
        },
    }


__all__ = [
    "BipartiteGraphState",
    "runtime_context_to_sample",
    "sample_to_bipartite_state",
]
