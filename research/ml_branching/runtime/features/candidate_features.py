from __future__ import annotations

import numpy as np


FEATURE_SCHEMA_VERSION = "candidate_features_v2"

FEATURE_NAMES = [
    "lp_value",
    "fractional_distance",
    "objective_coefficient",
    "node_lb",
    "node_ub",
    "bound_width",
    "column_nnz",
    "column_l1_norm",
    "column_l2_norm",
    "column_max_abs",
    "column_positive_count",
    "column_negative_count",
    "column_positive_sum",
    "column_negative_abs_sum",
    "node_depth",
    "parent_lp_bound",
    "incumbent_exists",
    "normalized_gap",
    "num_fractional_candidates",
    "num_variables",
    "num_continuous_variables",
    "num_binary_variables",
    "num_constraints",
    "num_fixed_binary_variables",
    "units",
]


def _metadata_value(metadata: dict, *keys, default=None):
    for key in keys:
        if key in metadata:
            return metadata[key]
    return default


def _instance_parameters(metadata: dict) -> dict:
    return metadata.get("instance_parameters") or metadata.get("config") or {}


def candidate_feature_matrix(sample: dict) -> tuple[np.ndarray, list[str]]:
    arrays = sample["arrays"]
    metadata = sample["metadata"]
    G = np.asarray(arrays["G"], dtype=float)
    c = np.asarray(arrays["internal_objective_coefficients"], dtype=float)
    node_lb = np.asarray(arrays["node_lb"], dtype=float)
    node_ub = np.asarray(arrays["node_ub"], dtype=float)
    lp_solution = np.asarray(arrays["lp_solution"], dtype=float)
    candidates = np.asarray(arrays["candidate_indices"], dtype=int)
    binary_indices = np.asarray(arrays.get("binary_variable_indices", []), dtype=int)

    parent_bound = float(_metadata_value(metadata, "parent_internal_lp_bound", "parent_lp_bound"))
    incumbent = metadata["incumbent_internal_value"]
    incumbent_exists = 0.0 if incumbent is None else 1.0
    if incumbent is None:
        normalized_gap = 0.0
    else:
        normalized_gap = max(0.0, parent_bound - float(incumbent)) / max(1.0, abs(float(incumbent)))
    params = _instance_parameters(metadata)
    num_binary = len(binary_indices)
    num_continuous = len(c) - num_binary
    num_fixed_binary = int(np.count_nonzero(np.abs(node_ub[binary_indices] - node_lb[binary_indices]) <= 1e-10)) if len(binary_indices) else 0
    units = float(params.get("units", num_binary))

    features = []
    for index in candidates:
        column = G[:, index]
        lp_value = float(lp_solution[index])
        row = [
            lp_value,
            abs(lp_value - round(lp_value)),
            float(c[index]),
            float(node_lb[index]),
            float(node_ub[index]),
            float(node_ub[index] - node_lb[index]),
            float(np.count_nonzero(np.abs(column) > 1e-12)),
            float(np.sum(np.abs(column))),
            float(np.linalg.norm(column)),
            float(np.max(np.abs(column))) if len(column) else 0.0,
            float(np.count_nonzero(column > 1e-12)),
            float(np.count_nonzero(column < -1e-12)),
            float(np.sum(column[column > 1e-12])) if len(column) else 0.0,
            float(np.sum(np.abs(column[column < -1e-12]))) if len(column) else 0.0,
            float(metadata["node_depth"]),
            parent_bound,
            incumbent_exists,
            normalized_gap,
            float(len(candidates)),
            float(len(c)),
            float(num_continuous),
            float(num_binary),
            float(G.shape[0]),
            float(num_fixed_binary),
            units,
        ]
        features.append(row)
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("candidate feature matrix has an unexpected shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("candidate feature matrix contains nan or inf")
    return matrix, FEATURE_NAMES.copy()
