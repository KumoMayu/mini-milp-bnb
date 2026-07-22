from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "branching_sample_v1"

REQUIRED_ARRAY_FIELDS = {
    "internal_objective_coefficients",
    "G",
    "h",
    "node_lb",
    "node_ub",
    "lp_solution",
    "binary_variable_indices",
    "candidate_indices",
    "candidate_lp_values",
    "child_0_bound",
    "child_1_bound",
    "child_0_runtime_sec",
    "child_1_runtime_sec",
    "delta_0",
    "delta_1",
    "expert_scores",
}

REQUIRED_META_FIELDS = {
    "schema_version",
    "dataset_id",
    "split",
    "family_name",
    "scale_group",
    "instance_id",
    "instance_seed",
    "instance_parameters",
    "node_id",
    "node_depth",
    "objective_sense",
    "parent_internal_lp_bound",
    "incumbent_internal_value",
    "child_0_status",
    "child_1_status",
    "expert_selected_variable",
    "scoring_mode",
    "epsilon",
    "tolerance",
    "lp_backend",
    "solver_config",
    "generated_at",
}


def save_sample(path: str | Path, arrays: dict, metadata: dict) -> None:
    metadata = dict(metadata)
    metadata["schema_version"] = SCHEMA_VERSION
    payload = {key: np.asarray(value, dtype=float) if key not in {"binary_variable_indices", "candidate_indices"} else np.asarray(value, dtype=int) for key, value in arrays.items()}
    payload["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


def validate_sample(sample: dict) -> None:
    arrays = sample["arrays"]
    metadata = sample["metadata"]
    missing_arrays = sorted(REQUIRED_ARRAY_FIELDS - set(arrays))
    missing_meta = sorted(REQUIRED_META_FIELDS - set(metadata))
    if missing_arrays or missing_meta:
        raise ValueError(f"branching sample missing arrays={missing_arrays} metadata={missing_meta}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version={metadata['schema_version']!r}")
    n_candidates = len(arrays["candidate_indices"])
    if n_candidates <= 0:
        raise ValueError("candidate_indices must be non-empty")
    for field in (
        "candidate_lp_values",
        "child_0_bound",
        "child_1_bound",
        "child_0_runtime_sec",
        "child_1_runtime_sec",
        "delta_0",
        "delta_1",
        "expert_scores",
    ):
        if len(arrays[field]) != n_candidates:
            raise ValueError(f"{field} length does not match candidate_indices")
    for field in ("child_0_status", "child_1_status"):
        if len(metadata[field]) != n_candidates:
            raise ValueError(f"{field} length does not match candidate_indices")
    n_vars = len(arrays["internal_objective_coefficients"])
    if arrays["G"].ndim != 2:
        raise ValueError("G must be a two-dimensional matrix")
    if arrays["G"].shape[1] != n_vars:
        raise ValueError("G.shape[1] must match internal objective coefficient length")
    for field in ("node_lb", "node_ub", "lp_solution"):
        if len(arrays[field]) != n_vars:
            raise ValueError(f"{field} length must match number of variables")
    if int(metadata["expert_selected_variable"]) not in set(map(int, arrays["candidate_indices"])):
        raise ValueError("expert_selected_variable must belong to candidate_indices")
