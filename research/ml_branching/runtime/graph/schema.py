from __future__ import annotations


GRAPH_SCHEMA_VERSION = "bipartite_graph_state_v1"

VARIABLE_FEATURE_NAMES = [
    "objective_coeff_scaled",
    "lp_value_scaled",
    "node_lb_scaled",
    "node_ub_scaled",
    "bound_width_scaled",
    "is_binary",
    "is_fractional_candidate",
    "fractional_distance",
    "is_fixed",
    "distance_to_lb",
    "distance_to_ub",
]

CONSTRAINT_FEATURE_NAMES = [
    "rhs_scaled",
    "activity_scaled",
    "slack_scaled",
    "row_l1_scaled",
    "row_l2_scaled",
    "nnz_scaled",
    "near_active",
]

EDGE_FEATURE_NAMES = ["coefficient_scaled"]

GLOBAL_FEATURE_NAMES = [
    "node_depth_scaled",
    "incumbent_exists",
    "normalized_gap",
    "candidate_count_scaled",
    "fixed_binary_ratio",
]


__all__ = [
    "CONSTRAINT_FEATURE_NAMES",
    "EDGE_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "GRAPH_SCHEMA_VERSION",
    "VARIABLE_FEATURE_NAMES",
]
