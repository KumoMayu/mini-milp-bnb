from __future__ import annotations


DEFAULT_EPSILON = 1e-6
DEFAULT_WEIGHT_MU = 0.25


def infeasible_improvement_from_parent(parent_bound: float) -> float:
    """Finite deterministic improvement used when a probe child is infeasible."""
    return max(1.0, abs(float(parent_bound)))


def child_delta(
    parent_bound: float,
    child_bound: float | None,
    child_status: str,
    infeasible_improvement: float | None = None,
) -> float:
    """Return internal-bound improvement for one strong-branching child.

    The solver internally maximizes.  A child with a smaller LP upper bound is
    an improvement for pruning.  Infeasible children receive a configurable
    finite improvement so scoring remains deterministic and testable.
    """
    parent = float(parent_bound)
    if child_status == "infeasible":
        if infeasible_improvement is None:
            infeasible_improvement = infeasible_improvement_from_parent(parent)
        return float(infeasible_improvement)
    if child_status != "optimal" or child_bound is None:
        return 0.0
    return max(0.0, parent - float(child_bound))


def score_from_deltas(
    delta_0: float,
    delta_1: float,
    mode: str = "product",
    epsilon: float = DEFAULT_EPSILON,
    mu: float = DEFAULT_WEIGHT_MU,
    weight_min: float = 0.8,
    weight_max: float = 0.2,
) -> float:
    """Score a candidate from its two child improvements.

    `product` is the default, but the raw child bounds and deltas are stored
    so future experiments can rescore old data without recollecting LP probes.
    """
    d0 = max(0.0, float(delta_0))
    d1 = max(0.0, float(delta_1))
    if mode == "product":
        return max(d0, epsilon) * max(d1, epsilon)
    if mode == "weighted":
        return (1.0 - float(mu)) * min(d0, d1) + float(mu) * max(d0, d1)
    if mode == "weighted_product":
        return (max(d0, epsilon) ** weight_min) * (max(d1, epsilon) ** weight_max)
    if mode == "minmax":
        return min(d0, d1) + 0.1 * max(d0, d1)
    raise ValueError('score_mode must be "product", "weighted", "weighted_product", or "minmax"')
