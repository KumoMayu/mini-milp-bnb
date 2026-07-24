from __future__ import annotations

import numpy as np
import pytest

from solver import solve_milp
from solver.branching import BranchingContext, PseudocostPolicy


class _LP:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)


def _context(x=(0.0, 0.25, 0.75), candidates=(1, 2), bound=10.0):
    return BranchingContext(
        problem=None,
        node_id=0,
        node_depth=0,
        node_lb=np.zeros(3),
        node_ub=np.ones(3),
        lp_result=_LP(x),
        candidate_indices=tuple(candidates),
        incumbent_internal_value=None,
        current_node_internal_bound=float(bound),
        tolerance=1e-8,
    )


def test_pseudocost_updates_bound_improvement_direction_and_sign():
    policy = PseudocostPolicy()
    context = _context(x=(0.0, 0.25, 0.75), candidates=(1,), bound=10.0)
    policy.observe_branch_result(context, branch_var=1, branch_value=0, child_status="optimal", child_bound=8.0)
    policy.observe_branch_result(context, branch_var=1, branch_value=1, child_status="optimal", child_bound=9.0)
    stats = policy.history_snapshot()[1]
    assert stats["toward_zero_count"] == 1
    assert stats["toward_one_count"] == 1
    assert stats["toward_zero_average"] == pytest.approx((10.0 - 8.0) / 0.25)
    assert stats["toward_one_average"] == pytest.approx((10.0 - 9.0) / 0.75)


def test_pseudocost_ignores_non_improving_and_nonoptimal_children():
    policy = PseudocostPolicy()
    context = _context(x=(0.0, 0.5), candidates=(1,), bound=10.0)
    policy.observe_branch_result(context, 1, 0, "optimal", 11.0)
    policy.observe_branch_result(context, 1, 1, "infeasible", None)
    stats = policy.history_snapshot()[1]
    assert stats["toward_zero_count"] == 1
    assert stats["toward_zero_average"] == pytest.approx(0.0)
    assert stats["toward_one_count"] == 0


def test_pseudocost_falls_back_deterministically_without_history():
    policy = PseudocostPolicy()
    context = _context(x=(0.0, 0.25, 0.75), candidates=(1, 2), bound=10.0)
    assert policy.select_variable(context) == 2
    assert policy.select_variable(context) == 2


def test_pseudocost_uses_history_after_observations():
    policy = PseudocostPolicy()
    context = _context(x=(0.0, 0.25, 0.75), candidates=(1, 2), bound=10.0)
    policy.observe_branch_result(context, 1, 0, "optimal", 7.0)
    policy.observe_branch_result(context, 1, 1, "optimal", 7.0)
    policy.observe_branch_result(context, 2, 0, "optimal", 9.9)
    policy.observe_branch_result(context, 2, 1, "optimal", 9.9)
    assert policy.select_variable(context) == 1


def test_pseudocost_completed_objective_matches_most_fractional():
    pytest.importorskip("scipy")
    from examples.branch_and_bound_demo import build_problem

    problem = build_problem()
    baseline = solve_milp(problem, lp_backend="scipy_highs", max_nodes=200)
    pseudo = solve_milp(problem, lp_backend="scipy_highs", branching_policy=PseudocostPolicy(), max_nodes=200)
    assert baseline.status == pseudo.status == "optimal"
    assert pseudo.objective_value == pytest.approx(baseline.objective_value)
