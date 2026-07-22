import numpy as np
import pytest

from examples.fixed_charge_block import build_problem
from ml_branching.training.oracle.strong_branching import StrongBranchingPolicy, make_strong_branch_child_bounds, strong_branch_candidates
from solver import solve_milp
from solver.branch_and_bound import fractional_binary_candidates
from solver.branching import BranchingContext
from solver.lp_backends import get_lp_relaxation_solver


def _root_context():
    pytest.importorskip("scipy")
    problem=build_problem()
    lp_solver=get_lp_relaxation_solver("scipy_highs")
    lp=lp_solver(problem,problem.lb,problem.ub,1e-8,None,True)
    candidates=fractional_binary_candidates(lp.x,problem.binary_indices,1e-8)
    return BranchingContext(
        problem=problem,
        node_id=0,
        node_depth=0,
        node_lb=problem.lb.copy(),
        node_ub=problem.ub.copy(),
        lp_result=lp,
        candidate_indices=candidates,
        incumbent_internal_value=None,
        current_node_internal_bound=float(lp.objective_value),
        tolerance=1e-8,
    )


def test_probe_child_bounds_are_one_sided_and_original_bounds_unchanged():
    context=_root_context()
    candidate=context.candidate_indices[0]
    original_lb=context.node_lb.copy()
    original_ub=context.node_ub.copy()

    child0_lb,child0_ub,status0=make_strong_branch_child_bounds(context.node_lb,context.node_ub,candidate,0)
    child1_lb,child1_ub,status1=make_strong_branch_child_bounds(context.node_lb,context.node_ub,candidate,1)

    assert status0=="open"
    assert status1=="open"
    assert child0_lb[candidate]==pytest.approx(original_lb[candidate])
    assert child0_ub[candidate]==pytest.approx(0.0)
    assert child1_lb[candidate]==pytest.approx(1.0)
    assert child1_ub[candidate]==pytest.approx(original_ub[candidate])
    assert np.allclose(context.node_lb,original_lb)
    assert np.allclose(context.node_ub,original_ub)


def test_strong_branching_records_all_candidates_and_tie_break_is_smallest_index(monkeypatch):
    context=_root_context()
    records=strong_branch_candidates(context,lp_backend="scipy_highs")
    assert {record.candidate_index for record in records}==set(context.candidate_indices)
    assert all(record.candidate_index in context.candidate_indices for record in records)
    assert all(record.child_0_status for record in records)
    assert all(record.child_1_status for record in records)

    policy=StrongBranchingPolicy(lp_backend="scipy_highs")
    monkeypatch.setattr(
        "ml_branching.training.oracle.strong_branching.strong_branch_candidates",
        lambda *args,**kwargs: [
            records[0].__class__(**{**records[0].__dict__,"candidate_index": 7,"score": 1.0}),
            records[0].__class__(**{**records[0].__dict__,"candidate_index": 5,"score": 1.0}),
        ],
    )
    assert policy.select_variable(context)==5


def test_strong_branching_policy_solves_same_objective_as_default():
    pytest.importorskip("scipy")
    problem=build_problem()
    default=solve_milp(problem,lp_backend="scipy_highs")
    policy=StrongBranchingPolicy(lp_backend="scipy_highs")
    strong=solve_milp(problem,branching_policy=policy,lp_backend="scipy_highs")

    assert default.status=="optimal"
    assert strong.status=="optimal"
    assert strong.objective_value==pytest.approx(default.objective_value)
    assert policy.probe_lp_solved>0
