import pytest

from examples.fixed_charge_block import build_problem
from solver import solve_milp


def test_scipy_highs_backend_matches_active_set_when_available():
    pytest.importorskip("scipy")

    problem=build_problem()
    active=solve_milp(problem,lp_backend="active_set")
    highs=solve_milp(problem,lp_backend="scipy_highs")

    assert active.status=="optimal"
    assert highs.status=="optimal"
    assert highs.objective_value==pytest.approx(active.objective_value)
