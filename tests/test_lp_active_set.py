import numpy as np
import pytest

from solver import MILPProblem
from solver.lp_active_set import solve_lp_relaxation


def test_lp_relaxation_max_objective():
    problem=MILPProblem.from_standard(
        c=[1,1],
        G=[
            [1,1],
            [1,0],
            [0,1],
        ],
        h=[4,3,2],
        sense="max",
        lb=[0,0],
        ub=[10,10],
        var_types=["C","C"],
    )

    result=solve_lp_relaxation(problem,problem.lb,problem.ub)

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(4.0)
    assert result.x is not None
    assert np.all(problem.G@result.x<=problem.h+1e-8)


def test_lp_relaxation_infeasible():
    problem=MILPProblem.from_standard(
        c=[1],
        G=[
            [1],
            [-1],
        ],
        h=[0,-1],
        sense="max",
        lb=[0],
        ub=[1],
        var_types=["C"],
    )

    result=solve_lp_relaxation(problem,problem.lb,problem.ub)

    assert result.status=="infeasible"
    assert result.x is None


def test_node_bounds_enter_relaxation():
    problem=MILPProblem.from_standard(
        c=[1],
        G=np.zeros((0,1)),
        h=[],
        sense="max",
        lb=[0],
        ub=[10],
        var_types=["C"],
    )

    result=solve_lp_relaxation(problem,np.array([0.0]),np.array([3.0]))

    assert result.status=="optimal"
    assert result.x[0]==pytest.approx(3.0)
    assert result.objective_value==pytest.approx(3.0)


def test_min_problem_internal_conversion_for_lp():
    problem=MILPProblem.from_standard(
        c=[1],
        G=[[-1]],
        h=[-2],
        sense="min",
        lb=[0],
        ub=[5],
        var_types=["C"],
    )

    result=solve_lp_relaxation(problem,problem.lb,problem.ub)

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(-2.0)
    assert problem.recover_objective_value(result.objective_value)==pytest.approx(2.0)
