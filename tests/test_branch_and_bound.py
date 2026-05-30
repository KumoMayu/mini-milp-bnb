import pytest

from examples.fixed_charge_block import build_problem as build_fixed_charge
from examples.general_integer_block import build_problem as build_general_integer
from solver import MILPProblem, solve_milp


def test_fixed_charge_block_objective():
    result=solve_milp(build_fixed_charge())

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(28.5)
    assert result.x_continuous is not None
    assert result.y_integer is not None


def test_general_integer_block_objective():
    result=solve_milp(build_general_integer())

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(19.0)
    assert result.num_pruned_bound>=1


def test_infeasible_case_returns_infeasible():
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
        var_types=["I"],
    )

    result=solve_milp(problem)

    assert result.status=="infeasible"
    assert result.objective_value is None
    assert result.num_pruned_infeasible>=1


def test_integral_lp_solution_prunes_by_optimality():
    problem=MILPProblem.from_standard(
        c=[1],
        G=[[1]],
        h=[1],
        sense="max",
        lb=[0],
        ub=[1],
        var_types=["B"],
    )

    result=solve_milp(problem)

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(1.0)
    assert result.num_pruned_optimality>=1
