import pytest

from examples.fixed_charge_block import build_problem
from solver import FirstFractionalPolicy, MostFractionalPolicy, solve_milp
from solver.branch_and_bound import choose_binary_branch_variable


def test_branching_rule_wrapper_keeps_historical_most_fractional_tie_break():
    x=[0.0,0.0,0.5,0.5]
    assert choose_binary_branch_variable(x,[2,3],rule="most_fractional")==3
    assert choose_binary_branch_variable(x,[2,3],rule="first_fractional")==2


def test_default_policy_matches_explicit_most_fractional():
    problem=build_problem()
    default=solve_milp(problem)
    explicit=solve_milp(problem,branching_policy=MostFractionalPolicy())

    assert default.status=="optimal"
    assert explicit.status=="optimal"
    assert explicit.objective_value==pytest.approx(default.objective_value)
    assert explicit.log==default.log


def test_first_fractional_policy_runs_through_adapter():
    result=solve_milp(build_problem(),branching_policy=FirstFractionalPolicy())

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(28.5)


def test_branching_policy_conflicts_with_non_default_rule():
    with pytest.raises(ValueError,match="branching_policy"):
        solve_milp(
            build_problem(),
            branching_rule="first_fractional",
            branching_policy=MostFractionalPolicy(),
        )
