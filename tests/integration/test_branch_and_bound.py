import pytest

from examples.branch_and_bound_demo import build_problem as build_fixed_charge
from examples.unit_commitment_tiny import build_problem as build_unit_commitment
from solver import MILPProblem, solve_milp
from tests.integration.problem_builders import build_production_expansion


def test_fixed_charge_block_objective():
    result=solve_milp(build_fixed_charge())

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(28.5)
    assert result.x_continuous is not None
    assert result.y_integer is not None
    assert result.initial_incumbent_found


def test_unit_commitment_tiny_objective():
    result=solve_milp(build_unit_commitment())

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(26.0)
    assert result.initial_incumbent_found


def test_production_expansion_binary_objective():
    result=solve_milp(build_production_expansion())

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(31.5)
    assert result.num_pruned_bound>=1


def test_matrix_presolve_toggle_keeps_same_objective():
    problem=build_production_expansion()

    with_presolve=solve_milp(problem,use_matrix_presolve=True)
    without_presolve=solve_milp(problem,use_matrix_presolve=False)

    assert with_presolve.status=="optimal"
    assert without_presolve.status=="optimal"
    assert with_presolve.objective_value==pytest.approx(without_presolve.objective_value)
    assert with_presolve.num_removed_rows>0


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
        var_types=["B"],
    )

    result=solve_milp(problem)

    assert result.status=="infeasible"
    assert result.objective_value is None
    assert result.num_pruned_infeasible>=1


def test_branching_only_uses_binary_y_variables():
    problem=build_fixed_charge()
    result=solve_milp(problem)

    branch_logs=[line for line in result.log if "no_pruning_branch" in line]
    assert branch_logs
    for line in branch_logs:
        assert "branch_var_group=y" in line
        index=int(line.split("branch_var=")[1].split()[0])
        assert index in problem.binary_indices


def test_statistics_fields_are_recorded():
    result=solve_milp(build_production_expansion())

    assert result.num_lp_candidates_checked>0
    assert result.num_fixed_vars_eliminated>0
    assert result.num_removed_rows>0
    assert result.num_free_vars_total>0
    assert result.num_heuristic_lp_solved>=1
    assert result.global_bound==pytest.approx(result.objective_value)
    assert result.relative_gap==pytest.approx(0.0)
