from __future__ import annotations

import inspect

import numpy as np
import pytest

from examples.fixed_charge_block import build_problem as build_fixed_charge
from examples.production_expansion_binary import build_problem as build_production_expansion
from solver import BranchAndBoundSolver, LPResult, MILPProblem, solve_milp
from solver.lp_backends import get_lp_relaxation_solver


def build_switch_problem() -> MILPProblem:
    return MILPProblem.from_standard(
        c=[1.0, -0.5],
        G=[[1.0, -2.0]],
        h=[0.0],
        sense="max",
        lb=[0.0, 0.0],
        ub=[2.0, 1.0],
        var_types=["C", "B"],
        name="switch_max",
    )


def solve_node(problem, lb, ub, *, presolve=False, max_iterations=1000):
    backend = get_lp_relaxation_solver("two_phase_simplex")
    return backend(
        problem,
        np.asarray(lb, dtype=float),
        np.asarray(ub, dtype=float),
        1e-9,
        None,
        presolve,
        None,
        max_iterations,
    )


def test_backend_registry_creates_two_phase_backend_and_alias():
    primary = get_lp_relaxation_solver("two_phase_simplex")
    alias = get_lp_relaxation_solver("custom_two_phase")

    assert primary is alias
    assert callable(primary)


def test_root_node_lp_is_optimal_and_returns_original_variables():
    problem = build_switch_problem()
    result = solve_node(problem, problem.lb, problem.ub)

    assert result.status == "optimal"
    assert result.backend == "two_phase_simplex"
    assert result.objective_value == pytest.approx(1.5)
    assert np.allclose(result.x, [2.0, 1.0])
    assert result.objective_value == pytest.approx(problem.internal_c @ result.x)
    assert result.basis_indices


def test_node_fixing_binary_to_zero_is_respected():
    problem = build_switch_problem()
    result = solve_node(problem, [0.0, 0.0], [2.0, 0.0])

    assert result.status == "optimal"
    assert np.allclose(result.x, [0.0, 0.0])
    assert result.objective_value == pytest.approx(0.0)
    assert result.num_fixed_vars == 1


def test_node_fixing_binary_to_one_is_respected():
    problem = build_switch_problem()
    result = solve_node(problem, [0.0, 1.0], [2.0, 1.0])

    assert result.status == "optimal"
    assert np.allclose(result.x, [2.0, 1.0])
    assert result.objective_value == pytest.approx(1.5)
    assert result.num_fixed_vars == 1


def test_infeasible_fixed_child_is_reported_infeasible():
    problem = MILPProblem.from_standard(
        c=[1.0],
        G=[[-1.0]],
        h=[-1.0],
        sense="max",
        lb=[0.0],
        ub=[1.0],
        var_types=["B"],
    )
    result = solve_node(problem, [0.0], [0.0])

    assert result.status == "infeasible"
    assert result.x is None


def test_bound_pruning_remains_active_with_two_phase_backend():
    result = solve_milp(
        build_production_expansion(),
        lp_backend="two_phase_simplex",
    )

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(31.5)
    assert result.num_pruned_bound >= 1


def test_complete_small_milp_is_solved_with_two_phase_backend():
    result = solve_milp(build_fixed_charge(), lp_backend="two_phase_simplex")

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(28.5)
    assert result.lp_backend == "two_phase_simplex"
    assert result.num_simplex_iterations > 0
    assert result.lp_runtime_sec > 0.0


def test_minimization_milp_matches_active_set():
    problem = build_production_expansion()
    active = solve_milp(problem, lp_backend="active_set")
    custom = solve_milp(problem, lp_backend="two_phase_simplex")

    assert active.status == custom.status == "optimal"
    assert custom.objective_value == pytest.approx(active.objective_value)


def test_maximization_milp_is_solved_in_original_direction():
    result = solve_milp(build_switch_problem(), lp_backend="two_phase_simplex")

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(1.5)
    assert np.allclose(result.x, [2.0, 1.0])


def test_repeated_bnb_runs_are_deterministic_except_for_time():
    problem = build_production_expansion()
    first = solve_milp(problem, lp_backend="two_phase_simplex")
    second = solve_milp(problem, lp_backend="two_phase_simplex")

    assert first.status == second.status == "optimal"
    assert first.objective_value == pytest.approx(second.objective_value)
    assert np.allclose(first.x, second.x)
    assert first.num_nodes == second.num_nodes
    assert first.num_lp_solved == second.num_lp_solved
    assert first.num_simplex_iterations == second.num_simplex_iterations
    assert first.log == second.log


def test_two_phase_bnb_matches_scipy_highs():
    pytest.importorskip("scipy.optimize")
    problem = build_fixed_charge()
    custom = solve_milp(problem, lp_backend="two_phase_simplex")
    highs = solve_milp(problem, lp_backend="scipy_highs")

    assert custom.status == highs.status == "optimal"
    assert custom.objective_value == pytest.approx(highs.objective_value)


def test_two_phase_bnb_matches_gurobi_when_available():
    gp = pytest.importorskip("gurobipy")
    problem = build_production_expansion()
    model = gp.Model("two_phase_bnb_test")
    model.Params.OutputFlag = 0
    variables = [
        model.addVar(
            lb=float(problem.lb[index]),
            ub=float(problem.ub[index]),
            vtype=gp.GRB.BINARY if kind == "B" else gp.GRB.CONTINUOUS,
        )
        for index, kind in enumerate(problem.var_types)
    ]
    model.setObjective(
        gp.quicksum(float(problem.c[index]) * variable for index, variable in enumerate(variables)),
        gp.GRB.MINIMIZE,
    )
    for row in range(problem.num_constraints):
        model.addConstr(
            gp.quicksum(float(problem.G[row, col]) * variables[col] for col in range(problem.num_vars))
            <= float(problem.h[row])
        )
    model.optimize()

    custom = solve_milp(problem, lp_backend="two_phase_simplex")
    assert model.Status == gp.GRB.OPTIMAL
    assert custom.status == "optimal"
    assert custom.objective_value == pytest.approx(float(model.ObjVal))


def test_active_set_backend_still_solves_existing_case():
    result = solve_milp(build_fixed_charge(), lp_backend="active_set")

    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(28.5)
    assert result.num_lp_candidates_checked > 0


def test_active_set_remains_the_default_backend():
    signature = inspect.signature(solve_milp)
    result = solve_milp(build_fixed_charge())

    assert signature.parameters["lp_backend"].default == "active_set"
    assert result.lp_backend == "active_set"
    assert result.num_lp_candidates_checked > 0


def test_iteration_limit_terminates_bnb_without_reporting_optimal():
    result = solve_milp(
        build_switch_problem(),
        lp_backend="two_phase_simplex",
        use_matrix_presolve=False,
        max_lp_iterations=0,
    )

    assert result.status == "iteration_limit"
    assert result.status != "optimal"
    assert result.num_pruned_infeasible == 0


def test_numerical_error_terminates_bnb_without_infeasibility_prune():
    problem = build_switch_problem()
    solver = BranchAndBoundSolver(problem, lp_backend="two_phase_simplex")

    def numerical_error_backend(*args, **kwargs):
        return LPResult(
            status="numerical_error",
            objective_value=None,
            x=None,
            num_candidates_checked=0,
            message="synthetic numerical failure",
            backend="two_phase_simplex",
        )

    solver.lp_solver = numerical_error_backend
    result = solver.solve()

    assert result.status == "numerical_error"
    assert result.status != "optimal"
    assert result.num_pruned_infeasible == 0
