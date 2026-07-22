from __future__ import annotations

import numpy as np
import pytest

from solver import MILPProblem, TableauSimplexSolver
from solver.lp_backends import get_lp_relaxation_solver
from solver.lp_standard_form import UnsupportedStandardFormError
from solver.lp_tableau_simplex import ITERATION_LIMIT, OPTIMAL, UNBOUNDED


TOL = 1e-8


def assert_optimal(result, c, A, b, expected_objective):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    assert result.status == OPTIMAL
    assert result.objective_value == pytest.approx(expected_objective, abs=TOL)
    assert result.x is not None
    assert np.all(A @ result.x <= b + TOL)
    assert np.all(result.x >= -TOL)
    assert float(c @ result.x) == pytest.approx(result.objective_value, abs=TOL)
    assert result.backend == "tableau_simplex"
    assert result.num_candidates_checked == 0
    assert result.runtime_sec >= 0.0


def test_unique_optimal_solution():
    c = [1.0]
    A = [[1.0]]
    b = [3.0]

    result = TableauSimplexSolver().solve(c, A, b)

    assert_optimal(result, c, A, b, 3.0)
    assert np.allclose(result.x, [3.0])


def test_axis_optimum():
    c = [2.0, 1.0]
    A = [[1.0, 1.0]]
    b = [4.0]

    result = TableauSimplexSolver().solve(c, A, b)

    assert_optimal(result, c, A, b, 8.0)
    assert np.allclose(result.x, [4.0, 0.0])


def test_multiple_constraints_form_optimal_vertex():
    c = [3.0, 2.0]
    A = [[1.0, 1.0], [2.0, 1.0]]
    b = [4.0, 5.0]

    result = TableauSimplexSolver().solve(c, A, b)

    assert_optimal(result, c, A, b, 9.0)
    assert np.allclose(result.x, [1.0, 3.0])


def test_inactive_constraint_does_not_change_optimum():
    c = [2.0, 1.0]
    A = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    b = [4.0, 3.0, 10.0]

    result = TableauSimplexSolver().solve(c, A, b)

    assert_optimal(result, c, A, b, 7.0)
    assert np.allclose(result.x, [3.0, 1.0])
    assert float(np.asarray(A)[2] @ result.x) < b[2]


def test_unbounded_problem():
    result = TableauSimplexSolver().solve(c=[1.0], A=[[-1.0]], b=[0.0])

    assert result.status == UNBOUNDED
    assert result.objective_value is None
    assert result.x is None


def test_degenerate_basic_feasible_solution():
    c = [1.0, 1.0]
    A = [[1.0, 0.0], [1.0, 1.0]]
    b = [0.0, 1.0]

    result = TableauSimplexSolver().solve(c, A, b)

    assert_optimal(result, c, A, b, 1.0)
    assert np.allclose(result.x, [0.0, 1.0])
    assert result.num_iterations == 2


def test_multiple_optimal_solutions_return_one_feasible_optimum():
    c = [1.0, 1.0]
    A = [[1.0, 1.0]]
    b = [1.0]

    result = TableauSimplexSolver().solve(c, A, b)

    assert_optimal(result, c, A, b, 1.0)
    assert result.x.sum() == pytest.approx(1.0)


def test_zero_objective_coefficients():
    c = [0.0, 0.0]
    A = [[1.0, 1.0]]
    b = [2.0]

    result = TableauSimplexSolver().solve(c, A, b)

    assert_optimal(result, c, A, b, 0.0)
    assert np.allclose(result.x, [0.0, 0.0])
    assert result.num_iterations == 0


def test_initial_slack_basis_is_already_optimal():
    c = [-1.0, -2.0]
    A = [[1.0, 1.0]]
    b = [3.0]

    result = TableauSimplexSolver().solve(c, A, b)

    assert_optimal(result, c, A, b, 0.0)
    assert np.allclose(result.x, [0.0, 0.0])
    assert result.num_iterations == 0
    assert result.basis_indices == (2,)


def test_iteration_limit_is_not_reported_as_optimal():
    result = TableauSimplexSolver(max_iterations=1).solve(
        c=[3.0, 2.0],
        A=[[1.0, 1.0], [2.0, 1.0]],
        b=[4.0, 5.0],
    )

    assert result.status == ITERATION_LIMIT
    assert result.num_iterations == 1
    assert result.x is not None
    assert result.objective_value == pytest.approx(7.5)


def test_negative_rhs_is_explicitly_unsupported():
    with pytest.raises(UnsupportedStandardFormError, match="requires b >= 0"):
        TableauSimplexSolver().solve(c=[1.0], A=[[-1.0]], b=[-1.0])


def test_repeated_runs_are_deterministic():
    solver = TableauSimplexSolver()
    inputs = {
        "c": [3.0, 2.0],
        "A": [[1.0, 1.0], [2.0, 1.0]],
        "b": [4.0, 5.0],
    }

    first = solver.solve(**inputs)
    second = solver.solve(**inputs)

    assert first.status == second.status == OPTIMAL
    assert first.objective_value == pytest.approx(second.objective_value)
    assert np.allclose(first.x, second.x)
    assert first.num_iterations == second.num_iterations
    assert first.basis_indices == second.basis_indices
    assert first.iteration_log == second.iteration_log


def test_minimization_objective_is_recovered_in_original_direction():
    c = [-1.0]
    A = [[1.0]]
    b = [2.0]

    result = TableauSimplexSolver().solve(c, A, b, sense="min")

    assert_optimal(result, c, A, b, -2.0)
    assert np.allclose(result.x, [2.0])


def test_backend_registry_exposes_tableau_without_changing_default():
    problem = MILPProblem.from_standard(
        c=[1.0],
        G=[[1.0]],
        h=[2.0],
        lb=[0.0],
        ub=[5.0],
        var_types=["C"],
        sense="max",
    )

    backend = get_lp_relaxation_solver("tableau_simplex")
    result = backend(problem, problem.lb, problem.ub)

    assert_optimal(
        result,
        problem.internal_c,
        np.vstack([problem.G, [[1.0]]]),
        [2.0, 5.0],
        2.0,
    )
    assert get_lp_relaxation_solver("active_set").__module__ == "solver.lp_active_set"


def test_verbose_flag_controls_pivot_output(capsys):
    inputs = {
        "c": [3.0, 2.0],
        "A": [[1.0, 1.0], [2.0, 1.0]],
        "b": [4.0, 5.0],
    }

    TableauSimplexSolver(verbose=False).solve(**inputs)
    assert capsys.readouterr().out == ""

    TableauSimplexSolver(verbose=True).solve(**inputs)
    output = capsys.readouterr().out
    assert "iteration | entering | leaving | objective | basis" in output
    assert "x1" in output
    assert "status=optimal" in output
