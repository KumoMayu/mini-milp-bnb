from __future__ import annotations

import numpy as np
import pytest

from solver.lp_standard_form import standardize_general_lp
from solver.lp_tableau_simplex import ITERATION_LIMIT, OPTIMAL, UNBOUNDED
from solver.lp_two_phase_simplex import (
    INFEASIBLE,
    UNSUPPORTED,
    TwoPhaseTableauSimplexSolver,
)


TOL = 1e-8


def assert_general_optimal(
    result,
    c,
    A,
    b,
    constraint_senses,
    lb,
    ub,
    expected_objective,
    objective_sense="max",
):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    assert result.status == OPTIMAL
    assert result.objective_value == pytest.approx(expected_objective, abs=TOL)
    assert result.x is not None
    assert np.all(result.x >= lb - TOL)
    assert np.all(result.x[np.isfinite(ub)] <= ub[np.isfinite(ub)] + TOL)
    activities = A @ result.x
    for activity, rhs, row_sense in zip(activities, b, constraint_senses):
        if row_sense == "<=":
            assert activity <= rhs + TOL
        elif row_sense == ">=":
            assert activity >= rhs - TOL
        else:
            assert activity == pytest.approx(rhs, abs=TOL)
    assert float(c @ result.x) == pytest.approx(result.objective_value, abs=TOL)
    assert result.num_iterations == (
        result.phase_one_iterations + result.phase_two_iterations
    )
    assert result.solve_time == result.runtime_sec

    try:
        from scipy.optimize import linprog
    except ImportError:
        return

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    for row, row_sense in enumerate(constraint_senses):
        if row_sense == "<=":
            A_ub.append(A[row])
            b_ub.append(b[row])
        elif row_sense == ">=":
            A_ub.append(-A[row])
            b_ub.append(-b[row])
        else:
            A_eq.append(A[row])
            b_eq.append(b[row])
    reference = linprog(
        c if objective_sense == "min" else -c,
        A_ub=None if not A_ub else np.asarray(A_ub),
        b_ub=None if not b_ub else np.asarray(b_ub),
        A_eq=None if not A_eq else np.asarray(A_eq),
        b_eq=None if not b_eq else np.asarray(b_eq),
        bounds=list(zip(lb, ub)),
        method="highs",
    )
    assert reference.success, reference.message
    reference_objective = float(reference.fun)
    if objective_sense == "max":
        reference_objective = -reference_objective
    assert result.objective_value == pytest.approx(reference_objective, abs=TOL)


def test_greater_equal_constraint_optimal():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[1.0],
        A=[[1.0]],
        b=[1.0],
        constraint_senses=[">="],
        lb=[0.0],
        ub=[3.0],
    )

    assert_general_optimal(result, [1], [[1]], [1], [">="], [0], [3], 3.0)
    assert result.phase_one_iterations > 0


def test_equality_constraint_optimal():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[1.0],
        A=[[1.0]],
        b=[2.0],
        constraint_senses=["="],
        lb=[0.0],
        ub=[np.inf],
    )

    assert_general_optimal(result, [1], [[1]], [2], ["="], [0], [np.inf], 2.0)


def test_negative_rhs_flips_constraint_direction():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[1.0],
        A=[[-1.0]],
        b=[-2.0],
        constraint_senses=["<="],
        lb=[0.0],
        ub=[4.0],
    )

    assert_general_optimal(result, [1], [[-1]], [-2], ["<="], [0], [4], 4.0)


def test_nonzero_lower_bound_is_shifted():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[-1.0],
        A=np.zeros((0, 1)),
        b=[],
        constraint_senses=[],
        lb=[2.0],
        ub=[5.0],
    )

    assert_general_optimal(
        result,
        [-1],
        np.zeros((0, 1)),
        [],
        [],
        [2],
        [5],
        -2.0,
    )
    assert np.allclose(result.x, [2.0])


def test_finite_lower_and_upper_bounds():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[1.0],
        A=np.zeros((0, 1)),
        b=[],
        constraint_senses=[],
        lb=[1.0],
        ub=[4.0],
    )

    assert_general_optimal(
        result,
        [1],
        np.zeros((0, 1)),
        [],
        [],
        [1],
        [4],
        4.0,
    )


def test_fixed_variable_is_eliminated_and_recovered():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[2.0, 1.0],
        A=[[1.0, 1.0]],
        b=[10.0],
        constraint_senses=["<="],
        lb=[3.0, 0.0],
        ub=[3.0, 4.0],
    )

    assert_general_optimal(
        result,
        [2, 1],
        [[1, 1]],
        [10],
        ["<="],
        [3, 0],
        [3, 4],
        10.0,
    )
    assert np.allclose(result.x, [3.0, 4.0])
    assert result.num_fixed_vars == 1


def test_lower_bound_objective_constant_is_restored():
    standard = standardize_general_lp(
        c=[2.0],
        A=[[1.0]],
        b=[5.0],
        constraint_senses=["<="],
        lb=[3.0],
        ub=[np.inf],
        sense="max",
    )
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[2.0],
        A=[[1.0]],
        b=[5.0],
        constraint_senses=["<="],
        lb=[3.0],
        ub=[np.inf],
    )

    assert standard.objective_constant_internal == pytest.approx(6.0)
    assert_general_optimal(result, [2], [[1]], [5], ["<="], [3], [np.inf], 10.0)


def test_phase_one_detects_infeasible_problem():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[1.0],
        A=[[1.0], [1.0]],
        b=[2.0, 1.0],
        constraint_senses=[">=", "<="],
        lb=[0.0],
        ub=[np.inf],
    )

    assert result.status == INFEASIBLE
    assert result.x is None
    assert result.objective_value is None
    assert result.phase_one_iterations > 0


def test_zero_artificial_basic_is_pivoted_out_after_phase_one():
    standard = standardize_general_lp(
        c=[0.0],
        A=[[-1.0]],
        b=[0.0],
        constraint_senses=["="],
        lb=[0.0],
        ub=[np.inf],
    )
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[0.0],
        A=[[-1.0]],
        b=[0.0],
        constraint_senses=["="],
        lb=[0.0],
        ub=[np.inf],
    )

    assert_general_optimal(result, [0], [[-1]], [0], ["="], [0], [np.inf], 0.0)
    assert any("I-cleanup" in line for line in result.iteration_log)
    remaining_columns = standard.num_tableau_variables - len(standard.artificial_indices)
    assert all(index < remaining_columns for index in result.basis_indices)


def test_redundant_equality_row_is_removed():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[1.0],
        A=[[1.0], [1.0]],
        b=[1.0, 1.0],
        constraint_senses=["=", "="],
        lb=[0.0],
        ub=[np.inf],
    )

    assert_general_optimal(
        result,
        [1],
        [[1], [1]],
        [1, 1],
        ["=", "="],
        [0],
        [np.inf],
        1.0,
    )
    assert len(result.basis_indices) == 1


def test_general_form_unbounded_problem():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[1.0],
        A=[[1.0]],
        b=[1.0],
        constraint_senses=[">="],
        lb=[0.0],
        ub=[np.inf],
    )

    assert result.status == UNBOUNDED
    assert result.x is None
    assert result.phase_one_iterations > 0


def test_mixed_constraint_senses():
    c = [3.0, 2.0]
    A = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    b = [5.0, 1.0, 2.0]
    senses = ["<=", ">=", "="]
    result = TwoPhaseTableauSimplexSolver().solve(
        c=c,
        A=A,
        b=b,
        constraint_senses=senses,
        lb=[0.0, 0.0],
        ub=[np.inf, np.inf],
    )

    assert_general_optimal(result, c, A, b, senses, [0, 0], [np.inf, np.inf], 13.0)
    assert np.allclose(result.x, [3.0, 2.0])


def test_general_form_minimization():
    c = [1.0, 2.0]
    A = [[1.0, 1.0]]
    b = [4.0]
    senses = [">="]
    result = TwoPhaseTableauSimplexSolver().solve(
        c=c,
        A=A,
        b=b,
        constraint_senses=senses,
        lb=[1.0, 1.0],
        ub=[np.inf, np.inf],
        sense="min",
    )

    assert_general_optimal(
        result,
        c,
        A,
        b,
        senses,
        [1, 1],
        [np.inf, np.inf],
        5.0,
        objective_sense="min",
    )
    assert np.allclose(result.x, [3.0, 1.0])


def test_general_solver_is_deterministic():
    inputs = {
        "c": [3.0, 2.0],
        "A": [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        "b": [5.0, 1.0, 2.0],
        "constraint_senses": ["<=", ">=", "="],
        "lb": [0.0, 0.0],
        "ub": [np.inf, np.inf],
    }
    solver = TwoPhaseTableauSimplexSolver()

    first = solver.solve(**inputs)
    second = solver.solve(**inputs)

    assert first.status == second.status == OPTIMAL
    assert first.objective_value == pytest.approx(second.objective_value)
    assert np.allclose(first.x, second.x)
    assert first.phase_one_iterations == second.phase_one_iterations
    assert first.phase_two_iterations == second.phase_two_iterations
    assert first.basis_indices == second.basis_indices
    assert first.iteration_log == second.iteration_log


def test_random_general_lps_match_scipy_highs():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(20260722)
    solver = TwoPhaseTableauSimplexSolver(tolerance=1e-9, max_iterations=1000)

    for case_index in range(20):
        n = 1 + case_index % 4
        m = 3 + case_index % 3
        lb = rng.uniform(-2.0, 1.0, size=n)
        ub = lb + rng.uniform(1.0, 5.0, size=n)
        feasible_x = rng.uniform(lb, ub)
        A = rng.uniform(-2.0, 2.0, size=(m, n))
        senses = tuple(("<=", ">=", "=")[row % 3] for row in range(m))
        b = np.empty(m, dtype=float)
        for row, row_sense in enumerate(senses):
            activity = float(A[row] @ feasible_x)
            margin = float(rng.uniform(0.1, 2.0))
            if row_sense == "<=":
                b[row] = activity + margin
            elif row_sense == ">=":
                b[row] = activity - margin
            else:
                b[row] = activity
        c = rng.uniform(-3.0, 3.0, size=n)
        objective_sense = "min" if case_index % 2 else "max"

        custom = solver.solve(c, A, b, senses, lb, ub, objective_sense)
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        for row, row_sense in enumerate(senses):
            if row_sense == "<=":
                A_ub.append(A[row])
                b_ub.append(b[row])
            elif row_sense == ">=":
                A_ub.append(-A[row])
                b_ub.append(-b[row])
            else:
                A_eq.append(A[row])
                b_eq.append(b[row])
        reference = scipy_optimize.linprog(
            c if objective_sense == "min" else -c,
            A_ub=None if not A_ub else np.asarray(A_ub),
            b_ub=None if not b_ub else np.asarray(b_ub),
            A_eq=None if not A_eq else np.asarray(A_eq),
            b_eq=None if not b_eq else np.asarray(b_eq),
            bounds=list(zip(lb, ub)),
            method="highs",
        )

        assert reference.success, reference.message
        reference_objective = float(reference.fun)
        if objective_sense == "max":
            reference_objective = -reference_objective
        assert_general_optimal(
            custom,
            c,
            A,
            b,
            senses,
            lb,
            ub,
            reference_objective,
            objective_sense=objective_sense,
        )


def test_true_free_variable_returns_unsupported():
    result = TwoPhaseTableauSimplexSolver().solve(
        c=[1.0],
        A=np.zeros((0, 1)),
        b=[],
        constraint_senses=[],
        lb=[-np.inf],
        ub=[np.inf],
    )

    assert result.status == UNSUPPORTED
    assert "finite lower bound" in result.message


def test_phase_one_iteration_limit_is_not_optimal():
    result = TwoPhaseTableauSimplexSolver(max_iterations=0).solve(
        c=[1.0],
        A=[[1.0]],
        b=[1.0],
        constraint_senses=[">="],
        lb=[0.0],
        ub=[3.0],
    )

    assert result.status == ITERATION_LIMIT
    assert result.phase_one_iterations == 0
    assert result.phase_two_iterations == 0
