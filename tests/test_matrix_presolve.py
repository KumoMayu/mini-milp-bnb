import numpy as np
import pytest

from solver import presolve_node_matrix, reconstruct_solution


def test_fixed_variable_compression_shifts_matrix_and_objective():
    result=presolve_node_matrix(
        c=[2.0,1.0],
        G=[[1.0,1.0]],
        h=[4.0],
        lb=[2.0,0.0],
        ub=[2.0,5.0],
    )

    assert result.status=="ok"
    assert np.allclose(result.c_reduced,[1.0])
    assert np.allclose(result.G_reduced,[[1.0]])
    assert np.allclose(result.h_reduced,[2.0])
    assert np.allclose(result.lb_reduced,[0.0])
    assert np.allclose(result.ub_reduced,[2.0])
    assert np.allclose(result.free_indices,[1])
    assert np.allclose(result.fixed_indices,[0])
    assert result.objective_constant==pytest.approx(4.0)
    assert result.fixed_variables==1


def test_row_activity_detects_infeasibility():
    result=presolve_node_matrix(
        c=[1.0],
        G=[[1.0]],
        h=[0.0],
        lb=[1.0],
        ub=[2.0],
    )

    assert result.status=="infeasible"
    assert "lower activity" in result.infeasible_reason


def test_redundant_row_is_removed():
    result=presolve_node_matrix(
        c=[1.0],
        G=[[1.0]],
        h=[10.0],
        lb=[0.0],
        ub=[1.0],
    )

    assert result.status=="ok"
    assert result.G_reduced.shape==(0,1)
    assert result.removed_rows==1


def test_bound_tightening_for_positive_and_negative_coefficients():
    upper=presolve_node_matrix(
        c=[1.0],
        G=[[1.0]],
        h=[3.0],
        lb=[0.0],
        ub=[10.0],
    )
    lower=presolve_node_matrix(
        c=[1.0],
        G=[[-1.0]],
        h=[-2.0],
        lb=[0.0],
        ub=[10.0],
    )

    assert upper.status=="ok"
    assert upper.ub_reduced[0]==pytest.approx(3.0)
    assert upper.tightened_bounds>=1
    assert lower.status=="ok"
    assert lower.lb_reduced[0]==pytest.approx(2.0)
    assert lower.tightened_bounds>=1


def test_zero_rows_are_removed_or_detected_as_infeasible():
    feasible=presolve_node_matrix(
        c=[1.0],
        G=[[0.0]],
        h=[1.0],
        lb=[0.0],
        ub=[1.0],
    )
    infeasible=presolve_node_matrix(
        c=[1.0],
        G=[[0.0]],
        h=[-1.0],
        lb=[0.0],
        ub=[1.0],
    )

    assert feasible.status=="ok"
    assert feasible.removed_rows==1
    assert infeasible.status=="infeasible"
    assert "zero row" in infeasible.infeasible_reason


def test_reconstruct_solution_inserts_fixed_values():
    result=presolve_node_matrix(
        c=[2.0,1.0],
        G=[[1.0,1.0]],
        h=[4.0],
        lb=[2.0,0.0],
        ub=[2.0,5.0],
    )

    x_full=reconstruct_solution(result,[1.5])

    assert np.allclose(x_full,[2.0,1.5])
