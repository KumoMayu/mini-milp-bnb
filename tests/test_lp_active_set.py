import numpy as np
import pytest

from solver import MILPProblem
from solver.lp_active_set import compress_fixed_variables, solve_lp_relaxation


def test_lp_relaxation_uses_original_constraints_and_bounds():
    problem=MILPProblem.from_standard(
        c=[1],
        G=[[1]],
        h=[2],
        sense="max",
        lb=[0],
        ub=[5],
        var_types=["C"],
    )

    result=solve_lp_relaxation(problem,problem.lb,problem.ub)

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(2.0)
    assert result.num_candidates_checked==2
    assert result.num_removed_rows==1
    assert result.num_tightened_bounds==1


def test_fixed_variable_compression_shifts_constraints_and_objective():
    c=np.array([2.0,1.0])
    G=np.array([[1.0,1.0]])
    h=np.array([4.0])
    lb=np.array([2.0,0.0])
    ub=np.array([2.0,4.0])

    c_free,G_free,h_shifted,lb_free,ub_free,fixed_values,free_indices,const=compress_fixed_variables(
        c,G,h,lb,ub
    )

    assert np.allclose(c_free,[1.0])
    assert np.allclose(G_free,[[1.0]])
    assert np.allclose(h_shifted,[2.0])
    assert np.allclose(lb_free,[0.0])
    assert np.allclose(ub_free,[4.0])
    assert np.allclose(fixed_values,[2.0,0.0])
    assert np.allclose(free_indices,[1])
    assert const==pytest.approx(4.0)


def test_compressed_lp_returns_full_solution_and_same_objective():
    problem=MILPProblem.from_standard(
        c=[2,1],
        G=[[1,1]],
        h=[4],
        sense="max",
        lb=[0,0],
        ub=[5,5],
        var_types=["C","C"],
    )

    result=solve_lp_relaxation(problem,np.array([2.0,0.0]),np.array([2.0,5.0]))

    assert result.status=="optimal"
    assert result.objective_value==pytest.approx(6.0)
    assert np.allclose(result.x,[2.0,2.0])
    assert result.num_free_vars==1
    assert result.num_fixed_vars==1


def test_lp_relaxation_can_disable_matrix_presolve():
    problem=MILPProblem.from_standard(
        c=[2,1],
        G=[[1,1]],
        h=[4],
        sense="max",
        lb=[0,0],
        ub=[5,5],
        var_types=["C","C"],
    )

    with_presolve=solve_lp_relaxation(
        problem,
        np.array([2.0,0.0]),
        np.array([2.0,5.0]),
        use_matrix_presolve=True,
    )
    without_presolve=solve_lp_relaxation(
        problem,
        np.array([2.0,0.0]),
        np.array([2.0,5.0]),
        use_matrix_presolve=False,
    )

    assert with_presolve.status=="optimal"
    assert without_presolve.status=="optimal"
    assert with_presolve.objective_value==pytest.approx(without_presolve.objective_value)
    assert np.allclose(with_presolve.x,without_presolve.x)
    assert with_presolve.num_candidates_checked<without_presolve.num_candidates_checked


def test_all_variables_fixed_feasible_or_infeasible():
    feasible=MILPProblem.from_standard(
        c=[1],
        G=[[1]],
        h=[1],
        sense="max",
        lb=[0],
        ub=[1],
        var_types=["B"],
    )
    feasible_result=solve_lp_relaxation(feasible,np.array([1.0]),np.array([1.0]))
    assert feasible_result.status=="optimal"
    assert feasible_result.objective_value==pytest.approx(1.0)
    assert feasible_result.num_candidates_checked==0

    infeasible=MILPProblem.from_standard(
        c=[1],
        G=[[1]],
        h=[0],
        sense="max",
        lb=[0],
        ub=[1],
        var_types=["B"],
    )
    infeasible_result=solve_lp_relaxation(infeasible,np.array([1.0]),np.array([1.0]))
    assert infeasible_result.status=="infeasible"
    assert infeasible_result.x is None


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
