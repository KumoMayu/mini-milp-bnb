from __future__ import annotations

import numpy as np

from .matrix_presolve import presolve_node_matrix, reconstruct_solution
from .result import LPResult


def solve_lp_relaxation_scipy_highs(
    problem,
    node_lb,
    node_ub,
    tol: float = 1e-8,
    max_candidates: int | None = None,
    use_matrix_presolve: bool = True,
    matrix_presolve_options=None,
) -> LPResult:
    """Solve one node LP relaxation with scipy.optimize.linprog(method='highs').

    The project still owns the B&B tree, incumbent logic, pruning, and
    branching. This optional backend only replaces the node LP relaxation:

        max c^T z
        s.t. Gz <= h, lb <= z <= ub

    SciPy linprog minimizes, so the LP is passed as min -c^T z.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise RuntimeError(
            "lp_backend='scipy_highs' requires scipy. Install optional dependencies with "
            "'.venv/bin/python -m pip install -r requirements-optional.txt'."
        ) from exc

    c=problem.internal_c
    lb=np.asarray(node_lb,dtype=float)
    ub=np.asarray(node_ub,dtype=float)

    if np.any(lb>ub+tol):
        return LPResult(
            status="infeasible",
            objective_value=None,
            x=None,
            num_candidates_checked=0,
            message="node lower bound exceeds upper bound",
            num_free_vars=0,
            num_fixed_vars=0,
            backend="scipy_highs",
        )

    if use_matrix_presolve:
        presolve=presolve_node_matrix(c,problem.G,problem.h,lb,ub,tol,options=matrix_presolve_options)
        if presolve.status!="ok":
            return LPResult(
                status="infeasible",
                objective_value=None,
                x=None,
                num_candidates_checked=0,
                message=presolve.infeasible_reason or "node matrix presolve detected infeasible node",
                num_free_vars=len(presolve.free_indices),
                num_fixed_vars=len(presolve.fixed_indices),
                num_removed_rows=presolve.removed_rows,
                num_tightened_bounds=presolve.tightened_bounds,
                backend="scipy_highs",
            )
        c_lp=presolve.c_reduced
        G_lp=presolve.G_reduced
        h_lp=presolve.h_reduced
        lb_lp=presolve.lb_reduced
        ub_lp=presolve.ub_reduced
        objective_constant=presolve.objective_constant
        num_fixed=len(presolve.fixed_indices)
        num_free=len(c_lp)
        removed_rows=presolve.removed_rows
        tightened_bounds=presolve.tightened_bounds
    else:
        presolve=None
        c_lp=c
        G_lp=problem.G
        h_lp=problem.h
        lb_lp=lb
        ub_lp=ub
        objective_constant=0.0
        fixed_mask=np.abs(ub-lb)<=tol
        num_fixed=int(np.count_nonzero(fixed_mask))
        num_free=int(len(c)-num_fixed)
        removed_rows=0
        tightened_bounds=0

    if len(c_lp)==0:
        if presolve is None:
            x=np.asarray([],dtype=float)
        else:
            x=presolve.fixed_values.copy()
        return LPResult(
            status="optimal",
            objective_value=float(objective_constant),
            x=x,
            num_candidates_checked=0,
            message="all variables fixed, fixed point feasible",
            num_free_vars=0,
            num_fixed_vars=num_fixed,
            num_removed_rows=removed_rows,
            num_tightened_bounds=tightened_bounds,
            backend="scipy_highs",
        )

    bounds=[(float(lb_lp[i]),float(ub_lp[i])) for i in range(len(c_lp))]
    result=linprog(
        c=-c_lp,
        A_ub=G_lp,
        b_ub=h_lp,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        status="infeasible" if result.status in (2,3) else "lp_error"
        return LPResult(
            status=status,
            objective_value=None,
            x=None,
            num_candidates_checked=0,
            message=str(result.message),
            num_free_vars=num_free,
            num_fixed_vars=num_fixed,
            num_removed_rows=removed_rows,
            num_tightened_bounds=tightened_bounds,
            backend="scipy_highs",
        )

    x_lp=np.asarray(result.x,dtype=float)
    x=x_lp if presolve is None else reconstruct_solution(presolve,x_lp)
    return LPResult(
        status="optimal",
        objective_value=float(objective_constant+c_lp@x_lp),
        x=x,
        num_candidates_checked=0,
        message="optimal LP relaxation solved by scipy highs",
        num_free_vars=num_free,
        num_fixed_vars=num_fixed,
        num_removed_rows=removed_rows,
        num_tightened_bounds=tightened_bounds,
        backend="scipy_highs",
    )
