from __future__ import annotations

from itertools import combinations

import numpy as np

from .matrix_presolve import presolve_node_matrix, reconstruct_solution
from .result import LPResult


def compress_fixed_variables(c,G,h,lb,ub,tol: float = 1e-8):
    """Eliminate variables fixed by node bounds before active-set enumeration.

    This is a transparent preprocessing step, not a black-box LP method. If
    lb_i == ub_i at a B&B node, z_i is already known. Moving G_F z_F to the
    right-hand side reduces the LP dimension from all variables to the still
    free variables.
    """
    c=np.asarray(c,dtype=float)
    G=np.asarray(G,dtype=float)
    h=np.asarray(h,dtype=float)
    lb=np.asarray(lb,dtype=float)
    ub=np.asarray(ub,dtype=float)

    fixed_mask=np.abs(ub-lb)<=tol
    free_indices=np.flatnonzero(~fixed_mask)
    fixed_indices=np.flatnonzero(fixed_mask)

    fixed_values=np.zeros(len(c),dtype=float)
    if len(fixed_indices)>0:
        fixed_values[fixed_indices]=0.5*(lb[fixed_indices]+ub[fixed_indices])

    c_free=c[free_indices]
    G_free=G[:,free_indices]
    if len(fixed_indices)>0:
        h_shifted=h-G[:,fixed_indices]@fixed_values[fixed_indices]
        fixed_objective_constant=float(c[fixed_indices]@fixed_values[fixed_indices])
    else:
        h_shifted=h.copy()
        fixed_objective_constant=0.0

    return (
        c_free,
        G_free,
        h_shifted,
        lb[free_indices],
        ub[free_indices],
        fixed_values,
        free_indices,
        fixed_objective_constant,
    )


def solve_lp_relaxation(
    problem,
    node_lb,
    node_ub,
    tol: float = 1e-8,
    max_candidates: int | None = None,
    use_matrix_presolve: bool = True,
    matrix_presolve_options=None,
    max_iterations: int | None = None,
) -> LPResult:
    """Solve one node LP relaxation with active-set enumeration.

    The LP backend is intentionally small and prototype-oriented. It builds
    Mz <= q from original constraints plus the current node bounds, enumerates
    candidate vertices, and picks the feasible vertex with maximum internal
    objective value. Its complexity grows with the number of active-set
    combinations. A future simplex backend can replace this function: both
    methods use the fact that LP optima occur at vertices, but simplex walks
    between vertices rather than enumerating every candidate vertex.
    """
    del max_iterations
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
            backend="active_set",
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
                backend="active_set",
            )

        c_free=presolve.c_reduced
        G_free=presolve.G_reduced
        h_shifted=presolve.h_reduced
        lb_free=presolve.lb_reduced
        ub_free=presolve.ub_reduced
        fixed_objective_constant=presolve.objective_constant
        n_free=len(c_free)
        n_fixed=len(presolve.fixed_indices)
        removed_rows=presolve.removed_rows
        tightened_bounds=presolve.tightened_bounds
    else:
        c_free=c
        G_free=problem.G
        h_shifted=problem.h
        lb_free=lb
        ub_free=ub
        fixed_objective_constant=0.0
        n_free=problem.num_vars
        n_fixed=0
        removed_rows=0
        tightened_bounds=0
        presolve=None

    if n_free==0:
        x=presolve.fixed_values.copy()
        return LPResult(
            status="optimal",
            objective_value=float(fixed_objective_constant),
            x=x,
            num_candidates_checked=0,
            message="all variables fixed, fixed point feasible",
            num_free_vars=0,
            num_fixed_vars=n_fixed,
            num_removed_rows=removed_rows,
            num_tightened_bounds=tightened_bounds,
            backend="active_set",
        )

    # Build the compressed node relaxation M_R z_R <= q_R. Bounds are not
    # metadata here: B&B branching works by tightening node lb/ub, and fixed
    # variables have already been moved to h_shifted.
    eye=np.eye(n_free)
    M=np.vstack([G_free,eye,-eye])
    q=np.concatenate([h_shifted,ub_free,-lb_free])

    best_value=None
    best_x_free=None
    checked=0

    # In an n-variable bounded LP, an optimal solution can be found at a
    # vertex. A nondegenerate vertex is determined by n active constraints.
    for active_indices in combinations(range(len(q)),n_free):
        checked+=1
        if max_candidates is not None and checked>max_candidates:
            return LPResult(
                status="candidate_limit",
                objective_value=None,
                x=None,
                num_candidates_checked=checked-1,
                message="active-set candidate limit reached",
                num_free_vars=n_free,
                num_fixed_vars=n_fixed,
                num_removed_rows=removed_rows,
                num_tightened_bounds=tightened_bounds,
                backend="active_set",
            )
        active_M=M[list(active_indices),:]
        active_q=q[list(active_indices)]

        try:
            x_free=np.linalg.solve(active_M,active_q)
        except np.linalg.LinAlgError:
            continue

        if np.any(~np.isfinite(x_free)):
            continue
        if np.any(M@x_free>q+tol):
            continue

        value=float(fixed_objective_constant+c_free@x_free)
        if best_value is None or value>best_value+tol:
            best_value=value
            best_x_free=x_free

    if best_x_free is None:
        return LPResult(
            status="infeasible",
            objective_value=None,
            x=None,
            num_candidates_checked=checked,
            message="no feasible LP vertex found",
            num_free_vars=n_free,
            num_fixed_vars=n_fixed,
            num_removed_rows=removed_rows,
            num_tightened_bounds=tightened_bounds,
            backend="active_set",
        )

    if use_matrix_presolve:
        best_x=reconstruct_solution(presolve,best_x_free)
    else:
        best_x=best_x_free

    return LPResult(
        status="optimal",
        objective_value=float(best_value),
        x=best_x,
        num_candidates_checked=checked,
        message="optimal LP relaxation vertex found",
        num_free_vars=n_free,
        num_fixed_vars=n_fixed,
        num_removed_rows=removed_rows,
        num_tightened_bounds=tightened_bounds,
        backend="active_set",
    )
