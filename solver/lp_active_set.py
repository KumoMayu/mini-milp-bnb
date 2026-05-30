from __future__ import annotations

from itertools import combinations

import numpy as np

from .result import LPResult


def solve_lp_relaxation(problem,node_lb,node_ub,tol: float = 1e-8) -> LPResult:
    """Solve one node LP relaxation with active-set enumeration.

    The LP backend is intentionally small and prototype-oriented. It builds
    Mz <= q from original constraints plus the current node bounds, enumerates
    candidate vertices, and picks the feasible vertex with maximum internal
    objective value. Its complexity grows with the number of active-set
    combinations. A future simplex backend can replace this function: both
    methods use the fact that LP optima occur at vertices, but simplex walks
    between vertices rather than enumerating every candidate vertex.
    """
    n=problem.num_vars
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
        )

    # Build the full node relaxation Mz <= q. Bounds are not metadata here:
    # they are constraints, and B&B branching works by tightening node lb/ub.
    eye=np.eye(n)
    M=np.vstack([problem.G,eye,-eye])
    q=np.concatenate([problem.h,ub,-lb])

    best_value=None
    best_x=None
    checked=0

    # In an n-variable bounded LP, an optimal solution can be found at a
    # vertex. A nondegenerate vertex is determined by n active constraints.
    for active_indices in combinations(range(len(q)),n):
        checked+=1
        active_M=M[list(active_indices),:]
        active_q=q[list(active_indices)]

        try:
            x=np.linalg.solve(active_M,active_q)
        except np.linalg.LinAlgError:
            continue

        if np.any(~np.isfinite(x)):
            continue
        if np.any(M@x>q+tol):
            continue

        value=float(c@x)
        if best_value is None or value>best_value+tol:
            best_value=value
            best_x=x

    if best_x is None:
        return LPResult(
            status="infeasible",
            objective_value=None,
            x=None,
            num_candidates_checked=checked,
            message="no feasible LP vertex found",
        )

    return LPResult(
        status="optimal",
        objective_value=float(best_value),
        x=best_x,
        num_candidates_checked=checked,
        message="optimal LP relaxation vertex found",
    )
