from __future__ import annotations

from .lp_active_set import solve_lp_relaxation


def get_lp_relaxation_solver(lp_backend: str):
    backend=str(lp_backend).lower()
    if backend=="active_set":
        return solve_lp_relaxation
    if backend=="scipy_highs":
        from .lp_scipy_highs import solve_lp_relaxation_scipy_highs

        return solve_lp_relaxation_scipy_highs
    if backend in {"tableau_simplex","custom_tableau"}:
        from .lp_tableau_simplex import solve_lp_relaxation_tableau

        return solve_lp_relaxation_tableau
    if backend in {"two_phase_simplex","custom_two_phase"}:
        from .lp_two_phase_simplex import solve_lp_relaxation_two_phase

        return solve_lp_relaxation_two_phase
    raise ValueError(
        'lp_backend must be "active_set", "scipy_highs", "tableau_simplex", '
        'or "two_phase_simplex"'
    )
