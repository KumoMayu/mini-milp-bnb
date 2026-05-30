from __future__ import annotations

from .branch_and_bound import BranchAndBoundSolver


def solve_milp(
    problem,
    tol: float = 1e-8,
    max_nodes: int = 10000,
    node_selection: str = "dfs",
    branching_rule: str = "most_fractional",
    verbose: bool = False,
):
    """Convenience API for solving a MILPProblem with the B&B solver."""
    solver=BranchAndBoundSolver(
        problem=problem,
        tol=tol,
        max_nodes=max_nodes,
        node_selection=node_selection,
        branching_rule=branching_rule,
        verbose=verbose,
    )
    return solver.solve()
