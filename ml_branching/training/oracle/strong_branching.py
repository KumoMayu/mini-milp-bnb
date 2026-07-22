from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from solver.branching import BranchingContext
from solver.lp_backends import get_lp_relaxation_solver

from .scoring import child_delta, score_from_deltas


@dataclass(frozen=True)
class StrongBranchCandidate:
    candidate_index: int
    candidate_lp_value: float
    parent_bound: float
    child_0_status: str
    child_0_bound: float | None
    child_0_time_sec: float
    child_0_lp_solved: int
    child_1_status: str
    child_1_bound: float | None
    child_1_time_sec: float
    child_1_lp_solved: int
    delta_0: float
    delta_1: float
    score: float

    @property
    def child0_status(self) -> str:
        return self.child_0_status

    @property
    def child0_bound(self) -> float | None:
        return self.child_0_bound

    @property
    def child0_time_sec(self) -> float:
        return self.child_0_time_sec

    @property
    def child1_status(self) -> str:
        return self.child_1_status

    @property
    def child1_bound(self) -> float | None:
        return self.child_1_bound

    @property
    def child1_time_sec(self) -> float:
        return self.child_1_time_sec


def make_strong_branch_child_bounds(
    node_lb,
    node_ub,
    candidate_index: int,
    branch_value: int,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, str]:
    lb = np.asarray(node_lb, dtype=float).copy()
    ub = np.asarray(node_ub, dtype=float).copy()
    if branch_value == 0:
        ub[candidate_index] = min(float(ub[candidate_index]), 0.0)
        direction = "y_j=0"
    elif branch_value == 1:
        lb[candidate_index] = max(float(lb[candidate_index]), 1.0)
        direction = "y_j=1"
    else:
        raise ValueError("strong branching only supports binary branch values 0 and 1")
    if np.any(lb > ub + tol):
        return lb, ub, "infeasible"
    return lb, ub, "open"


def _probe_child(
    problem,
    node_lb,
    node_ub,
    candidate_index: int,
    value: float,
    lp_solver,
    tol: float,
    max_lp_candidates: int | None,
    use_matrix_presolve: bool,
):
    lb, ub, pre_status = make_strong_branch_child_bounds(
        node_lb,
        node_ub,
        candidate_index,
        int(value),
        tol,
    )
    if pre_status == "infeasible":
        return "infeasible", None, 0.0, 0
    start = perf_counter()
    lp = lp_solver(problem, lb, ub, tol, max_lp_candidates, use_matrix_presolve)
    elapsed = perf_counter() - start
    return lp.status, None if lp.objective_value is None else float(lp.objective_value), elapsed, 1


def strong_branch_candidates(
    context: BranchingContext,
    lp_backend: str = "scipy_highs",
    max_lp_candidates: int | None = None,
    use_matrix_presolve: bool = True,
    score_mode: str = "product",
    infeasible_improvement: float | None = None,
    epsilon: float = 1e-6,
    mu: float = 0.25,
) -> list[StrongBranchCandidate]:
    parent_bound = context.current_node_internal_bound
    if parent_bound is None:
        raise ValueError("strong branching requires current_node_internal_bound")
    lp_solver = get_lp_relaxation_solver(lp_backend)
    records: list[StrongBranchCandidate] = []
    for candidate in context.candidate_indices:
        if int(candidate) not in context.candidate_indices:
            raise ValueError("candidate index is not in context.candidate_indices")
        child_0_status, child_0_bound, child_0_time, child_0_lp_solved = _probe_child(
            context.problem,
            context.node_lb,
            context.node_ub,
            int(candidate),
            0.0,
            lp_solver,
            context.tolerance,
            max_lp_candidates,
            use_matrix_presolve,
        )
        child_1_status, child_1_bound, child_1_time, child_1_lp_solved = _probe_child(
            context.problem,
            context.node_lb,
            context.node_ub,
            int(candidate),
            1.0,
            lp_solver,
            context.tolerance,
            max_lp_candidates,
            use_matrix_presolve,
        )
        delta_0 = child_delta(parent_bound, child_0_bound, child_0_status, infeasible_improvement)
        delta_1 = child_delta(parent_bound, child_1_bound, child_1_status, infeasible_improvement)
        score = score_from_deltas(delta_0, delta_1, mode=score_mode, epsilon=epsilon, mu=mu)
        records.append(
            StrongBranchCandidate(
                candidate_index=int(candidate),
                candidate_lp_value=float(context.lp_result.x[int(candidate)]),
                parent_bound=float(parent_bound),
                child_0_status=child_0_status,
                child_0_bound=child_0_bound,
                child_0_time_sec=child_0_time,
                child_0_lp_solved=child_0_lp_solved,
                child_1_status=child_1_status,
                child_1_bound=child_1_bound,
                child_1_time_sec=child_1_time,
                child_1_lp_solved=child_1_lp_solved,
                delta_0=float(delta_0),
                delta_1=float(delta_1),
                score=float(score),
            )
        )
    return records


class StrongBranchingPolicy:
    """Strong branching as an explicit policy, mainly for expert comparison."""

    def __init__(
        self,
        lp_backend: str = "scipy_highs",
        max_lp_candidates: int | None = None,
        use_matrix_presolve: bool = True,
        score_mode: str = "product",
        infeasible_improvement: float | None = None,
        epsilon: float = 1e-6,
        mu: float = 0.25,
    ) -> None:
        self.lp_backend = lp_backend
        self.max_lp_candidates = max_lp_candidates
        self.use_matrix_presolve = use_matrix_presolve
        self.score_mode = score_mode
        self.infeasible_improvement = infeasible_improvement
        self.epsilon = epsilon
        self.mu = mu
        self.probe_lp_solved = 0
        self.probe_time_sec = 0.0
        self.last_records: list[StrongBranchCandidate] = []

    def select_variable(self, context: BranchingContext) -> int:
        records = strong_branch_candidates(
            context,
            lp_backend=self.lp_backend,
            max_lp_candidates=self.max_lp_candidates,
            use_matrix_presolve=self.use_matrix_presolve,
            score_mode=self.score_mode,
            infeasible_improvement=self.infeasible_improvement,
            epsilon=self.epsilon,
            mu=self.mu,
        )
        self.last_records = records
        self.probe_lp_solved += sum(r.child_0_lp_solved + r.child_1_lp_solved for r in records)
        self.probe_time_sec += sum(r.child_0_time_sec + r.child_1_time_sec for r in records)
        best = max(records, key=lambda r: (r.score, -r.candidate_index))
        return int(best.candidate_index)
