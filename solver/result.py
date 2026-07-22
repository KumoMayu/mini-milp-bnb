from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LPResult:
    status: str
    objective_value: float | None
    x: np.ndarray | None
    num_candidates_checked: int
    message: str
    num_free_vars: int = 0
    num_fixed_vars: int = 0
    num_removed_rows: int = 0
    num_tightened_bounds: int = 0
    backend: str = "active_set"
    num_iterations: int = 0
    basis_indices: tuple[int, ...] = ()
    runtime_sec: float = 0.0
    iteration_log: tuple[str, ...] = ()
    phase_one_iterations: int = 0
    phase_two_iterations: int = 0

    @property
    def solve_time(self) -> float:
        return self.runtime_sec


LPRelaxationResult = LPResult


@dataclass
class MILPResult:
    status: str
    objective_value: float | None
    x: np.ndarray | None
    x_continuous: np.ndarray | None
    y_integer: np.ndarray | None
    internal_objective_value: float | None
    best_bound: float | None
    global_bound: float | None
    relative_gap: float | None
    num_nodes: int
    num_lp_solved: int
    num_pruned_infeasible: int
    num_pruned_bound: int
    num_pruned_optimality: int
    num_integer_solutions: int
    num_lp_candidates_checked: int
    num_fixed_vars_eliminated: int
    num_removed_rows: int
    num_tightened_bounds: int
    num_free_vars_total: int
    num_heuristic_lp_solved: int
    initial_incumbent_found: bool
    runtime_sec: float
    log: list[str]

    def simple_summary(self) -> str:
        if self.x is None:
            solution = "None"
        else:
            solution = np.array2string(
                self.x,
                precision=6,
                suppress_small=True,
                separator=", ",
            )

        lines = [
            f"Status: {self.status}",
            f"Objective value: {self.objective_value}",
            f"Solution: {solution}",
            f"Best bound: {self.best_bound}",
            f"Global bound: {self.global_bound}",
            f"Relative gap: {self.relative_gap}",
            f"Nodes explored: {self.num_nodes}",
            f"LP relaxations solved: {self.num_lp_solved}",
            f"Active-set candidates checked: {self.num_lp_candidates_checked}",
            f"Fixed variables eliminated: {self.num_fixed_vars_eliminated}",
            f"Rows removed by presolve: {self.num_removed_rows}",
            f"Bounds tightened by presolve: {self.num_tightened_bounds}",
            f"Pruned by infeasibility: {self.num_pruned_infeasible}",
            f"Pruned by bound: {self.num_pruned_bound}",
            f"Pruned by optimality: {self.num_pruned_optimality}",
            f"Integer solutions found: {self.num_integer_solutions}",
            f"Initial incumbent found: {self.initial_incumbent_found}",
            f"Runtime seconds: {self.runtime_sec:.6f}",
        ]
        return "\n".join(lines)
