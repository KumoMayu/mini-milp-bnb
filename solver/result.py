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


LPRelaxationResult=LPResult


@dataclass
class MILPResult:
    status: str
    objective_value: float | None
    x: np.ndarray | None
    x_continuous: np.ndarray | None
    y_integer: np.ndarray | None
    internal_objective_value: float | None
    best_bound: float | None
    num_nodes: int
    num_lp_solved: int
    num_pruned_infeasible: int
    num_pruned_bound: int
    num_pruned_optimality: int
    num_integer_solutions: int
    runtime_sec: float
    log: list[str]

    def simple_summary(self) -> str:
        if self.x is None:
            solution="None"
        else:
            solution=np.array2string(self.x,precision=6,suppress_small=True,separator=", ")

        lines=[
            f"Status: {self.status}",
            f"Objective value: {self.objective_value}",
            f"Solution: {solution}",
            f"Best bound: {self.best_bound}",
            f"Nodes explored: {self.num_nodes}",
            f"LP relaxations solved: {self.num_lp_solved}",
            f"Pruned by infeasibility: {self.num_pruned_infeasible}",
            f"Pruned by bound: {self.num_pruned_bound}",
            f"Pruned by optimality: {self.num_pruned_optimality}",
            f"Integer solutions found: {self.num_integer_solutions}",
            f"Runtime seconds: {self.runtime_sec:.6f}",
        ]
        return "\n".join(lines)
