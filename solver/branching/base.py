from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BranchingContext:
    problem: object
    node_id: int
    node_depth: int
    node_lb: np.ndarray
    node_ub: np.ndarray
    lp_result: object
    candidate_indices: tuple[int, ...]
    incumbent_internal_value: float | None
    current_node_internal_bound: float | None
    tolerance: float


class BranchingPolicy(Protocol):
    def select_variable(self, context: BranchingContext) -> int:
        """Return a global variable index from context.candidate_indices."""
