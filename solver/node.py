from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BBNode:
    node_id: int
    depth: int
    lb: np.ndarray
    ub: np.ndarray
    parent_id: int | None = None
    branch_var: int | None = None
    branch_value: float | None = None
    branch_direction: str | None = None
    lp_bound: float | None = None
