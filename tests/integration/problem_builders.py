from __future__ import annotations

import numpy as np

from solver import MILPProblem


def build_production_expansion() -> MILPProblem:
    operating_cost = np.array([4.0, 2.5, 3.5])
    fixed_cost = np.array([3.0, 8.0, 5.0])
    capacity = np.array([3.0, 5.0, 4.0])

    A = np.array(
        [
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0, 0, 0],
            [-capacity[0], 0, 0],
            [0, -capacity[1], 0],
            [0, 0, -capacity[2]],
        ],
        dtype=float,
    )
    return MILPProblem.from_blocks(
        c_x=operating_cost,
        c_y=fixed_cost,
        A=A,
        B=B,
        b=np.array([-7.0, 0, 0, 0]),
        x_lb=np.zeros(3),
        x_ub=capacity,
        sense="min",
        name="production_expansion_test",
    )
