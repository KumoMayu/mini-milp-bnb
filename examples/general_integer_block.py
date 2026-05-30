from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from solver import MILPProblem, solve_milp


def build_problem() -> MILPProblem:
    # x: continuous overtime resource.
    # y: two general integer procurement/shift decisions.
    c_x=np.array([4.0])
    c_y=np.array([3.0,5.0])

    A=np.array(
        [
            [-1],
            [-2],
        ],
        dtype=float,
    )
    B=np.array(
        [
            [-2,-1],
            [-1,-3],
        ],
        dtype=float,
    )
    b=np.array([-7,-9],dtype=float)

    return MILPProblem.from_blocks(
        c_x=c_x,
        c_y=c_y,
        A=A,
        B=B,
        b=b,
        x_lb=np.array([0.0]),
        x_ub=np.array([3.0]),
        y_lb=np.array([0.0,0.0]),
        y_ub=np.array([5.0,4.0]),
        y_types=["I","I"],
        sense="min",
        name="general_integer_block",
    )


def main() -> None:
    problem=build_problem()
    result=solve_milp(problem)

    print("Example: general_integer_block")
    print("Block MILP: continuous resource x, general integer decisions y")
    print(result.simple_summary())
    print(f"x continuous: {result.x_continuous}")
    print(f"y integer: {result.y_integer}")


if __name__=="__main__":
    main()
