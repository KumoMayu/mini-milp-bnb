from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from solver import MILPProblem, solve_milp


def build_problem() -> MILPProblem:
    operating_cost=np.array([4.0,2.5,3.5])
    fixed_cost=np.array([3.0,8.0,5.0])
    capacity=np.array([3.0,5.0,4.0])
    demand=7.0

    A=np.array(
        [
            [-1,-1,-1],
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ],
        dtype=float,
    )
    B=np.array(
        [
            [0,0,0],
            [-capacity[0],0,0],
            [0,-capacity[1],0],
            [0,0,-capacity[2]],
        ],
        dtype=float,
    )
    b=np.array([-demand,0,0,0],dtype=float)

    return MILPProblem.from_blocks(
        c_x=operating_cost,
        c_y=fixed_cost,
        A=A,
        B=B,
        b=b,
        x_lb=np.zeros(3),
        x_ub=capacity,
        sense="min",
        name="production_expansion_binary",
    )


def main() -> None:
    problem=build_problem()
    result=solve_milp(problem)

    print("Example: production_expansion_binary")
    print("Binary capacity expansion: continuous supply x, binary build decisions y")
    print(result.simple_summary())
    print(f"x supply: {result.x_continuous}")
    print(f"y build: {result.y_integer}")


if __name__=="__main__":
    main()
