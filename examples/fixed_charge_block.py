from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from solver import MILPProblem, solve_milp


def build_problem() -> MILPProblem:
    variable_cost=np.array([2.0,3.0,1.5])
    fixed_cost=np.array([5.0,4.0,9.0])
    capacity=np.array([4.0,5.0,7.0])
    demand=9.0

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
        c_x=variable_cost,
        c_y=fixed_cost,
        A=A,
        B=B,
        b=b,
        x_lb=np.zeros(3),
        x_ub=capacity,
        y_lb=np.zeros(3),
        y_ub=np.ones(3),
        y_types=["B","B","B"],
        sense="min",
        name="fixed_charge_block",
    )


def main() -> None:
    problem=build_problem()
    result=solve_milp(problem)

    print("Example: fixed_charge_block")
    print("Block MILP: continuous production x, binary activation y")
    print(result.simple_summary())
    print(f"x production: {result.x_continuous}")
    print(f"y activation: {result.y_integer}")


if __name__=="__main__":
    main()
