from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from solver import MILPProblem, solve_milp


def build_problem() -> MILPProblem:
    generation_cost=np.array([2.0,3.0])
    fixed_cost=np.array([8.0,3.0])
    p_min=np.array([2.0,1.0])
    p_max=np.array([6.0,5.0])
    demand=7.0

    A=np.array(
        [
            [-1,-1],
            [-1,0],
            [0,-1],
            [1,0],
            [0,1],
        ],
        dtype=float,
    )
    B=np.array(
        [
            [0,0],
            [p_min[0],0],
            [0,p_min[1]],
            [-p_max[0],0],
            [0,-p_max[1]],
        ],
        dtype=float,
    )
    b=np.array([-demand,0,0,0,0],dtype=float)

    return MILPProblem.from_blocks(
        c_x=generation_cost,
        c_y=fixed_cost,
        A=A,
        B=B,
        b=b,
        x_lb=np.zeros(2),
        x_ub=p_max,
        sense="min",
        name="unit_commitment_tiny",
    )


def main() -> None:
    problem=build_problem()
    result=solve_milp(problem)

    print("Example: unit_commitment_tiny")
    print("Tiny one-period unit commitment: dispatch x, commitment y")
    print(result.simple_summary())
    print(f"x dispatch: {result.x_continuous}")
    print(f"y commitment: {result.y_integer}")


if __name__=="__main__":
    main()
