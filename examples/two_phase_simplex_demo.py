from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver import TwoPhaseTableauSimplexSolver


def main() -> None:
    # max 3 x1 + 2 x2
    # x1 + x2 <= 5, x1 >= 1.5, x2 = 2
    # 0.5 <= x1 <= 4, 1 <= x2 <= 4
    c = np.array([3.0, 2.0])
    A = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    b = np.array([5.0, 1.5, 2.0])

    solver = TwoPhaseTableauSimplexSolver(verbose=True)
    result = solver.solve(
        c=c,
        A=A,
        b=b,
        constraint_senses=["<=", ">=", "="],
        lb=[0.5, 1.0],
        ub=[4.0, 4.0],
        sense="max",
    )

    print(f"solution={result.x}")
    print(f"objective={result.objective_value}")


if __name__ == "__main__":
    main()
