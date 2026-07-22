from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver import TableauSimplexSolver


def main() -> None:
    c = np.array([3.0, 2.0])
    A = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    b = np.array([4.0, 5.0])

    solver = TableauSimplexSolver(verbose=True)
    result = solver.solve(c, A, b)

    print(f"solution={result.x}")
    print(f"objective={result.objective_value}")


if __name__ == "__main__":
    main()
