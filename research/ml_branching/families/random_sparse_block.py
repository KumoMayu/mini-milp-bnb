from __future__ import annotations

import numpy as np

from solver import MILPProblem

from .base import FamilyInstance, instance_stats


class RandomSparseBlockFamily:
    family_name = "random_sparse_block"

    def generate(self, seed: int, size: int, split: str, scale_group: str) -> FamilyInstance:
        rng = np.random.default_rng(seed)
        n_y = int(size)
        n_x = max(4, n_y + 2)
        rows = max(8, 2 * n_y + 4)
        x_ub = rng.uniform(2.0, 6.0, n_x).round(3)
        x_ref = rng.uniform(0.2, 0.65, n_x) * x_ub
        y_ref = rng.integers(0, 2, n_y).astype(float)
        if np.sum(y_ref) < max(2, n_y // 3):
            y_ref[: max(2, n_y // 3)] = 1.0
        A = np.zeros((rows, n_x))
        B = np.zeros((rows, n_y))
        for r in range(rows):
            x_cols = rng.choice(n_x, size=max(2, min(n_x, 2 + r % 4)), replace=False)
            y_cols = rng.choice(n_y, size=max(2, min(n_y, 2 + (r + 1) % 3)), replace=False)
            A[r, x_cols] = rng.uniform(-0.8, 1.4, len(x_cols))
            B[r, y_cols] = rng.uniform(-1.2, 1.2, len(y_cols))
        for j in range(min(n_y, n_x)):
            row = rows - 1 - (j % min(rows, n_y))
            A[row, j % n_x] = 1.0
            B[row, j] = -x_ub[j % n_x]
        slack = rng.uniform(0.5, 2.0, rows)
        b = A @ x_ref + B @ y_ref + slack
        demand_rows = min(3, rows)
        for r in range(demand_rows):
            A[r, :] = -rng.uniform(0.0, 1.0, n_x) * (rng.random(n_x) < 0.55)
            if not np.any(A[r, :]):
                A[r, r % n_x] = -1.0
            B[r, :] = 0.0
            b[r] = 0.45 * float(A[r, :] @ x_ref)

        name = f"{split}_{self.family_name}_x{n_x}_y{n_y}_seed{seed}"
        problem = MILPProblem.from_blocks(
            c_x=rng.uniform(0.5, 3.0, n_x),
            c_y=rng.uniform(0.5, 4.0, n_y),
            A=A,
            B=B,
            b=b,
            x_lb=np.zeros(n_x),
            x_ub=x_ub,
            sense="min",
            name=name,
        )
        parameters = {
            "family_name": self.family_name,
            "instance_id": name,
            "seed": int(seed),
            "size": n_y,
            "units": n_y,
            "split": split,
            "scale_group": scale_group,
            "n_x": n_x,
            "n_y": n_y,
            "rows": rows,
            "x_ub": x_ub.tolist(),
            "density": float((np.count_nonzero(A) + np.count_nonzero(B)) / (rows * (n_x + n_y))),
            "stats": instance_stats(problem),
        }
        return FamilyInstance(self.family_name, name, int(seed), n_y, split, scale_group, problem, parameters)
