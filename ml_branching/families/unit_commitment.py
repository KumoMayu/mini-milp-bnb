from __future__ import annotations

import numpy as np

from solver import MILPProblem

from .base import FamilyInstance, instance_stats


class UnitCommitmentFamily:
    family_name = "unit_commitment"

    def generate(self, seed: int, size: int, split: str, scale_group: str) -> FamilyInstance:
        rng = np.random.default_rng(seed)
        units = int(size)
        periods = 3 + int(units >= 5)
        n_x = units * periods
        n_y = units * periods
        p_min = rng.uniform(0.5, 1.5, units).round(3)
        p_max = rng.uniform(3.5, 7.5, units).round(3)
        ramp = rng.uniform(1.5, 4.0, units).round(3)
        variable_cost = rng.uniform(1.0, 4.0, units).round(3)
        fixed_cost = rng.uniform(1.0, 5.0, units).round(3)
        demand = rng.uniform(0.40, 0.58, periods) * float(np.sum(p_max))
        reserve = rng.uniform(0.05, 0.10, periods) * float(np.sum(p_max))

        def idx(i: int, t: int) -> int:
            return i * periods + t

        A_rows = []
        B_rows = []
        b_values = []
        for t in range(periods):
            a = np.zeros(n_x)
            for i in range(units):
                a[idx(i, t)] = -1.0
            A_rows.append(a)
            B_rows.append(np.zeros(n_y))
            b_values.append(-float(demand[t]))
            b = np.zeros(n_y)
            for i in range(units):
                b[idx(i, t)] = -p_max[i]
            A_rows.append(np.zeros(n_x))
            B_rows.append(b)
            b_values.append(-float(demand[t] + reserve[t]))
        for i in range(units):
            for t in range(periods):
                a = np.zeros(n_x)
                b = np.zeros(n_y)
                a[idx(i, t)] = 1.0
                b[idx(i, t)] = -p_max[i]
                A_rows.append(a)
                B_rows.append(b)
                b_values.append(0.0)
                a2 = np.zeros(n_x)
                b2 = np.zeros(n_y)
                a2[idx(i, t)] = -1.0
                b2[idx(i, t)] = p_min[i]
                A_rows.append(a2)
                B_rows.append(b2)
                b_values.append(0.0)
            for t in range(1, periods):
                a = np.zeros(n_x)
                a[idx(i, t)] = 1.0
                a[idx(i, t - 1)] = -1.0
                A_rows.append(a)
                B_rows.append(np.zeros(n_y))
                b_values.append(float(ramp[i]))
                A_rows.append(-a)
                B_rows.append(np.zeros(n_y))
                b_values.append(float(ramp[i]))

        name = f"{split}_{self.family_name}_u{units}_t{periods}_seed{seed}"
        problem = MILPProblem.from_blocks(
            c_x=np.tile(variable_cost, periods),
            c_y=np.tile(fixed_cost, periods),
            A=np.vstack(A_rows),
            B=np.vstack(B_rows),
            b=np.asarray(b_values, dtype=float),
            x_lb=np.zeros(n_x),
            x_ub=np.repeat(p_max, periods),
            sense="min",
            name=name,
        )
        parameters = {
            "family_name": self.family_name,
            "instance_id": name,
            "seed": int(seed),
            "size": units,
            "units": units,
            "periods": periods,
            "split": split,
            "scale_group": scale_group,
            "p_min": p_min.tolist(),
            "p_max": p_max.tolist(),
            "ramp": ramp.tolist(),
            "variable_cost": variable_cost.tolist(),
            "fixed_cost": fixed_cost.tolist(),
            "demand": demand.tolist(),
            "reserve": reserve.tolist(),
            "stats": instance_stats(problem),
        }
        return FamilyInstance(self.family_name, name, int(seed), units, split, scale_group, problem, parameters)
