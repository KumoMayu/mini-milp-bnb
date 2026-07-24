from __future__ import annotations

import numpy as np

from solver import MILPProblem

from .base import FamilyInstance, instance_stats


class FixedChargeMultiResourceFamily:
    family_name = "fixed_charge_multi_resource"

    def generate(self, seed: int, size: int, split: str, scale_group: str) -> FamilyInstance:
        rng = np.random.default_rng(seed)
        units = int(size)
        resource_count = 2 + int(units >= 6)
        capacity = rng.uniform(4.0, 9.0, units).round(3)
        variable_cost = rng.uniform(1.0, 4.0, units).round(3)
        fixed_cost = rng.uniform(1.5, 5.0, units).round(3)
        groups = min(3, max(2, units // 2))
        group_matrix = rng.uniform(0.2, 1.0, (groups, units))
        demand = (0.28 + 0.06 * rng.random(groups)) * (group_matrix @ capacity)
        resource = rng.uniform(0.2, 1.2, (resource_count, units))
        resource_limits = (0.60 + 0.10 * rng.random(resource_count)) * (resource @ capacity)

        A_rows = []
        B_rows = []
        b_values = []
        for g in range(groups):
            A_rows.append(-group_matrix[g])
            B_rows.append(np.zeros(units))
            b_values.append(-float(demand[g]))
        for k in range(resource_count):
            A_rows.append(resource[k])
            B_rows.append(np.zeros(units))
            b_values.append(float(resource_limits[k]))
        for i in range(units):
            a = np.zeros(units)
            b = np.zeros(units)
            a[i] = 1.0
            b[i] = -capacity[i]
            A_rows.append(a)
            B_rows.append(b)
            b_values.append(0.0)

        name = f"{split}_{self.family_name}_n{units}_seed{seed}"
        problem = MILPProblem.from_blocks(
            c_x=variable_cost,
            c_y=fixed_cost,
            A=np.vstack(A_rows),
            B=np.vstack(B_rows),
            b=np.asarray(b_values, dtype=float),
            x_lb=np.zeros(units),
            x_ub=capacity,
            sense="min",
            name=name,
        )
        parameters = {
            "family_name": self.family_name,
            "instance_id": name,
            "seed": int(seed),
            "size": units,
            "units": units,
            "split": split,
            "scale_group": scale_group,
            "capacity": capacity.tolist(),
            "variable_cost": variable_cost.tolist(),
            "fixed_cost": fixed_cost.tolist(),
            "group_matrix": group_matrix.tolist(),
            "demand": demand.tolist(),
            "resource": resource.tolist(),
            "resource_limits": resource_limits.tolist(),
            "stats": instance_stats(problem),
        }
        return FamilyInstance(self.family_name, name, int(seed), units, split, scale_group, problem, parameters)
