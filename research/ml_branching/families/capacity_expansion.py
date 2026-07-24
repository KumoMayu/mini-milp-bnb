from __future__ import annotations

import numpy as np

from solver import MILPProblem

from .base import FamilyInstance, instance_stats


class CapacityExpansionFamily:
    family_name = "capacity_expansion"

    def generate(self, seed: int, size: int, split: str, scale_group: str) -> FamilyInstance:
        rng = np.random.default_rng(seed)
        modules = int(size)
        regions = max(3, modules // 2 + 1)
        n_x = modules * regions
        module_cap = rng.uniform(3.5, 8.5, modules).round(3)
        fixed_cost = rng.uniform(2.0, 8.0, modules).round(3)
        op_cost = rng.uniform(0.8, 3.8, (modules, regions)).round(3)
        demand = rng.uniform(1.0, 3.2, regions).round(3)
        demand *= min(0.65 * float(np.sum(module_cap)) / float(np.sum(demand)), 1.8)
        resource = rng.uniform(0.2, 1.0, modules)
        resource_limit = 0.72 * float(resource @ module_cap)

        def x_index(m: int, r: int) -> int:
            return m * regions + r

        A_rows = []
        B_rows = []
        b_values = []
        for r in range(regions):
            a = np.zeros(n_x)
            for m in range(modules):
                a[x_index(m, r)] = -1.0
            A_rows.append(a)
            B_rows.append(np.zeros(modules))
            b_values.append(-float(demand[r]))
        for m in range(modules):
            a = np.zeros(n_x)
            for r in range(regions):
                a[x_index(m, r)] = 1.0
            b = np.zeros(modules)
            b[m] = -module_cap[m]
            A_rows.append(a)
            B_rows.append(b)
            b_values.append(0.0)
        A_rows.append(np.zeros(n_x))
        B_rows.append(resource)
        b_values.append(resource_limit)

        name = f"{split}_{self.family_name}_m{modules}_r{regions}_seed{seed}"
        problem = MILPProblem.from_blocks(
            c_x=op_cost.reshape(-1),
            c_y=fixed_cost,
            A=np.vstack(A_rows),
            B=np.vstack(B_rows),
            b=np.asarray(b_values, dtype=float),
            x_lb=np.zeros(n_x),
            x_ub=np.repeat(module_cap, regions),
            sense="min",
            name=name,
        )
        parameters = {
            "family_name": self.family_name,
            "instance_id": name,
            "seed": int(seed),
            "size": modules,
            "units": modules,
            "regions": regions,
            "split": split,
            "scale_group": scale_group,
            "module_cap": module_cap.tolist(),
            "fixed_cost": fixed_cost.tolist(),
            "op_cost": op_cost.tolist(),
            "demand": demand.tolist(),
            "resource": resource.tolist(),
            "resource_limit": float(resource_limit),
            "stats": instance_stats(problem),
        }
        return FamilyInstance(self.family_name, name, int(seed), modules, split, scale_group, problem, parameters)
