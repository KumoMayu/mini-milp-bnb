from __future__ import annotations

import numpy as np

from solver import MILPProblem

from .base import FamilyInstance, instance_stats


class ActivatedResourceAllocationFamily:
    family_name = "activated_resource_allocation"

    def generate(self, seed: int, size: int, split: str, scale_group: str) -> FamilyInstance:
        rng = np.random.default_rng(seed)
        projects = int(size)
        resource_count = 3
        need_count = 2 + int(projects >= 6)
        max_alloc = rng.uniform(3.0, 8.0, projects).round(3)
        variable_cost = rng.uniform(0.8, 3.0, projects).round(3)
        fixed_cost = rng.uniform(1.0, 5.0, projects).round(3)
        resource_use = rng.uniform(0.2, 1.3, (resource_count, projects))
        value = rng.uniform(0.3, 1.4, (need_count, projects))
        budget = (0.55 + 0.12 * rng.random(resource_count)) * (resource_use @ max_alloc)
        need = (0.25 + 0.08 * rng.random(need_count)) * (value @ max_alloc)
        max_active = max(2, int(np.ceil(0.75 * projects)))

        A_rows = []
        B_rows = []
        b_values = []
        for g in range(need_count):
            A_rows.append(-value[g])
            B_rows.append(np.zeros(projects))
            b_values.append(-float(need[g]))
        for k in range(resource_count):
            A_rows.append(resource_use[k])
            B_rows.append(np.zeros(projects))
            b_values.append(float(budget[k]))
        for i in range(projects):
            a = np.zeros(projects)
            b = np.zeros(projects)
            a[i] = 1.0
            b[i] = -max_alloc[i]
            A_rows.append(a)
            B_rows.append(b)
            b_values.append(0.0)
        A_rows.append(np.zeros(projects))
        B_rows.append(np.ones(projects))
        b_values.append(float(max_active))

        name = f"{split}_{self.family_name}_p{projects}_seed{seed}"
        problem = MILPProblem.from_blocks(
            c_x=variable_cost,
            c_y=fixed_cost,
            A=np.vstack(A_rows),
            B=np.vstack(B_rows),
            b=np.asarray(b_values, dtype=float),
            x_lb=np.zeros(projects),
            x_ub=max_alloc,
            sense="min",
            name=name,
        )
        parameters = {
            "family_name": self.family_name,
            "instance_id": name,
            "seed": int(seed),
            "size": projects,
            "units": projects,
            "split": split,
            "scale_group": scale_group,
            "max_alloc": max_alloc.tolist(),
            "variable_cost": variable_cost.tolist(),
            "fixed_cost": fixed_cost.tolist(),
            "resource_use": resource_use.tolist(),
            "budget": budget.tolist(),
            "value": value.tolist(),
            "need": need.tolist(),
            "max_active": int(max_active),
            "stats": instance_stats(problem),
        }
        return FamilyInstance(self.family_name, name, int(seed), projects, split, scale_group, problem, parameters)
