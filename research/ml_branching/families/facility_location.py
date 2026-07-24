from __future__ import annotations

import numpy as np

from solver import MILPProblem

from .base import FamilyInstance, instance_stats


class FacilityLocationFamily:
    family_name = "facility_location"

    def generate(self, seed: int, size: int, split: str, scale_group: str) -> FamilyInstance:
        rng = np.random.default_rng(seed)
        facilities = int(size)
        customers = max(3, facilities - 1)
        n_x = facilities * customers
        capacity = rng.uniform(5.0, 11.0, facilities).round(3)
        demand = rng.uniform(1.5, 4.0, customers).round(3)
        demand *= min(0.75 * float(np.sum(capacity)) / float(np.sum(demand)), 1.6)
        fixed_cost = rng.uniform(2.0, 7.0, facilities).round(3)
        ship_cost = rng.uniform(0.5, 4.5, (facilities, customers)).round(3)

        def x_index(i: int, j: int) -> int:
            return i * customers + j

        A_rows = []
        B_rows = []
        b_values = []
        for j in range(customers):
            a = np.zeros(n_x)
            for i in range(facilities):
                a[x_index(i, j)] = -1.0
            A_rows.append(a)
            B_rows.append(np.zeros(facilities))
            b_values.append(-float(demand[j]))
        for i in range(facilities):
            a = np.zeros(n_x)
            for j in range(customers):
                a[x_index(i, j)] = 1.0
            b = np.zeros(facilities)
            b[i] = -capacity[i]
            A_rows.append(a)
            B_rows.append(b)
            b_values.append(0.0)
        for i in range(facilities):
            for j in range(customers):
                a = np.zeros(n_x)
                a[x_index(i, j)] = 1.0
                b = np.zeros(facilities)
                b[i] = -min(capacity[i], demand[j])
                A_rows.append(a)
                B_rows.append(b)
                b_values.append(0.0)

        name = f"{split}_{self.family_name}_f{facilities}_c{customers}_seed{seed}"
        problem = MILPProblem.from_blocks(
            c_x=ship_cost.reshape(-1),
            c_y=fixed_cost,
            A=np.vstack(A_rows),
            B=np.vstack(B_rows),
            b=np.asarray(b_values, dtype=float),
            x_lb=np.zeros(n_x),
            x_ub=np.repeat(capacity, customers),
            sense="min",
            name=name,
        )
        parameters = {
            "family_name": self.family_name,
            "instance_id": name,
            "seed": int(seed),
            "size": facilities,
            "units": facilities,
            "customers": customers,
            "split": split,
            "scale_group": scale_group,
            "capacity": capacity.tolist(),
            "demand": demand.tolist(),
            "fixed_cost": fixed_cost.tolist(),
            "ship_cost": ship_cost.tolist(),
            "stats": instance_stats(problem),
        }
        return FamilyInstance(self.family_name, name, int(seed), facilities, split, scale_group, problem, parameters)
