from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_branching.families.base import FamilyInstance, instance_stats
from solver import MILPProblem


@dataclass(frozen=True)
class UnitCommitmentSpec:
    units: int
    periods: int
    demand_low: float = 0.42
    demand_high: float = 0.62
    reserve_low: float = 0.05
    reserve_high: float = 0.12


def build_unit_commitment_problem(parameters: dict, name: str | None = None) -> MILPProblem:
    units = int(parameters["units"])
    periods = int(parameters["periods"])
    p_min = np.asarray(parameters["p_min"], dtype=float)
    p_max = np.asarray(parameters["p_max"], dtype=float)
    ramp = np.asarray(parameters["ramp"], dtype=float)
    variable_cost = np.asarray(parameters["variable_cost"], dtype=float)
    fixed_cost = np.asarray(parameters["fixed_cost"], dtype=float)
    demand = np.asarray(parameters["demand"], dtype=float)
    reserve = np.asarray(parameters["reserve"], dtype=float)
    has_initial_output = "initial_output" in parameters
    initial_output = np.asarray(parameters.get("initial_output", np.zeros(units)), dtype=float)
    n_x = units * periods
    n_y = units * periods

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

        if has_initial_output:
            a0 = np.zeros(n_x)
            a0[idx(i, 0)] = 1.0
            A_rows.append(a0)
            B_rows.append(np.zeros(n_y))
            b_values.append(float(initial_output[i] + ramp[i]))
            A_rows.append(-a0)
            B_rows.append(np.zeros(n_y))
            b_values.append(float(ramp[i] - initial_output[i]))

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

    return MILPProblem.from_blocks(
        c_x=np.tile(variable_cost, periods),
        c_y=np.tile(fixed_cost, periods),
        A=np.vstack(A_rows),
        B=np.vstack(B_rows),
        b=np.asarray(b_values, dtype=float),
        x_lb=np.zeros(n_x),
        x_ub=np.repeat(p_max, periods),
        sense="min",
        name=name or str(parameters.get("instance_id", "unit_commitment")),
    )


def instance_from_parameters(parameters: dict) -> FamilyInstance:
    """Rebuild one saved unit commitment instance from explicit parameters."""
    problem = build_unit_commitment_problem(parameters, name=str(parameters["instance_id"]))
    params = dict(parameters)
    params["stats"] = instance_stats(problem)
    return FamilyInstance(
        family_name="unit_commitment",
        instance_id=str(params["instance_id"]),
        seed=int(params["seed"]),
        size=int(params["units"]),
        split=str(params.get("split", "reconstructed")),
        scale_group=str(params.get("scale_group", "reconstructed")),
        problem=problem,
        parameters=params,
    )


class UnitCommitmentGenerator:
    family_name = "unit_commitment"

    def generate(
        self,
        seed: int,
        units: int,
        periods: int,
        split: str,
        scale_group: str,
        demand_low: float = 0.42,
        demand_high: float = 0.62,
        reserve_low: float = 0.05,
        reserve_high: float = 0.12,
    ) -> FamilyInstance:
        rng = np.random.default_rng(int(seed))
        units = int(units)
        periods = int(periods)
        p_min = rng.uniform(0.4, 1.6, units).round(3)
        p_max = rng.uniform(3.5, 8.0, units).round(3)
        ramp = rng.uniform(1.2, 4.2, units).round(3)
        variable_cost = rng.uniform(0.8, 4.2, units).round(3)
        fixed_cost = rng.uniform(0.8, 6.0, units).round(3)
        base = rng.uniform(demand_low, demand_high, periods)
        demand = base * float(np.sum(p_max))
        reserve = rng.uniform(reserve_low, reserve_high, periods) * float(np.sum(p_max))
        initial_commitment = rng.integers(0, 2, units)
        if np.sum(initial_commitment) == 0:
            initial_commitment[int(rng.integers(0, units))] = 1
        initial_output = initial_commitment * rng.uniform(0.0, 0.35, units) * p_max
        name = f"{split}_unit_commitment_u{units}_t{periods}_seed{seed}"
        parameters = {
            "family_name": self.family_name,
            "instance_id": name,
            "seed": int(seed),
            "size": int(units),
            "units": int(units),
            "periods": int(periods),
            "split": split,
            "scale_group": scale_group,
            "p_min": p_min.tolist(),
            "p_max": p_max.tolist(),
            "ramp": ramp.tolist(),
            "variable_cost": variable_cost.tolist(),
            "fixed_cost": fixed_cost.tolist(),
            "demand": demand.tolist(),
            "reserve": reserve.tolist(),
            "initial_commitment": initial_commitment.astype(int).tolist(),
            "initial_output": initial_output.tolist(),
        }
        problem = build_unit_commitment_problem(parameters, name=name)
        parameters["stats"] = instance_stats(problem)
        return FamilyInstance(self.family_name, name, int(seed), units, split, scale_group, problem, parameters)


def generate_from_config(split: str, spec: dict, index: int, master_seed: int) -> FamilyInstance:
    units_values = list(spec.get("units", [4]))
    periods_values = list(spec.get("periods", [4]))
    units = int(units_values[index % len(units_values)])
    periods = int(periods_values[(index // max(1, len(units_values))) % len(periods_values)])
    seed = int(master_seed) + int(spec.get("seed_offset", 0)) + int(index)
    return UnitCommitmentGenerator().generate(
        seed=seed,
        units=units,
        periods=periods,
        split=split,
        scale_group=str(spec.get("scale_group", split)),
        demand_low=float(spec.get("demand_low", 0.42)),
        demand_high=float(spec.get("demand_high", 0.62)),
        reserve_low=float(spec.get("reserve_low", 0.05)),
        reserve_high=float(spec.get("reserve_high", 0.12)),
    )


__all__ = [
    "UnitCommitmentGenerator",
    "UnitCommitmentSpec",
    "build_unit_commitment_problem",
    "generate_from_config",
    "instance_from_parameters",
]
