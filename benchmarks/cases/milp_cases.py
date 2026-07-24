from __future__ import annotations

import numpy as np

from solver import MILPProblem


def _case(family: str, scale: str, seed: int, problem: MILPProblem, application: str):
    from .registry import BenchmarkCase

    return BenchmarkCase(
        case_id=f"{family}_{scale}_seed_{seed}",
        family=family,
        scale=scale,
        seed=seed,
        category="milp",
        problem=problem,
        expected_status="optimal",
        metadata={
            "application": application,
            "num_variables": problem.num_vars,
            "num_integer_variables": problem.num_integer,
            "num_constraints": problem.num_constraints,
            "density": float(np.count_nonzero(problem.G) / max(1, problem.G.size)),
        },
    )


def build_knapsack(scale: str, seed: int):
    rng = np.random.default_rng(seed)
    n = 14 if scale == "small" else 30
    weights = rng.integers(2, 16, size=n).astype(float)
    values = rng.integers(4, 30, size=n).astype(float)
    capacity = 0.42 * float(weights.sum())
    problem = MILPProblem.from_standard(
        c=values,
        G=weights.reshape(1, -1),
        h=[capacity],
        sense="max",
        lb=np.zeros(n),
        ub=np.ones(n),
        var_types=["B"] * n,
        name=f"knapsack_{scale}_{seed}",
    )
    return _case("knapsack", scale, seed, problem, "binary item selection under capacity")


def build_set_cover(scale: str, seed: int):
    rng = np.random.default_rng(seed)
    elements, sets = ((10, 16) if scale == "small" else (22, 36))
    cover = rng.random((elements, sets)) < 0.28
    for row in range(elements):
        if not cover[row].any():
            cover[row, int(rng.integers(0, sets))] = True
    costs = rng.uniform(1.0, 10.0, size=sets).round(3)
    problem = MILPProblem.from_standard(
        c=costs,
        G=-cover.astype(float),
        h=-np.ones(elements),
        sense="min",
        lb=np.zeros(sets),
        ub=np.ones(sets),
        var_types=["B"] * sets,
        name=f"set_cover_{scale}_{seed}",
    )
    return _case("set_cover", scale, seed, problem, "minimum-cost set cover")


def build_facility_location(scale: str, seed: int):
    rng = np.random.default_rng(seed)
    facilities, customers = ((8, 10) if scale == "small" else (13, 18))
    assignment_count = facilities * customers
    fixed_cost = rng.uniform(8.0, 24.0, size=facilities)
    assignment_cost = rng.uniform(0.5, 7.0, size=(customers, facilities))
    c = np.concatenate([assignment_cost.ravel(), fixed_cost])
    rows = []
    rhs = []
    for customer in range(customers):
        row = np.zeros(assignment_count + facilities)
        start = customer * facilities
        row[start : start + facilities] = -1.0
        rows.append(row)
        rhs.append(-1.0)
    for customer in range(customers):
        for facility in range(facilities):
            row = np.zeros(assignment_count + facilities)
            row[customer * facilities + facility] = 1.0
            row[assignment_count + facility] = -1.0
            rows.append(row)
            rhs.append(0.0)
    problem = MILPProblem.from_standard(
        c=c,
        G=np.vstack(rows),
        h=np.asarray(rhs),
        sense="min",
        lb=np.zeros_like(c),
        ub=np.ones_like(c),
        var_types=["C"] * assignment_count + ["B"] * facilities,
        name=f"facility_location_{scale}_{seed}",
    )
    return _case(
        "facility_location",
        scale,
        seed,
        problem,
        "facility opening and continuous customer assignment",
    )


def build_unit_commitment(scale: str, seed: int):
    rng = np.random.default_rng(seed)
    units = 12 if scale == "small" else 28
    capacity = rng.uniform(3.0, 9.0, size=units).round(3)
    minimum = (capacity * rng.uniform(0.05, 0.25, size=units)).round(3)
    variable_cost = rng.uniform(1.0, 5.0, size=units).round(3)
    fixed_cost = rng.uniform(2.0, 12.0, size=units).round(3)
    demand = 0.58 * float(capacity.sum())
    rows = [-np.ones(2 * units)]
    rows[0][units:] = 0.0
    rhs = [-demand]
    for index in range(units):
        upper = np.zeros(2 * units)
        upper[index] = 1.0
        upper[units + index] = -capacity[index]
        rows.append(upper)
        rhs.append(0.0)
        lower = np.zeros(2 * units)
        lower[index] = -1.0
        lower[units + index] = minimum[index]
        rows.append(lower)
        rhs.append(0.0)
    problem = MILPProblem.from_standard(
        c=np.concatenate([variable_cost, fixed_cost]),
        G=np.vstack(rows),
        h=np.asarray(rhs),
        sense="min",
        lb=np.zeros(2 * units),
        ub=np.concatenate([capacity, np.ones(units)]),
        var_types=["C"] * units + ["B"] * units,
        name=f"unit_commitment_{scale}_{seed}",
    )
    return _case(
        "unit_commitment",
        scale,
        seed,
        problem,
        "single-period generation dispatch and commitment",
    )


def build_lot_sizing(scale: str, seed: int):
    rng = np.random.default_rng(seed)
    periods = 10 if scale == "small" else 24
    demand = rng.uniform(2.0, 7.0, size=periods).round(3)
    capacity = np.maximum(demand * 1.5, rng.uniform(7.0, 12.0, size=periods)).round(3)
    production_cost = rng.uniform(1.0, 4.0, size=periods).round(3)
    holding_cost = rng.uniform(0.1, 0.8, size=periods).round(3)
    setup_cost = rng.uniform(2.0, 9.0, size=periods).round(3)
    n = 3 * periods
    rows = []
    rhs = []
    for period in range(periods):
        balance = np.zeros(n)
        balance[period] = 1.0
        balance[periods + period] = -1.0
        if period > 0:
            balance[periods + period - 1] = 1.0
        rows.extend([balance, -balance])
        rhs.extend([demand[period], -demand[period]])
        linking = np.zeros(n)
        linking[period] = 1.0
        linking[2 * periods + period] = -capacity[period]
        rows.append(linking)
        rhs.append(0.0)
    problem = MILPProblem.from_standard(
        c=np.concatenate([production_cost, holding_cost, setup_cost]),
        G=np.vstack(rows),
        h=np.asarray(rhs),
        sense="min",
        lb=np.zeros(n),
        ub=np.concatenate(
            [
                capacity,
                np.full(periods, float(demand.sum())),
                np.ones(periods),
            ]
        ),
        var_types=["C"] * (2 * periods) + ["B"] * periods,
        name=f"lot_sizing_{scale}_{seed}",
    )
    return _case(
        "lot_sizing",
        scale,
        seed,
        problem,
        "multi-period production, inventory, and setup decisions",
    )


MILP_BUILDERS = {
    "knapsack": build_knapsack,
    "set_cover": build_set_cover,
    "facility_location": build_facility_location,
    "unit_commitment": build_unit_commitment,
    "lot_sizing": build_lot_sizing,
}
