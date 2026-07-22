from __future__ import annotations

from collections.abc import Callable

import numpy as np

from examples.fixed_charge_block import build_problem as build_fixed_charge
from examples.production_expansion_binary import build_problem as build_production_expansion
from examples.unit_commitment_tiny import build_problem as build_unit_commitment
from solver import MILPProblem


CaseBuilder=Callable[[],MILPProblem]


def build_scaling_problem(units: int) -> MILPProblem:
    if units<2:
        raise ValueError("units must be at least 2")

    capacity=np.array([3.0+float(i%3) for i in range(units)],dtype=float)
    variable_cost=np.array([2.0+0.35*float(i%4) for i in range(units)],dtype=float)
    fixed_cost=np.array([4.0+1.25*float(i%3) for i in range(units)],dtype=float)
    demand=0.62*float(np.sum(capacity))

    A_rows=[-np.ones(units,dtype=float)]
    B_rows=[np.zeros(units,dtype=float)]
    b_values=[-demand]

    for i in range(units):
        a=np.zeros(units,dtype=float)
        b=np.zeros(units,dtype=float)
        a[i]=1.0
        b[i]=-capacity[i]
        A_rows.append(a)
        B_rows.append(b)
        b_values.append(0.0)

    return MILPProblem.from_blocks(
        c_x=variable_cost,
        c_y=fixed_cost,
        A=np.vstack(A_rows),
        B=np.vstack(B_rows),
        b=np.array(b_values,dtype=float),
        x_lb=np.zeros(units,dtype=float),
        x_ub=capacity,
        sense="min",
        name=f"scaling_units_{units}",
    )


def build_seeded_scaling_problem(units: int,seed: int) -> MILPProblem:
    if units<2:
        raise ValueError("units must be at least 2")

    rng=np.random.default_rng(seed)
    capacity=rng.uniform(3.0,7.0,size=units).round(3)
    variable_cost=rng.uniform(1.2,4.0,size=units).round(3)
    fixed_cost=rng.uniform(2.5,8.5,size=units).round(3)
    demand_fraction=float(rng.uniform(0.55,0.78))
    demand=round(demand_fraction*float(np.sum(capacity)),3)

    A_rows=[-np.ones(units,dtype=float)]
    B_rows=[np.zeros(units,dtype=float)]
    b_values=[-demand]

    for i in range(units):
        a=np.zeros(units,dtype=float)
        b=np.zeros(units,dtype=float)
        a[i]=1.0
        b[i]=-capacity[i]
        A_rows.append(a)
        B_rows.append(b)
        b_values.append(0.0)

    return MILPProblem.from_blocks(
        c_x=variable_cost,
        c_y=fixed_cost,
        A=np.vstack(A_rows),
        B=np.vstack(B_rows),
        b=np.array(b_values,dtype=float),
        x_lb=np.zeros(units,dtype=float),
        x_ub=capacity,
        sense="min",
        name=f"batch_units_{units}_seed_{seed}",
    )


CORE_CASES: list[tuple[str,CaseBuilder]]=[
    ("fixed_charge_block",build_fixed_charge),
    ("unit_commitment_tiny",build_unit_commitment),
    ("production_expansion_binary",build_production_expansion),
]

SCALING_CASES: list[tuple[str,CaseBuilder]]=[
    (f"scaling_units_{units}",lambda units=units: build_scaling_problem(units))
    for units in (2,3,4,5)
]

ALL_CASES=CORE_CASES+SCALING_CASES

BATCH_CASES: list[tuple[str,CaseBuilder,int,int]]=[
    (
        f"batch_units_{units}_seed_{seed}",
        lambda units=units,seed=seed: build_seeded_scaling_problem(units,seed),
        units,
        seed,
    )
    for units in (2,3,4,5)
    for seed in range(5)
]
