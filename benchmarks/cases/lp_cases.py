from __future__ import annotations

import numpy as np


def _feasible_random_lp(
    *,
    family: str,
    scale: str,
    seed: int,
    num_variables: int,
    num_constraints: int,
    density: float,
):
    from .registry import BenchmarkCase, GeneralLPProblem

    rng = np.random.default_rng(seed)
    lb = np.zeros(num_variables)
    ub = rng.uniform(2.0, 6.0, size=num_variables)
    feasible = rng.uniform(0.1 * ub, 0.65 * ub)
    mask = rng.random((num_constraints, num_variables)) < density
    empty_rows = np.flatnonzero(~mask.any(axis=1))
    if empty_rows.size:
        mask[empty_rows, rng.integers(0, num_variables, size=empty_rows.size)] = True
    A = rng.uniform(0.1, 1.5, size=(num_constraints, num_variables)) * mask
    slack = rng.uniform(0.1, 0.8, size=num_constraints)
    b = A @ feasible + slack
    c = rng.uniform(0.2, 2.0, size=num_variables)
    problem = GeneralLPProblem(
        name=f"{family}_{scale}_seed_{seed}",
        c=c,
        A=A,
        b=b,
        constraint_senses=("<=",) * num_constraints,
        lb=lb,
        ub=ub,
        sense="max",
    )
    return BenchmarkCase(
        case_id=problem.name,
        family=family,
        scale=scale,
        seed=seed,
        category="lp",
        problem=problem,
        expected_status="optimal",
        metadata={
            "application": "bounded reproducible random LP",
            "num_variables": num_variables,
            "num_integer_variables": 0,
            "num_constraints": num_constraints,
            "density": float(np.count_nonzero(A) / A.size),
        },
    )


def build_general_lp(scale: str, seed: int):
    if scale == "small":
        return _feasible_random_lp(
            family="general_lp",
            scale=scale,
            seed=seed,
            num_variables=40,
            num_constraints=80,
            density=0.15,
        )
    return _feasible_random_lp(
        family="general_lp",
        scale=scale,
        seed=seed,
        num_variables=110,
        num_constraints=220,
        density=0.08,
    )


def build_dense_lp(scale: str, seed: int):
    if scale == "small":
        return _feasible_random_lp(
            family="dense_lp",
            scale=scale,
            seed=seed,
            num_variables=40,
            num_constraints=85,
            density=0.82,
        )
    return _feasible_random_lp(
        family="dense_lp",
        scale=scale,
        seed=seed,
        num_variables=100,
        num_constraints=200,
        density=0.90,
    )


def build_numerical_lp(scale: str, seed: int):
    from .registry import BenchmarkCase, GeneralLPProblem

    rng = np.random.default_rng(seed)
    mode = ("optimal", "infeasible", "unbounded")[seed % 3]
    n = 14 if scale == "small" else 45

    if mode == "infeasible":
        lb = np.zeros(n)
        ub = np.ones(n)
        A = np.zeros((2, n))
        A[:, 0] = 1.0
        problem = GeneralLPProblem(
            name=f"numerical_lp_{scale}_seed_{seed}",
            c=np.ones(n),
            A=A,
            b=np.array([0.0, 1.0]),
            constraint_senses=("<=", ">="),
            lb=lb,
            ub=ub,
            sense="max",
        )
        expected_status = "infeasible"
        structure = "contradictory mixed-sense rows"
    elif mode == "unbounded":
        lb = np.zeros(n)
        ub = np.full(n, np.inf)
        A = np.zeros((1, n))
        A[0, 1] = 1.0
        c = np.zeros(n)
        c[0] = 1.0
        problem = GeneralLPProblem(
            name=f"numerical_lp_{scale}_seed_{seed}",
            c=c,
            A=A,
            b=np.array([1.0]),
            constraint_senses=("<=",),
            lb=lb,
            ub=ub,
            sense="max",
        )
        expected_status = "unbounded"
        structure = "finite lower bounds with an unbounded objective ray"
    else:
        lb = rng.uniform(-1.0, 0.0, size=n)
        ub = lb + rng.uniform(1.0, 3.0, size=n)
        feasible = rng.uniform(lb, ub)
        lb[-1] = feasible[-1]
        ub[-1] = feasible[-1]
        base = rng.normal(size=(max(8, 2 * n), n))
        scale_factors = np.geomspace(
            1e-2 if scale == "small" else 1e-4,
            1e2 if scale == "small" else 1e4,
            num=base.shape[0],
        )
        base = base * scale_factors[:, None]
        rhs = base @ feasible + np.maximum(1e-6, np.abs(scale_factors) * 0.1)
        equality = rng.normal(size=n)
        greater = rng.normal(size=n)
        duplicate = base[0].copy()
        perturbation = 1e-5 if scale == "small" else 1e-9
        near_parallel = duplicate * (1.0 + perturbation)
        A = np.vstack([base, equality, greater, duplicate, near_parallel])
        b = np.concatenate(
            [
                rhs,
                [equality @ feasible],
                [greater @ feasible - 0.25],
                [rhs[0]],
                [rhs[0] * (1.0 + perturbation) + perturbation],
            ]
        )
        senses = (
            ("<=",) * base.shape[0]
            + ("=", ">=", "<=", "<=")
        )
        problem = GeneralLPProblem(
            name=f"numerical_lp_{scale}_seed_{seed}",
            c=rng.normal(size=n),
            A=A,
            b=b,
            constraint_senses=senses,
            lb=lb,
            ub=ub,
            sense="min" if seed % 2 else "max",
        )
        expected_status = "optimal"
        structure = "mixed scales, duplicate rows, degeneracy, equality, >=, fixed variable"

    return BenchmarkCase(
        case_id=problem.name,
        family="numerical_lp",
        scale=scale,
        seed=seed,
        category="lp",
        problem=problem,
        expected_status=expected_status,
        metadata={
            "application": structure,
            "num_variables": problem.c.size,
            "num_integer_variables": 0,
            "num_constraints": problem.b.size,
            "density": float(np.count_nonzero(problem.A) / max(1, problem.A.size)),
        },
    )


LP_BUILDERS = {
    "general_lp": build_general_lp,
    "dense_lp": build_dense_lp,
    "numerical_lp": build_numerical_lp,
}
