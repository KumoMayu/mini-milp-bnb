from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks.config import seeds_for_scale

from .lp_cases import LP_BUILDERS
from .milp_cases import MILP_BUILDERS


@dataclass(frozen=True)
class GeneralLPProblem:
    name: str
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    constraint_senses: tuple[str, ...]
    lb: np.ndarray
    ub: np.ndarray
    sense: str


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    family: str
    scale: str
    seed: int
    category: str
    problem: Any
    metadata: dict[str, Any]
    expected_status: str | None = None


def available_families(category: str | None = None) -> tuple[str, ...]:
    if category == "lp":
        return tuple(LP_BUILDERS)
    if category == "milp":
        return tuple(MILP_BUILDERS)
    if category is not None:
        raise ValueError('category must be "lp", "milp", or None')
    return tuple(LP_BUILDERS) + tuple(MILP_BUILDERS)


def build_case(family: str, scale: str, seed: int) -> BenchmarkCase:
    if scale not in {"small", "large"}:
        raise ValueError('scale must be "small" or "large"')
    builders = {**LP_BUILDERS, **MILP_BUILDERS}
    try:
        builder = builders[family]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark family: {family}") from exc
    return builder(scale, int(seed))


def iter_cases(scale: str, family: str | None = None):
    families = available_families() if family is None else (family,)
    for family_name in families:
        for seed in seeds_for_scale(scale):
            yield build_case(family_name, scale, seed)
