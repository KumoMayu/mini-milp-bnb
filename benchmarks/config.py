from __future__ import annotations

from dataclasses import dataclass


SMALL_SEEDS = (0, 1, 2)
LARGE_SEEDS = (0,)
MAX_TABLEAU_BYTES = 1_500_000_000


@dataclass(frozen=True)
class BenchmarkLimits:
    wall_time_sec: float
    node_limit: int
    simplex_iteration_limit: int
    tableau_memory_limit: int = MAX_TABLEAU_BYTES


LIMITS = {
    "small": BenchmarkLimits(
        wall_time_sec=30.0,
        node_limit=800,
        simplex_iteration_limit=20_000,
    ),
    "large": BenchmarkLimits(
        wall_time_sec=60.0,
        node_limit=500,
        simplex_iteration_limit=50_000,
    ),
}


def seeds_for_scale(scale: str) -> tuple[int, ...]:
    if scale == "small":
        return SMALL_SEEDS
    if scale == "large":
        return LARGE_SEEDS
    raise ValueError('scale must be "small" or "large"')
