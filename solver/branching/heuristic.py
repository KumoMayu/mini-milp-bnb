from __future__ import annotations

from dataclasses import dataclass

from .base import BranchingContext
from .pseudocost import PseudocostPolicy


def _fractional_distance(value: float) -> float:
    return abs(float(value) - round(float(value)))


def _validate_choice(choice: int, context: BranchingContext) -> int:
    if choice not in context.candidate_indices:
        raise ValueError("branching policy returned an index outside candidate_indices")
    return int(choice)


@dataclass(frozen=True)
class MostFractionalPolicy:
    """Match the historical most_fractional rule.

    The previous implementation used max((distance, index)), so equal
    fractional distance is broken by the larger global variable index.
    """

    def select_variable(self, context: BranchingContext) -> int:
        if not context.candidate_indices:
            raise ValueError("no fractional binary branching candidates")
        x = context.lp_result.x
        choice = max(
            ((
                _fractional_distance(float(x[index])),
                int(index),
            ) for index in context.candidate_indices)
        )[1]
        return _validate_choice(choice, context)


@dataclass(frozen=True)
class FirstFractionalPolicy:
    def select_variable(self, context: BranchingContext) -> int:
        if not context.candidate_indices:
            raise ValueError("no fractional binary branching candidates")
        return _validate_choice(min(int(index) for index in context.candidate_indices), context)


def policy_from_rule(branching_rule: str):
    rule = str(branching_rule)
    if rule == "most_fractional":
        return MostFractionalPolicy()
    if rule == "first_fractional":
        return FirstFractionalPolicy()
    if rule == "pseudocost":
        return PseudocostPolicy()
    raise ValueError('branching_rule must be "most_fractional", "first_fractional", or "pseudocost"')
