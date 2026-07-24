from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from solver import MILPProblem


@dataclass(frozen=True)
class FamilyInstance:
    family_name: str
    instance_id: str
    seed: int
    size: int
    split: str
    scale_group: str
    problem: MILPProblem
    parameters: dict

    @property
    def units(self) -> int:
        return int(self.size)


class MILPFamily(Protocol):
    family_name: str

    def generate(self, seed: int, size: int, split: str, scale_group: str) -> FamilyInstance:
        ...


def instance_stats(problem: MILPProblem) -> dict:
    return {
        "num_variables": int(problem.num_vars),
        "num_continuous": int(len(problem.continuous_indices)),
        "num_binary": int(len(problem.binary_indices)),
        "num_constraints": int(problem.num_constraints),
    }
