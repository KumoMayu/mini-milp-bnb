from __future__ import annotations

from dataclasses import dataclass, field

from .base import BranchingContext


@dataclass
class PseudocostStats:
    toward_zero_total: float = 0.0
    toward_zero_count: int = 0
    toward_one_total: float = 0.0
    toward_one_count: int = 0

    @property
    def toward_zero_average(self) -> float | None:
        if self.toward_zero_count <= 0:
            return None
        return self.toward_zero_total / self.toward_zero_count

    @property
    def toward_one_average(self) -> float | None:
        if self.toward_one_count <= 0:
            return None
        return self.toward_one_total / self.toward_one_count


@dataclass
class PseudocostPolicy:
    """Generic binary pseudocost branching policy.

    The solver stores internal LP bounds as maximization bounds. Branching
    should make a child LP bound no better than its parent, so the improvement
    is `parent_bound - child_bound` in this internal scale.
    """

    fallback: str = "most_fractional"
    initial_pseudocost: float = 1.0
    epsilon: float = 1e-9
    stats: dict[int, PseudocostStats] = field(default_factory=dict)

    def _stats(self, variable: int) -> PseudocostStats:
        variable = int(variable)
        if variable not in self.stats:
            self.stats[variable] = PseudocostStats()
        return self.stats[variable]

    def _fractional_distances(self, value: float) -> tuple[float, float]:
        value = float(value)
        return max(value, self.epsilon), max(1.0 - value, self.epsilon)

    def _average_or_initial(self, variable: int, branch_value: int) -> float:
        stats = self._stats(variable)
        average = stats.toward_zero_average if int(branch_value) == 0 else stats.toward_one_average
        if average is None:
            return float(self.initial_pseudocost)
        return max(float(average), float(self.epsilon))

    def score_candidate(self, context: BranchingContext, variable: int) -> float:
        value = float(context.lp_result.x[int(variable)])
        distance_zero, distance_one = self._fractional_distances(value)
        down = self._average_or_initial(variable, 0) * distance_zero
        up = self._average_or_initial(variable, 1) * distance_one
        return (max(down, self.epsilon) * max(up, self.epsilon)) + 0.1 * min(down, up)

    def select_variable(self, context: BranchingContext) -> int:
        if not context.candidate_indices:
            raise ValueError("no fractional binary branching candidates")
        if self.fallback != "most_fractional":
            raise ValueError('PseudocostPolicy currently supports fallback="most_fractional"')
        scores = []
        for variable in context.candidate_indices:
            stats = self._stats(int(variable))
            has_both = stats.toward_zero_count > 0 and stats.toward_one_count > 0
            score = self.score_candidate(context, int(variable))
            if not has_both:
                # Deterministic most-fractional fallback while the variable has
                # insufficient history, matching the project's baseline tie-break.
                score = abs(float(context.lp_result.x[int(variable)]) - round(float(context.lp_result.x[int(variable)])))
            scores.append((float(score), int(variable)))
        return int(max(scores)[1])

    def observe_branch_result(
        self,
        context: BranchingContext,
        branch_var: int,
        branch_value: int,
        child_status: str,
        child_bound: float | None,
    ) -> None:
        if context.current_node_internal_bound is None:
            return
        if str(child_status) != "optimal" or child_bound is None:
            return
        value = float(context.lp_result.x[int(branch_var)])
        distance_zero, distance_one = self._fractional_distances(value)
        distance = distance_zero if int(branch_value) == 0 else distance_one
        raw_improvement = float(context.current_node_internal_bound) - float(child_bound)
        improvement_per_unit = max(0.0, raw_improvement) / max(distance, self.epsilon)
        stats = self._stats(int(branch_var))
        if int(branch_value) == 0:
            stats.toward_zero_total += improvement_per_unit
            stats.toward_zero_count += 1
        elif int(branch_value) == 1:
            stats.toward_one_total += improvement_per_unit
            stats.toward_one_count += 1
        else:
            raise ValueError("PseudocostPolicy only supports binary branch values 0 and 1")

    def reset(self) -> None:
        self.stats.clear()

    def history_snapshot(self) -> dict[int, dict[str, float | int | None]]:
        return {
            int(variable): {
                "toward_zero_total": float(stats.toward_zero_total),
                "toward_zero_count": int(stats.toward_zero_count),
                "toward_zero_average": stats.toward_zero_average,
                "toward_one_total": float(stats.toward_one_total),
                "toward_one_count": int(stats.toward_one_count),
                "toward_one_average": stats.toward_one_average,
            }
            for variable, stats in sorted(self.stats.items())
        }


__all__ = ["PseudocostPolicy", "PseudocostStats"]
