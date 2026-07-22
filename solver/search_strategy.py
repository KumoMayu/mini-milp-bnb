from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _BestBoundEntry:
    priority: float
    node_id: int
    node: Any = field(compare=False)
    lp_result: Any = field(compare=False)
    internal_bound: float = field(compare=False)


class BestBoundNodePool:
    """Open-node pool that pops the largest internal LP bound first."""

    def __init__(self) -> None:
        self._heap: list[_BestBoundEntry] = []

    def push(self, node, lp_result, internal_bound: float) -> None:
        heapq.heappush(
            self._heap,
            _BestBoundEntry(
                priority=-float(internal_bound),
                node_id=int(node.node_id),
                node=node,
                lp_result=lp_result,
                internal_bound=float(internal_bound),
            ),
        )

    def pop(self):
        entry = heapq.heappop(self._heap)
        return entry.node, entry.lp_result, entry.internal_bound

    def __len__(self) -> int:
        return len(self._heap)

    def best_bound(self) -> float | None:
        if not self._heap:
            return None
        return float(self._heap[0].internal_bound)


class DepthFirstNodePool:
    """Open-node pool that pops the most recently opened node first."""

    def __init__(self) -> None:
        self._stack: list[tuple[Any, Any, float]] = []

    def push(self, node, lp_result, internal_bound: float) -> None:
        self._stack.append((node, lp_result, float(internal_bound)))

    def pop(self):
        return self._stack.pop()

    def __len__(self) -> int:
        return len(self._stack)

    def best_bound(self) -> float | None:
        if not self._stack:
            return None
        return max(float(item[2]) for item in self._stack)
