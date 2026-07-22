"""Runtime components for learned branching policies."""

__all__ = ["LearnedBranchingPolicy", "LearnedGNNBranchingPolicy"]


def __getattr__(name: str):
    if name in {"LearnedBranchingPolicy", "LearnedGNNBranchingPolicy"}:
        from .inference import LearnedBranchingPolicy, LearnedGNNBranchingPolicy

        return {
            "LearnedBranchingPolicy": LearnedBranchingPolicy,
            "LearnedGNNBranchingPolicy": LearnedGNNBranchingPolicy,
        }[name]
    raise AttributeError(name)
