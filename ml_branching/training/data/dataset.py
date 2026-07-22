from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import validate_sample


def load_sample(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files if key != "metadata_json"}
        metadata = json.loads(str(data["metadata_json"].item()))
    sample = {"path": str(path), "arrays": arrays, "metadata": metadata}
    validate_sample(sample)
    return sample


class BranchingDataset:
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    @classmethod
    def from_dir(cls, path: str | Path) -> "BranchingDataset":
        root = Path(path)
        files = sorted(root.rglob("*.npz"))
        if not files:
            raise ValueError(f"no branching samples found under {root}")
        return cls([load_sample(file) for file in files])

    def by_split(self, split: str) -> "BranchingDataset":
        return BranchingDataset(
            [
                sample
                for sample in self.samples
                if sample["metadata"].get("split", sample["metadata"].get("config", {}).get("split")) == split
            ]
        )

    def multi_candidate_samples(self, split: str | None = None, min_candidates: int = 2) -> list[dict]:
        samples = self.by_split(split).samples if split is not None else self.samples
        return [
            sample
            for sample in samples
            if len(sample["arrays"]["candidate_indices"]) >= int(min_candidates)
        ]

    def split_names(self) -> list[str]:
        return sorted(
            {
                sample["metadata"].get("split", sample["metadata"].get("config", {}).get("split"))
                for sample in self.samples
            }
        )

    def instance_ids(self) -> list[str]:
        return sorted({sample["metadata"]["instance_id"] for sample in self.samples})

    def assert_disjoint_splits(self) -> None:
        split_to_ids: dict[str, set[str]] = {}
        for sample in self.samples:
            split = sample["metadata"].get("split", sample["metadata"].get("config", {}).get("split"))
            split_to_ids.setdefault(split, set()).add(sample["metadata"]["instance_id"])
        splits = sorted(split_to_ids)
        for i, left in enumerate(splits):
            for right in splits[i + 1 :]:
                overlap = split_to_ids[left] & split_to_ids[right]
                if overlap:
                    raise ValueError(f"instance_id leakage between {left} and {right}: {sorted(overlap)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
