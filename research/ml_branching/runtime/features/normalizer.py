from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION


NORMALIZER_SCHEMA_VERSION = "feature_normalizer_v1"


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    schema_version: str = NORMALIZER_SCHEMA_VERSION
    epsilon: float = 1e-8

    @classmethod
    def fit(
        cls,
        matrices: list[np.ndarray],
        feature_names: list[str] | None = None,
        epsilon: float = 1e-8,
    ) -> "FeatureNormalizer":
        if not matrices:
            raise ValueError("cannot fit normalizer without feature matrices")
        names = feature_names or FEATURE_NAMES.copy()
        all_rows = np.vstack([np.asarray(matrix, dtype=float) for matrix in matrices])
        if all_rows.ndim != 2 or all_rows.shape[1] != len(names):
            raise ValueError("feature matrix dimension does not match feature_names")
        if not np.all(np.isfinite(all_rows)):
            raise ValueError("cannot fit normalizer on nan or inf features")
        mean = np.mean(all_rows, axis=0)
        std = np.std(all_rows, axis=0)
        std[std < epsilon] = 1.0
        return cls(mean=mean, std=std, feature_names=list(names), epsilon=float(epsilon))

    @property
    def scale(self) -> np.ndarray:
        return self.std

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix dimension does not match fitted normalizer")
        out = (matrix - self.mean) / self.std
        if not np.all(np.isfinite(out)):
            raise ValueError("normalized feature matrix contains nan or inf")
        return out

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureNormalizer":
        if data.get("schema_version", NORMALIZER_SCHEMA_VERSION) != NORMALIZER_SCHEMA_VERSION:
            raise ValueError(f"unsupported normalizer schema_version={data.get('schema_version')!r}")
        feature_names = list(data.get("feature_names", FEATURE_NAMES))
        if feature_names != FEATURE_NAMES:
            raise ValueError("normalizer feature_names do not match current FEATURE_NAMES")
        if data.get("feature_schema_version", FEATURE_SCHEMA_VERSION) != FEATURE_SCHEMA_VERSION:
            raise ValueError("normalizer feature_schema_version does not match current feature schema")
        std = data.get("std", data.get("scale"))
        if std is None:
            raise ValueError("normalizer is missing std")
        return cls(
            mean=np.asarray(data["mean"], dtype=float),
            std=np.asarray(std, dtype=float),
            feature_names=feature_names,
            feature_schema_version=data.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            schema_version=data.get("schema_version", NORMALIZER_SCHEMA_VERSION),
            epsilon=float(data.get("epsilon", 1e-8)),
        )
