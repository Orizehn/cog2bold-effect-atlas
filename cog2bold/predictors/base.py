"""Predictor abstraction.

A `CoefficientPredictor` maps `(cog_features, stim_encoding)` to basis coefficients.
The two implementations live in `linear.py` and `mlp.py`.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from cog2bold.config import PredictorConfig


class CoefficientPredictor(ABC):
    def __init__(self, config: PredictorConfig):
        self.config = config
        self.is_fitted = False
        self.cog_mean: Optional[np.ndarray] = None
        self.cog_std: Optional[np.ndarray] = None
        self.training_encodings: Optional[np.ndarray] = None
        self.n_stim_dim: int = 0

    # ---- core ----

    @abstractmethod
    def fit(
        self,
        cog_features: np.ndarray,
        stim_encodings: np.ndarray,
        target_coeffs: np.ndarray,
        cog_mean: np.ndarray = None,
        cog_std: np.ndarray = None,
        target_maps: np.ndarray = None,
        basis_object=None,
        val_data=None,
    ) -> "CoefficientPredictor": ...

    @abstractmethod
    def predict(
        self, cog_features: np.ndarray, stim_encodings: np.ndarray, warn_unseen: bool = False,
    ) -> np.ndarray: ...

    # ---- shared helpers ----

    def _set_normalisation(
        self, cog_features: np.ndarray, cog_mean=None, cog_std=None,
    ) -> None:
        self.cog_mean = cog_mean if cog_mean is not None else cog_features.mean(axis=0)
        self.cog_std = cog_std if cog_std is not None else cog_features.std(axis=0)
        self.cog_std = np.maximum(self.cog_std, 1e-8)

    def _build_Z(self, cog_features: np.ndarray, stim_encodings: np.ndarray) -> np.ndarray:
        cog_norm = (cog_features - self.cog_mean) / self.cog_std
        if self.config.cog_only:
            return cog_norm.astype(np.float64)
        return np.concatenate([cog_norm, stim_encodings], axis=1).astype(np.float64)

    def _record_training_encodings(self, stim_encodings: np.ndarray) -> None:
        self.training_encodings = np.unique(stim_encodings, axis=0)
        self.n_stim_dim = stim_encodings.shape[1]

    def _validate_encodings(self, stim_encodings: np.ndarray) -> None:
        if self.training_encodings is None:
            return
        for enc in stim_encodings:
            if not np.any(np.all(np.isclose(self.training_encodings, enc, atol=1e-6), axis=1)):
                print(f"[WARN] unseen stim encoding at predict time: {enc}")
                # only warn once per call
                return

    # ---- persistence ----

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "state": self._get_state(),
                    "is_fitted": self.is_fitted,
                    "cog_mean": self.cog_mean,
                    "cog_std": self.cog_std,
                    "training_encodings": self.training_encodings,
                    "n_stim_dim": self.n_stim_dim,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "CoefficientPredictor":
        from cog2bold.predictors import create_predictor  # late import

        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = create_predictor(data["config"])
        instance._set_state(data["state"])
        instance.is_fitted = data["is_fitted"]
        instance.cog_mean = data["cog_mean"]
        instance.cog_std = data["cog_std"]
        instance.training_encodings = data["training_encodings"]
        instance.n_stim_dim = data["n_stim_dim"]
        return instance

    @abstractmethod
    def _get_state(self) -> dict: ...

    @abstractmethod
    def _set_state(self, state: dict) -> None: ...
