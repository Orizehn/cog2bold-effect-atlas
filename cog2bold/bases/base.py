"""Basis abstraction.

A `BasisTransform` is a small, fittable spatial decomposition:
    coeffs = encode(map)
    map_recon = decode(coeffs)
    coeffs.shape == (n_components,)
    map.shape == (n_voxels,)   # 1D — z-maps only

Some bases (ridgecol, hybrid) need extra inputs (cog_features, stim_encoding).
The pipeline knows which by inspecting `requires_supervision`.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from cog2bold.config import BasisConfig


class BasisTransform(ABC):
    """Abstract base class for spatial basis decompositions."""

    requires_supervision: bool = False  # subclasses set True if encode/fit need cog/stim

    def __init__(self, config: BasisConfig):
        self.config = config
        self.is_fitted = False
        self.n_components = config.n_components
        self.original_shape: Optional[tuple] = None
        self.mean_map: Optional[np.ndarray] = None  # (n_voxels,) or None

    # ---- core API ----

    @abstractmethod
    def fit(self, train_maps: np.ndarray, **kwargs) -> "BasisTransform":
        """Fit basis on training maps with shape (n_train, n_voxels)."""

    @abstractmethod
    def encode(self, activation_map: np.ndarray, **kwargs) -> np.ndarray:
        """Project a single map (n_voxels,) to coefficients (n_components,)."""

    @abstractmethod
    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct (n_voxels,) from coefficients (n_components,)."""

    def encode_batch(self, maps: np.ndarray, **kwargs) -> np.ndarray:
        """Encode (n, n_voxels) -> (n, n_components). Subclasses override if vectorisable."""
        cog = kwargs.get("cog_features")
        stim = kwargs.get("stim_encodings")
        if cog is not None and stim is not None:
            return np.asarray([
                self.encode(m, cog_features=cog[i], stim_encoding=stim[i])
                for i, m in enumerate(maps)
            ])
        return np.asarray([self.encode(m) for m in maps])

    def decode_batch(self, coeffs: np.ndarray) -> np.ndarray:
        """Decode (n, n_components) -> (n, n_voxels)."""
        # Vectorised when possible: for affine bases use the decode matrix directly.
        W = self.get_decode_matrix()
        if W is not None:
            out = coeffs.astype(np.float32) @ W
            if self.mean_map is not None:
                out = out + self.mean_map.flatten()
            return out
        return np.asarray([self.decode(c) for c in coeffs])

    def get_decode_matrix(self) -> Optional[np.ndarray]:
        """Return W of shape (n_components, n_voxels) such that decode(c) = c @ W (+ mean).

        Bases that don't have a clean linear form return None.
        """
        return None

    # ---- persistence ----

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"config": self.config, "state": self._get_state(), "is_fitted": self.is_fitted}, f
            )

    @classmethod
    def load(cls, path: Path) -> "BasisTransform":
        with open(path, "rb") as f:
            data = pickle.load(f)
        from cog2bold.bases import create_basis  # late import to avoid cycles

        instance = create_basis(data["config"])
        instance._set_state(data["state"])
        instance.is_fitted = data["is_fitted"]
        return instance

    @abstractmethod
    def _get_state(self) -> dict: ...

    @abstractmethod
    def _set_state(self, state: dict) -> None: ...
