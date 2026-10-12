"""Identity basis — just flatten/reshape. Equivalent to ridge on raw voxels."""

from __future__ import annotations

import numpy as np

from cog2bold.bases.base import BasisTransform


class IdentityBasis(BasisTransform):
    def fit(self, train_maps: np.ndarray, **kwargs) -> "IdentityBasis":
        self.original_shape = train_maps.shape[1:]
        self.n_components = int(np.prod(self.original_shape))
        self.is_fitted = True
        return self

    def encode(self, activation_map: np.ndarray, **kwargs) -> np.ndarray:
        return activation_map.flatten()

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        return coefficients.reshape(self.original_shape)

    def get_decode_matrix(self) -> None:  # too large to materialise
        return None

    def _get_state(self) -> dict:
        return {"original_shape": self.original_shape, "n_components": self.n_components}

    def _set_state(self, state: dict) -> None:
        self.original_shape = state["original_shape"]
        self.n_components = state["n_components"]
