"""PCA basis."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from cog2bold.bases.base import BasisTransform
from cog2bold.config import BasisConfig


class PCABasis(BasisTransform):
    def __init__(self, config: BasisConfig):
        super().__init__(config)
        self.pca = PCA(n_components=config.n_components, random_state=config.random_seed)

    def fit(self, train_maps: np.ndarray, **kwargs) -> "PCABasis":
        n = train_maps.shape[0]
        self.original_shape = train_maps.shape[1:]
        X = train_maps.reshape(n, -1)
        self.pca.fit(X)
        self.mean_map = self.pca.mean_.reshape(self.original_shape).astype(np.float32)
        self.is_fitted = True
        var = self.pca.explained_variance_ratio_.sum()
        print(f"[PCA] {self.n_components} components, explained var={var:.3f}")
        return self

    def encode(self, activation_map: np.ndarray, **kwargs) -> np.ndarray:
        return self.pca.transform(activation_map.reshape(1, -1))[0]

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(coefficients.reshape(1, -1))[0].reshape(self.original_shape)

    def get_decode_matrix(self) -> np.ndarray:
        # decode(c) = c @ components_ + mean ; components_ shape (K, d)
        return self.pca.components_.astype(np.float32)

    def _get_state(self) -> dict:
        return {"pca": self.pca, "mean_map": self.mean_map, "original_shape": self.original_shape}

    def _set_state(self, state: dict) -> None:
        self.pca = state["pca"]
        self.mean_map = state["mean_map"]
        self.original_shape = state["original_shape"]
