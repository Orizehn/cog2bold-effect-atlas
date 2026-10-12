"""Random orthonormal basis — debugging baseline."""

from __future__ import annotations

import numpy as np

from cog2bold.bases.base import BasisTransform


class RandomOrthonormalBasis(BasisTransform):
    def __init__(self, config):
        super().__init__(config)
        self.Q: np.ndarray | None = None  # (K, d)

    def fit(self, train_maps: np.ndarray, **kwargs) -> "RandomOrthonormalBasis":
        n = train_maps.shape[0]
        self.original_shape = train_maps.shape[1:]
        d = int(np.prod(self.original_shape))
        rng = np.random.default_rng(self.config.random_seed)
        Q = rng.standard_normal((self.n_components, d))
        Qo, _ = np.linalg.qr(Q.T)
        self.Q = Qo.T[: self.n_components].astype(np.float32)
        self.mean_map = train_maps.mean(axis=0).astype(np.float32)
        self.is_fitted = True
        return self

    def encode(self, activation_map: np.ndarray, **kwargs) -> np.ndarray:
        return self.Q @ (activation_map - self.mean_map).astype(np.float32)

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        return self.Q.T @ coefficients.astype(np.float32) + self.mean_map

    def get_decode_matrix(self) -> np.ndarray:
        return self.Q.astype(np.float32)

    def _get_state(self) -> dict:
        return {"Q": self.Q, "mean_map": self.mean_map, "original_shape": self.original_shape}

    def _set_state(self, state: dict) -> None:
        self.Q = state["Q"]
        self.mean_map = state["mean_map"]
        self.original_shape = state["original_shape"]
