"""CCA basis: supervised — finds map directions maximally correlated with (cog, stim)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

from cog2bold.bases.base import BasisTransform
from cog2bold.config import BasisConfig


class CCABasis(BasisTransform):
    requires_supervision = True

    def __init__(self, config: BasisConfig):
        super().__init__(config)
        self.B: Optional[np.ndarray] = None  # (K, d)
        ep = config.extra_params
        ypd = ep.get("y_pca_dim", 128)
        self.y_pca_dim = int(ypd) if ypd is not None else None
        self.scale_x = bool(ep.get("scale_x", True))
        self.max_iter = int(ep.get("cca_max_iter", 500))
        self.tol = float(ep.get("cca_tol", 1e-6))
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None

    def fit(
        self,
        train_maps: np.ndarray,
        cog_features: np.ndarray = None,
        stim_encodings: np.ndarray = None,
        **kwargs,
    ) -> "CCABasis":
        if cog_features is None or stim_encodings is None:
            raise ValueError("CCABasis requires cog_features and stim_encodings")
        n = train_maps.shape[0]
        self.original_shape = train_maps.shape[1:]
        d = int(np.prod(self.original_shape))

        Y = train_maps.reshape(n, d).astype(np.float64)
        Yc = Y - Y.mean(axis=0)
        self.mean_map = train_maps.mean(axis=0).astype(np.float32)

        X = np.concatenate([cog_features, stim_encodings], axis=1).astype(np.float64)
        if self.scale_x:
            self.x_mean = X.mean(axis=0)
            self.x_std = np.maximum(X.std(axis=0), 1e-8)
            X = (X - self.x_mean) / self.x_std

        if self.y_pca_dim is not None and self.y_pca_dim > 0:
            r = min(self.y_pca_dim, n - 1, d)
            pca = PCA(n_components=r, svd_solver="randomized", random_state=self.config.random_seed)
            Yp = pca.fit_transform(Yc)
            Y_components = pca.components_
        else:
            Yp = Yc
            Y_components = np.eye(d)

        K = min(self.n_components, X.shape[1], Yp.shape[1])
        cca = CCA(n_components=K, max_iter=self.max_iter, tol=self.tol)
        cca.fit(X, Yp)

        B_unorth = (cca.y_weights_.T @ Y_components)  # (K, d)
        Q, _ = np.linalg.qr(B_unorth.T)
        self.B = Q.T.astype(np.float32)
        self.n_components = K
        self.config.n_components = K
        self.is_fitted = True
        print(f"[CCA] K={K}, X_dim={X.shape[1]}, Y_dim={d}, y_pca_dim={Yp.shape[1]}")
        return self

    def encode(self, activation_map: np.ndarray, **kwargs) -> np.ndarray:
        return self.B @ (activation_map - self.mean_map).astype(np.float32)

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        return (self.B.T @ coefficients.astype(np.float32)).reshape(self.original_shape) + self.mean_map

    def get_decode_matrix(self) -> np.ndarray:
        return self.B.astype(np.float32)

    def _get_state(self) -> dict:
        return {
            "B": self.B,
            "mean_map": self.mean_map,
            "original_shape": self.original_shape,
            "n_components": self.n_components,
            "y_pca_dim": self.y_pca_dim,
            "scale_x": self.scale_x,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
        }

    def _set_state(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
