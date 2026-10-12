"""RidgeColBasis — supervised basis where each row is a column of a ridge-fitted W.

Builds a design matrix Z = [cog_norm, stim_encoding, optional cog², optional cog×stim],
fits Y = Z @ W via ridge, then treats W's rows as basis vectors. The "coefficient"
for any sample is just the Z vector — a deterministic function of (cog, stim), not of
the map. This means coefficient loss is degenerate for this basis (predictor learns
identity). Use loss_mode='map' with this basis.

extra_params:
    - ridge_alpha (default 1.0)
    - include_quadratic (default False)
    - include_interactions (default False)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from cog2bold.bases.base import BasisTransform
from cog2bold.config import BasisConfig


class RidgeColBasis(BasisTransform):
    requires_supervision = True

    def __init__(self, config: BasisConfig):
        super().__init__(config)
        self.W: Optional[np.ndarray] = None  # (dim_Z, n_voxels)
        self.ridge_alpha = float(config.extra_params.get("ridge_alpha", 1.0))
        self.include_quadratic = bool(config.extra_params.get("include_quadratic", False))
        self.include_interactions = bool(config.extra_params.get("include_interactions", False))
        self.cog_mean: Optional[np.ndarray] = None
        self.cog_std: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(
        self,
        train_maps: np.ndarray,
        cog_features: np.ndarray = None,
        stim_encodings: np.ndarray = None,
        **kwargs,
    ) -> "RidgeColBasis":
        if cog_features is None or stim_encodings is None:
            raise ValueError("RidgeColBasis requires cog_features and stim_encodings")
        n_train = train_maps.shape[0]
        self.original_shape = train_maps.shape[1:]
        d = int(np.prod(self.original_shape))

        # Train-only cog stats
        self.cog_mean = cog_features.mean(axis=0)
        self.cog_std = np.maximum(cog_features.std(axis=0), 1e-8)
        cog_norm = (cog_features - self.cog_mean) / self.cog_std

        Z = self._build_Z_matrix(cog_norm, stim_encodings)
        Y = train_maps.reshape(n_train, d).astype(np.float64)

        # W = (Z'Z + α I)^-1 Z'Y
        ZtZ = Z.T @ Z
        ZtY = Z.T @ Y
        ridge = self.ridge_alpha * np.eye(Z.shape[1])
        self.W = np.linalg.solve(ZtZ + ridge, ZtY).astype(np.float32)

        self.n_components = Z.shape[1]
        self.config.n_components = self.n_components
        self.is_fitted = True

        train_mse = float(np.mean((Y - Z @ self.W) ** 2))
        print(
            f"[RidgeCol] dim_Z={self.n_components} (cog={cog_features.shape[1]}, "
            f"stim={stim_encodings.shape[1]}, quad={self.include_quadratic}, "
            f"inter={self.include_interactions}), train MSE={train_mse:.4g}"
        )
        return self

    def _build_Z_matrix(self, cog_norm: np.ndarray, stim_encodings: np.ndarray) -> np.ndarray:
        n_cog = cog_norm.shape[1]
        n_stim = stim_encodings.shape[1]
        parts = [cog_norm, stim_encodings]
        names = [f"cog{i}" for i in range(n_cog)] + [f"stim{i}" for i in range(n_stim)]
        if self.include_quadratic:
            parts.append(cog_norm ** 2)
            names += [f"cog{i}^2" for i in range(n_cog)]
        if self.include_interactions:
            inter = (cog_norm[:, :, None] * stim_encodings[:, None, :]).reshape(cog_norm.shape[0], -1)
            parts.append(inter)
            names += [f"cog{i}xstim{j}" for i in range(n_cog) for j in range(n_stim)]
        Z = np.concatenate(parts, axis=1).astype(np.float64)
        self.feature_names = names
        return Z

    def _build_z_vector(self, cog: np.ndarray, stim: np.ndarray) -> np.ndarray:
        cog_norm = (cog - self.cog_mean) / self.cog_std
        parts = [cog_norm, stim]
        if self.include_quadratic:
            parts.append(cog_norm ** 2)
        if self.include_interactions:
            parts.append(np.outer(cog_norm, stim).reshape(-1))
        return np.concatenate(parts).astype(np.float64)

    def encode(
        self,
        activation_map: np.ndarray,
        cog_features: np.ndarray = None,
        stim_encoding: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        # NB: the activation_map is intentionally ignored — coefficients = Z(cog, stim).
        if cog_features is None or stim_encoding is None:
            raise ValueError("RidgeColBasis.encode requires cog_features and stim_encoding")
        return self._build_z_vector(cog_features, stim_encoding)

    def encode_batch(self, maps: np.ndarray, **kwargs) -> np.ndarray:
        cog = kwargs["cog_features"]
        stim = kwargs["stim_encodings"]
        cog_norm = (cog - self.cog_mean) / self.cog_std
        return self._build_Z_matrix(cog_norm, stim).astype(np.float32)

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        return (coefficients.astype(np.float32) @ self.W).reshape(self.original_shape)

    def get_decode_matrix(self) -> np.ndarray:
        return self.W.astype(np.float32)

    def _get_state(self) -> dict:
        return {
            "W": self.W,
            "ridge_alpha": self.ridge_alpha,
            "include_quadratic": self.include_quadratic,
            "include_interactions": self.include_interactions,
            "cog_mean": self.cog_mean,
            "cog_std": self.cog_std,
            "original_shape": self.original_shape,
            "feature_names": self.feature_names,
            "n_components": self.n_components,
        }

    def _set_state(self, state: dict) -> None:
        self.W = state["W"]
        self.ridge_alpha = state["ridge_alpha"]
        self.include_quadratic = state["include_quadratic"]
        self.include_interactions = state["include_interactions"]
        self.cog_mean = state["cog_mean"]
        self.cog_std = state["cog_std"]
        self.original_shape = state["original_shape"]
        self.feature_names = state["feature_names"]
        self.n_components = state["n_components"]
