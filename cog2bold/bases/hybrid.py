"""Hybrid basis: supervised + unsupervised on residuals.

Fits a supervised basis (ridgecol/cca/pls), computes residuals, then fits an
unsupervised basis (pca/ica/atlas/random) on those residuals. The two parts
span (approximately) orthogonal subspaces. Their decode matrices stack to form
W_hybrid; encode concatenates `c = [c_sup; c_unsup]`.

extra_params:
    - supervised_method: 'ridgecol' | 'cca' | 'pls'  (default 'ridgecol')
    - unsupervised_method: 'pca' | 'ica' | 'atlas' | 'random'  (default 'pca')
    - sup_*, unsup_* prefixes pass through to the sub-basis extra_params
    - normalize_basis_vectors: bool (default False) — unit-norm rows of W_hybrid;
      coefficients are scaled accordingly so reconstruction is identical.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cog2bold.bases.base import BasisTransform
from cog2bold.config import BasisConfig


class HybridBasis(BasisTransform):
    requires_supervision = True

    def __init__(self, config: BasisConfig):
        super().__init__(config)
        self.sup_method = config.extra_params.get("supervised_method", "ridgecol")
        self.unsup_method = config.extra_params.get("unsupervised_method", "pca")
        self.sup_basis: Optional[BasisTransform] = None
        self.unsup_basis: Optional[BasisTransform] = None
        self.K_sup = 0
        self.K_unsup = 0
        self.W_hybrid: Optional[np.ndarray] = None
        self.normalize_basis_vectors = bool(
            config.extra_params.get("normalize_basis_vectors", False)
        )
        self.coeff_scales: Optional[np.ndarray] = None

    def fit(
        self,
        train_maps: np.ndarray,
        cog_features: np.ndarray = None,
        stim_encodings: np.ndarray = None,
        masker=None,
        **kwargs,
    ) -> "HybridBasis":
        from cog2bold.bases import create_basis  # late import

        if cog_features is None or stim_encodings is None:
            raise ValueError("HybridBasis requires cog_features and stim_encodings")
        n_train = train_maps.shape[0]
        self.original_shape = train_maps.shape[1:]

        X_dim = cog_features.shape[1] + stim_encodings.shape[1]
        # Budget passed to the supervised sub-basis. For CCA/PLS this caps the number
        # of components; RidgeCol IGNORES it and uses its full design width (dim_Z,
        # which grows with quadratic / interaction terms).
        K_sup_budget = max(1, min(X_dim, n_train - 1, self.n_components))

        sup_extra = {k[4:]: v for k, v in self.config.extra_params.items() if k.startswith("sup_")}
        unsup_extra = {k[6:]: v for k, v in self.config.extra_params.items() if k.startswith("unsup_")}

        sup_config = BasisConfig(
            basis_type=self.sup_method,
            n_components=K_sup_budget,
            random_seed=self.config.random_seed,
            extra_params=sup_extra,
        )
        self.sup_basis = create_basis(sup_config)
        self.sup_basis.fit(train_maps, cog_features=cog_features, stim_encodings=stim_encodings)

        # True supervised component count, read from the fitted decode matrix — for
        # RidgeCol this is dim_Z (incl. quadratic/interaction terms), which can exceed
        # the linear K_sup_budget. Top up to the requested total with unsupervised
        # components on the residual (so n_components is respected when dim_Z <= it).
        self.K_sup = int(self.sup_basis.get_decode_matrix().shape[0])
        self.K_unsup = max(0, self.n_components - self.K_sup)
        total = self.K_sup + self.K_unsup
        print(
            f"[Hybrid] total K={total} (requested {self.n_components}), "
            f"sup={self.sup_method} (K={self.K_sup}), unsup={self.unsup_method} (K={self.K_unsup}), "
            f"X_dim={X_dim}, n_train={n_train}"
        )
        if total > self.n_components:
            print(f"[Hybrid] note: supervised design (dim_Z={self.K_sup}) exceeds "
                  f"n_components={self.n_components}; total components = {total}.")

        # Residual-encoding: unsupervised on residuals
        if self.K_unsup > 0:
            sup_coeffs = self.sup_basis.encode_batch(
                train_maps, cog_features=cog_features, stim_encodings=stim_encodings
            )
            sup_recons = self.sup_basis.decode_batch(sup_coeffs)
            residuals = train_maps - sup_recons
            unsup_config = BasisConfig(
                basis_type=self.unsup_method,
                n_components=self.K_unsup,
                random_seed=self.config.random_seed,
                extra_params=unsup_extra,
            )
            self.unsup_basis = create_basis(unsup_config)
            if self.unsup_method == "atlas":
                if masker is None:
                    raise ValueError("HybridBasis with unsup='atlas' needs masker= kwarg")
                self.unsup_basis.fit(residuals, masker=masker)
            else:
                self.unsup_basis.fit(residuals)

        self._build_combined_decode_matrix()
        self.is_fitted = True
        return self

    def _build_combined_decode_matrix(self) -> None:
        W_sup = self.sup_basis.get_decode_matrix()
        if W_sup is None:
            raise ValueError(f"Supervised basis {self.sup_method!r} doesn't expose a decode matrix")
        if self.K_unsup > 0:
            W_unsup = self.unsup_basis.get_decode_matrix()
            if W_unsup is None:
                raise ValueError(f"Unsupervised basis {self.unsup_method!r} doesn't expose a decode matrix")
            self.W_hybrid = np.vstack([W_sup, W_unsup]).astype(np.float32)
        else:
            self.W_hybrid = W_sup.astype(np.float32)

        if self.normalize_basis_vectors:
            norms = np.maximum(np.linalg.norm(self.W_hybrid, axis=1), 1e-8)
            self.coeff_scales = norms.astype(np.float32)
            self.W_hybrid = (self.W_hybrid / norms[:, None]).astype(np.float32)

    def encode(
        self,
        activation_map: np.ndarray,
        cog_features: np.ndarray = None,
        stim_encoding: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        if self.sup_method == "ridgecol":
            if cog_features is None or stim_encoding is None:
                raise ValueError("Hybrid+ridgecol requires cog_features and stim_encoding")
            c_sup = self.sup_basis.encode(activation_map, cog_features=cog_features,
                                          stim_encoding=stim_encoding)
        else:
            c_sup = self.sup_basis.encode(activation_map)

        if self.K_unsup > 0:
            sup_recon = self.sup_basis.decode(c_sup)
            residual = activation_map - sup_recon
            c_unsup = self.unsup_basis.encode(residual)
            c = np.concatenate([c_sup, c_unsup])
        else:
            c = c_sup

        if self.coeff_scales is not None:
            c = c * self.coeff_scales
        return c

    def encode_batch(self, maps: np.ndarray, **kwargs) -> np.ndarray:
        cog = kwargs.get("cog_features")
        stim = kwargs.get("stim_encodings")
        if self.sup_method == "ridgecol":
            sup_coeffs = self.sup_basis.encode_batch(maps, cog_features=cog, stim_encodings=stim)
        else:
            sup_coeffs = self.sup_basis.encode_batch(maps)
        if self.K_unsup > 0:
            sup_recons = self.sup_basis.decode_batch(sup_coeffs)
            residuals = maps - sup_recons
            unsup_coeffs = self.unsup_basis.encode_batch(residuals)
            coeffs = np.concatenate([sup_coeffs, unsup_coeffs], axis=1)
        else:
            coeffs = sup_coeffs
        if self.coeff_scales is not None:
            coeffs = coeffs * self.coeff_scales[None, :]
        return coeffs.astype(np.float32)

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        flat = coefficients.astype(np.float32) @ self.W_hybrid
        if self.sup_basis is not None and self.sup_basis.mean_map is not None:
            flat = flat + self.sup_basis.mean_map.flatten()
        return flat.reshape(self.original_shape)

    def get_decode_matrix(self) -> np.ndarray:
        return self.W_hybrid.astype(np.float32)

    @property
    def mean_map(self):  # type: ignore[override]
        if self.sup_basis is not None:
            return self.sup_basis.mean_map
        return None

    @mean_map.setter
    def mean_map(self, value):
        # ABC sets self.mean_map = None in __init__; ignore — we delegate to sup_basis.
        pass

    def _get_state(self) -> dict:
        return {
            "sup_method": self.sup_method,
            "unsup_method": self.unsup_method,
            "K_sup": self.K_sup,
            "K_unsup": self.K_unsup,
            "original_shape": self.original_shape,
            "W_hybrid": self.W_hybrid,
            "normalize_basis_vectors": self.normalize_basis_vectors,
            "coeff_scales": self.coeff_scales,
            "sup_basis_state": self.sup_basis._get_state() if self.sup_basis else None,
            "sup_basis_config": self.sup_basis.config if self.sup_basis else None,
            "unsup_basis_state": self.unsup_basis._get_state() if self.unsup_basis else None,
            "unsup_basis_config": self.unsup_basis.config if self.unsup_basis else None,
        }

    def _set_state(self, state: dict) -> None:
        from cog2bold.bases import create_basis

        self.sup_method = state["sup_method"]
        self.unsup_method = state["unsup_method"]
        self.K_sup = state["K_sup"]
        self.K_unsup = state["K_unsup"]
        self.original_shape = state["original_shape"]
        self.W_hybrid = state.get("W_hybrid")
        self.normalize_basis_vectors = state.get("normalize_basis_vectors", False)
        self.coeff_scales = state.get("coeff_scales")
        if state["sup_basis_state"] is not None:
            self.sup_basis = create_basis(state["sup_basis_config"])
            self.sup_basis._set_state(state["sup_basis_state"])
            self.sup_basis.is_fitted = True
        if state["unsup_basis_state"] is not None:
            self.unsup_basis = create_basis(state["unsup_basis_config"])
            self.unsup_basis._set_state(state["unsup_basis_state"])
            self.unsup_basis.is_fitted = True
