"""ICA basis.

Bug fix from the legacy code: the previous version centered X manually before
calling FastICA.fit(X_centered), then added the manual mean back inside decode().
Since FastICA centers internally and stores the mean in `ica.mean_`, the manual
centering made the code only "accidentally" correct (because ica.mean_ ended up
~0). Here we just trust sklearn's API.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import FastICA

from cog2bold.bases.base import BasisTransform
from cog2bold.config import BasisConfig


class ICABasis(BasisTransform):
    def __init__(self, config: BasisConfig):
        super().__init__(config)
        self.ica = FastICA(
            n_components=config.n_components,
            random_state=config.random_seed,
            max_iter=500,
            tol=1e-3,
        )

    def fit(self, train_maps: np.ndarray, **kwargs) -> "ICABasis":
        n = train_maps.shape[0]
        self.original_shape = train_maps.shape[1:]
        X = train_maps.reshape(n, -1)
        self.ica.fit(X)
        # mean_ is stored on the estimator; we keep it here for the unified API
        self.mean_map = np.asarray(self.ica.mean_, dtype=np.float32).reshape(self.original_shape)
        self.is_fitted = True
        print(f"[ICA] fitted with {self.n_components} components")
        return self

    def encode(self, activation_map: np.ndarray, **kwargs) -> np.ndarray:
        return self.ica.transform(activation_map.reshape(1, -1))[0]

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        # inverse_transform adds ica.mean_ internally
        return self.ica.inverse_transform(coefficients.reshape(1, -1))[0].reshape(self.original_shape)

    def get_decode_matrix(self) -> np.ndarray:
        # decode(c) = c @ mixing_.T + ica.mean_   (mixing_ shape: (d, K))
        # We expose the (K, d) matrix and let decode_batch add mean_map.
        return self.ica.mixing_.T.astype(np.float32)

    def _get_state(self) -> dict:
        return {"ica": self.ica, "mean_map": self.mean_map, "original_shape": self.original_shape}

    def _set_state(self, state: dict) -> None:
        self.ica = state["ica"]
        self.mean_map = state["mean_map"]
        self.original_shape = state["original_shape"]
