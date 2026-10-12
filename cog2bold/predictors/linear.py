"""LinearPredictor — ridge regression in coefficient space, or closed-form ridge in
voxel space when loss_mode='map'.

Both paths run in milliseconds. The legacy `numerical-gradient` path is gone.

Map-loss closed form
--------------------
Setup: `decode(c) = c @ B + μ` (every basis we keep exposes `B = get_decode_matrix()`),
`pred = Z @ W`, so `pred_map = Z @ (W @ B) + μ`. Define `M = W @ B`.

  1. Solve in voxel space:  `M = (Zᵀ Z + α I)^-1 Zᵀ (Y − μ)`
  2. Recover `W`:
       - orthonormal B (pca / ica / random / cca / pls):  `W = M @ Bᵀ`
       - general B (ridgecol / hybrid):                   `W = M @ Bᵀ @ (B Bᵀ + λ I)^-1`

For loss types L1 / correlation we fall back to PyTorch + LBFGS, fully seeded.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cog2bold.config import PredictorConfig
from cog2bold.predictors.base import CoefficientPredictor


def _is_orthonormal(B: np.ndarray, tol: float = 1e-3) -> bool:
    K = B.shape[0]
    BBT = B @ B.T
    return np.allclose(BBT, np.eye(K, dtype=BBT.dtype), atol=tol)


class LinearPredictor(CoefficientPredictor):
    def __init__(self, config: PredictorConfig):
        super().__init__(config)
        self.W: Optional[np.ndarray] = None  # (input_dim, n_components)
        self.alpha = float(config.weight_decay)

    # ---- fit ----

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
    ) -> "LinearPredictor":
        self._set_normalisation(cog_features, cog_mean, cog_std)
        self._record_training_encodings(stim_encodings)
        Z = self._build_Z(cog_features, stim_encodings)

        if self.config.loss_mode == "coefficient":
            self._fit_coeff_loss(Z, target_coeffs)
        else:
            if basis_object is None or target_maps is None:
                raise ValueError("loss_mode='map' requires basis_object and target_maps")
            B = basis_object.get_decode_matrix()
            mu_flat = (
                np.asarray(basis_object.mean_map).flatten()
                if getattr(basis_object, "mean_map", None) is not None
                else None
            )
            if B is None:
                raise ValueError(
                    f"Basis {type(basis_object).__name__} doesn't expose a decode matrix; "
                    f"map loss is not supported for it. Use loss_mode='coefficient'."
                )
            map_loss_type = self.config.map_loss_type
            if map_loss_type == "l2":
                self._fit_map_l2_closed_form(Z, target_maps, B, mu_flat)
            else:
                self._fit_map_torch(Z, target_maps, B, mu_flat, map_loss_type)

        self.is_fitted = True
        return self

    def _fit_coeff_loss(self, Z: np.ndarray, target_coeffs: np.ndarray) -> None:
        Y = target_coeffs.astype(np.float64)
        ZtZ = Z.T @ Z
        ZtY = Z.T @ Y
        ridge = self.alpha * np.eye(Z.shape[1])
        self.W = np.linalg.solve(ZtZ + ridge, ZtY).astype(np.float32)
        train_mse = float(np.mean((Y - Z @ self.W) ** 2))
        print(f"[Linear/coeff] Z={Z.shape} -> coeffs={Y.shape}, α={self.alpha}, train MSE={train_mse:.4g}")

    def _fit_map_l2_closed_form(
        self, Z: np.ndarray, target_maps: np.ndarray, B: np.ndarray, mu_flat: Optional[np.ndarray],
    ) -> None:
        n = target_maps.shape[0]
        d = int(np.prod(target_maps.shape[1:]))
        Y = target_maps.reshape(n, d).astype(np.float64)
        if mu_flat is not None:
            Y = Y - mu_flat
        # Step 1: M = (Z'Z + α I)^-1 Z'Y           shape (p, d)
        ZtZ = Z.T @ Z
        ZtY = Z.T @ Y
        ridge = self.alpha * np.eye(Z.shape[1])
        M = np.linalg.solve(ZtZ + ridge, ZtY)
        # Step 2: recover W
        Bf = B.astype(np.float64)
        if _is_orthonormal(Bf):
            W = M @ Bf.T
        else:
            # W = M @ Bᵀ @ (B Bᵀ + λ I)^-1
            BBT = Bf @ Bf.T
            recover = np.linalg.solve(BBT + self.alpha * np.eye(B.shape[0]), Bf @ M.T).T
            W = recover  # shape (p, K)
        self.W = W.astype(np.float32)
        # Diagnostic
        Y_pred_flat = Z @ W @ Bf
        train_map_mse = float(np.mean((Y - Y_pred_flat) ** 2))
        print(
            f"[Linear/map-l2] closed form Z={Z.shape}, B={B.shape}, "
            f"orthonormal={_is_orthonormal(Bf)}, train map MSE={train_map_mse:.4g}"
        )

    def _fit_map_torch(
        self, Z: np.ndarray, target_maps: np.ndarray, B: np.ndarray, mu_flat: Optional[np.ndarray],
        loss_type: str,
    ) -> None:
        import torch

        torch.manual_seed(self.config.random_seed)
        n = target_maps.shape[0]
        d = int(np.prod(target_maps.shape[1:]))
        Y = torch.from_numpy(target_maps.reshape(n, d).astype(np.float32))
        if mu_flat is not None:
            Y = Y - torch.from_numpy(mu_flat.astype(np.float32))
        Zt = torch.from_numpy(Z.astype(np.float32))
        Bt = torch.from_numpy(B.astype(np.float32))

        p = Z.shape[1]
        K = B.shape[0]
        # init from coefficient-loss closed form (cheap warm start)
        with torch.no_grad():
            W0 = self._closed_form_warmstart(Zt, Bt, Y)
        W = torch.nn.Parameter(W0.clone())
        opt = torch.optim.LBFGS([W], lr=0.5, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def loss_fn(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
            if loss_type == "l1":
                return (pred - true).abs().mean()
            # correlation: maximise -> minimise negative mean correlation
            pc = pred - pred.mean(dim=1, keepdim=True)
            tc = true - true.mean(dim=1, keepdim=True)
            num = (pc * tc).sum(dim=1)
            den = pc.norm(dim=1) * tc.norm(dim=1) + 1e-8
            return -(num / den).mean()

        def closure():
            opt.zero_grad()
            pred = Zt @ W @ Bt
            loss = loss_fn(pred, Y) + self.alpha * (W ** 2).sum()
            loss.backward()
            return loss

        for step in range(5):  # multiple LBFGS rounds for stability
            opt.step(closure)
        self.W = W.detach().cpu().numpy().astype(np.float32)
        with torch.no_grad():
            final = loss_fn(Zt @ W @ Bt, Y).item()
        print(f"[Linear/map-{loss_type}] LBFGS final loss={final:.4g}")

    def _closed_form_warmstart(self, Zt, Bt, Y):
        import torch

        ZtZ = Zt.T @ Zt
        p = Zt.shape[1]
        I = torch.eye(p, dtype=Zt.dtype)
        # Solve Z'Z M = Z'Y
        rhs = Zt.T @ Y
        M = torch.linalg.solve(ZtZ + self.alpha * I, rhs)
        if _is_orthonormal(Bt.cpu().numpy()):
            return M @ Bt.T
        BBT = Bt @ Bt.T
        K = Bt.shape[0]
        return torch.linalg.solve(BBT + self.alpha * torch.eye(K, dtype=Bt.dtype), Bt @ M.T).T

    # ---- predict ----

    def predict(
        self, cog_features: np.ndarray, stim_encodings: np.ndarray, warn_unseen: bool = False,
    ) -> np.ndarray:
        if warn_unseen:
            self._validate_encodings(stim_encodings)
        Z = self._build_Z(cog_features, stim_encodings)
        return Z @ self.W

    # ---- persistence ----

    def _get_state(self) -> dict:
        return {"W": self.W, "alpha": self.alpha}

    def _set_state(self, state: dict) -> None:
        self.W = state["W"]
        self.alpha = state["alpha"]
