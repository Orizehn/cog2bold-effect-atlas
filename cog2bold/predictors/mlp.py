"""MLPPredictor — PyTorch MLP. Supports coefficient and map loss modes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cog2bold.config import PredictorConfig
from cog2bold.predictors.base import CoefficientPredictor


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPPredictor(CoefficientPredictor):
    def __init__(self, config: PredictorConfig):
        super().__init__(config)
        self.model: Optional[nn.Module] = None
        self.device = _select_device()

    # ---- model build ----

    def _build_model(self) -> nn.Module:
        layers = []
        in_dim = self.config.input_dim
        for h in self.config.hidden_dims or [128]:
            layers.append(nn.Linear(in_dim, h))
            if self.config.activation == "gelu":
                layers.append(nn.GELU())
            elif self.config.activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.config.output_dim))
        return nn.Sequential(*layers)

    def _build_Z_torch(self, cog: np.ndarray, stim: np.ndarray) -> torch.Tensor:
        cog_norm = (cog - self.cog_mean) / self.cog_std
        if self.config.cog_only:
            return torch.from_numpy(cog_norm.astype(np.float32))
        Z = np.concatenate([cog_norm, stim], axis=1).astype(np.float32)
        return torch.from_numpy(Z)

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
    ) -> "MLPPredictor":
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        self._set_normalisation(cog_features, cog_mean, cog_std)
        self._record_training_encodings(stim_encodings)

        if self.config.cog_only:
            self.config.input_dim = cog_features.shape[1]
        else:
            self.config.input_dim = cog_features.shape[1] + stim_encodings.shape[1]

        self.model = self._build_model().to(self.device)

        Z_train = self._build_Z_torch(cog_features, stim_encodings).to(self.device)

        loss_mode = self.config.loss_mode
        map_loss_type = self.config.map_loss_type

        if loss_mode == "map":
            if basis_object is None or target_maps is None:
                raise ValueError("loss_mode='map' requires basis_object and target_maps")
            W_decode = basis_object.get_decode_matrix()
            if W_decode is None:
                raise ValueError(
                    f"Basis {type(basis_object).__name__} doesn't expose a decode matrix; "
                    f"map loss not supported."
                )
            W_torch = torch.from_numpy(W_decode.astype(np.float32)).to(self.device)
            mu_flat = None
            if getattr(basis_object, "mean_map", None) is not None:
                mu_flat = torch.from_numpy(
                    np.asarray(basis_object.mean_map, dtype=np.float32).flatten()
                ).to(self.device)
            n_train = target_maps.shape[0]
            Y_train = torch.from_numpy(
                target_maps.reshape(n_train, -1).astype(np.float32)
            ).to(self.device)
        else:
            W_torch = mu_flat = None
            Y_train = torch.from_numpy(target_coeffs.astype(np.float32)).to(self.device)

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        n_train = Z_train.shape[0]
        bs = max(1, self.config.batch_size)
        for epoch in range(self.config.n_epochs):
            self.model.train()
            perm = torch.randperm(n_train, device=self.device)
            running = 0.0
            for i in range(0, n_train, bs):
                idx = perm[i : i + bs]
                z_batch = Z_train[idx]
                y_batch = Y_train[idx]

                pred_coeffs = self.model(z_batch)
                if loss_mode == "map":
                    pred_maps = pred_coeffs @ W_torch
                    if mu_flat is not None:
                        pred_maps = pred_maps + mu_flat
                    loss = _map_loss(pred_maps, y_batch, map_loss_type)
                else:
                    loss = (pred_coeffs - y_batch).pow(2).mean()

                opt.zero_grad()
                loss.backward()
                clip = getattr(self.config, "grad_clip_norm", None)
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                opt.step()
                running += float(loss.item()) * len(idx)
            running /= n_train
            if epoch % max(1, self.config.n_epochs // 10) == 0:
                print(f"  epoch {epoch:3d}/{self.config.n_epochs}  loss={running:.4g}")

        self.is_fitted = True
        return self

    # ---- predict ----

    def predict(
        self, cog_features: np.ndarray, stim_encodings: np.ndarray, warn_unseen: bool = False,
    ) -> np.ndarray:
        if warn_unseen:
            self._validate_encodings(stim_encodings)
        self.model.eval()
        with torch.no_grad():
            Z = self._build_Z_torch(cog_features, stim_encodings).to(self.device)
            pred = self.model(Z)
        return pred.cpu().numpy()

    # ---- persistence ----

    def _get_state(self) -> dict:
        return {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "input_dim": self.config.input_dim,
        }

    def _set_state(self, state: dict) -> None:
        self.config.input_dim = state["input_dim"]
        if state["model_state_dict"] is not None:
            self.model = self._build_model().to(self.device)
            self.model.load_state_dict(state["model_state_dict"])


def _map_loss(pred: torch.Tensor, true: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "l2":
        return (pred - true).pow(2).mean()
    if kind == "l1":
        return (pred - true).abs().mean()
    # correlation: vectorised over the batch
    pc = pred - pred.mean(dim=1, keepdim=True)
    tc = true - true.mean(dim=1, keepdim=True)
    num = (pc * tc).sum(dim=1)
    den = pc.norm(dim=1) * tc.norm(dim=1) + 1e-8
    return -(num / den).mean()
