"""`LoadedRun` — one object that turns a trained run directory into a map-predicting function.

Gathers what was previously scattered across the apps: config + basis + predictor, the
adapter's masker and feature names, the training slice of ``predictions.pkl`` (feature ranges,
archetypes and the bootstrap all need it), the optional per-contrast mean-map offset and the
optional atlas-space back-projection.

Effect maps are differences ``Σ_t w_t · pred(profile_t)`` with ``Σ w_t = 0``, so every additive
constant — the basis ``mean_map`` and the mean-map offset μ — cancels exactly. `combine` uses
that to stay in coefficient space until the very last matmul.
"""

from __future__ import annotations

import contextlib
import io as _io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cog2bold.bases.base import BasisTransform
from cog2bold.config import RunConfig
from cog2bold.datasets.registry import create_adapter
from cog2bold.io import load_json, load_pickle
from cog2bold.pipelines.mean_offset import MeanMapOffset
from cog2bold.predictors.base import CoefficientPredictor

_TRAIN_KEYS = (
    "cog_features_train", "stim_encodings_train", "coeffs_gt_train", "maps_gt_train",
    "subject_ids_train", "stim_ids_train", "is_contrast_train", "contrast_names_train",
)


@dataclass
class Condition:
    """A stimulus/contrast the model can be queried at. ``stim`` is empty for cog-only runs."""

    name: str
    stim: np.ndarray


class LoadedRun:
    def __init__(
        self,
        run_dir: Path,
        config: RunConfig,
        basis: BasisTransform,
        predictor: CoefficientPredictor,
        masker,
        feature_names: List[str],
        global_labels: List[str],
        train: Dict[str, np.ndarray],
        mean_offset: MeanMapOffset,
        metrics: Optional[dict] = None,
    ):
        self.run_dir = Path(run_dir)
        self.config = config
        self.basis = basis
        self.predictor = predictor
        self.masker = masker
        self.feature_names = feature_names
        self.global_labels = global_labels
        self.train = train
        self.mean_offset = mean_offset
        self.metrics = metrics or {}
        self.bundle_meta: Optional[dict] = None  # set when loaded from a shareable bundle
        self._decode_matrix = basis.get_decode_matrix()
        self._projector = None
        if config.train_atlas_cfg is not None:
            from cog2bold.utils.atlas_projector import get_atlas_projector

            self._projector = get_atlas_projector(masker, config.train_atlas_cfg.atlas_n_rois)
        if len(feature_names) != len(predictor.cog_mean):
            raise ValueError(
                f"{len(feature_names)} feature names vs predictor width {len(predictor.cog_mean)}"
            )

    # ---- construction ----

    @classmethod
    def from_dir(cls, run_dir, data_root: Optional[Path] = None, masker=None) -> "LoadedRun":
        """Load a trained run (or a bundle, see bundle.py). ``data_root`` overrides the path
        stored at train time."""
        run_dir = Path(run_dir)
        from cog2bold.analysis.effect_atlas.bundle import is_bundle, load_bundle

        if is_bundle(run_dir):
            return load_bundle(run_dir)
        config = RunConfig.load(run_dir / "config.json")
        dc = config.data_config
        adapter = create_adapter(
            dc.adapter_name,
            data_root=Path(data_root) if data_root else config.data_root,
            cog_base_cols=dc.cog_base_cols,
            **(dc.adapter_kwargs or {}),
        )
        preds = load_pickle(run_dir / "predictions.pkl")
        offset = MeanMapOffset(False)
        if (run_dir / "mean_maps_by_contrast.pkl").exists():
            offset.load_state(load_pickle(run_dir / "mean_maps_by_contrast.pkl"))
        metrics = load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
        return cls(
            run_dir=run_dir,
            config=config,
            basis=BasisTransform.load(run_dir / "basis.pkl"),
            predictor=CoefficientPredictor.load(run_dir / "predictor.pkl"),
            masker=masker if masker is not None else adapter.get_masker(),
            feature_names=list(adapter.cog_feature_names),
            global_labels=list(adapter.global_labels),
            train={k: np.asarray(preds[k]) for k in _TRAIN_KEYS if k in preds},
            mean_offset=offset,
            metrics=metrics,
        )

    # ---- descriptors ----

    @property
    def cog_only(self) -> bool:
        return bool(self.config.predictor_config.cog_only)

    @property
    def is_linear(self) -> bool:
        return self.config.predictor_config.model_type == "linear"

    @property
    def cog_mean(self) -> np.ndarray:
        return np.asarray(self.predictor.cog_mean, dtype=np.float64)

    @property
    def cog_std(self) -> np.ndarray:
        return np.asarray(self.predictor.cog_std, dtype=np.float64)

    @property
    def mask_img(self):
        return self.masker.mask_img_

    @property
    def n_voxels(self) -> int:
        return int(np.asarray(self.mask_img.dataobj).astype(bool).sum())

    @property
    def title(self) -> str:
        if self.bundle_meta:
            return self.bundle_meta["title"]
        parts = self.run_dir.parts
        return "/".join(parts[-3:])

    def train_row_names(self) -> np.ndarray:
        """Readable stim/contrast name per training row (matches the pipeline's convention)."""
        t = self.train
        return np.array([
            str(c) if bool(isc) else self.global_labels[int(s)]
            for c, isc, s in zip(t["contrast_names_train"], t["is_contrast_train"], t["stim_ids_train"])
        ])

    def subject_profiles(self) -> np.ndarray:
        """One raw cognitive vector per training subject (first occurrence)."""
        _, first = np.unique(self.train["subject_ids_train"], return_index=True)
        return np.asarray(self.train["cog_features_train"][np.sort(first)], dtype=np.float64)

    def conditions(self) -> List[Condition]:
        if self.cog_only:
            sc = self.config.data_config.single_contrast
            name = sc if isinstance(sc, str) else ("+".join(sc) if sc else "map")
            return [Condition(name=name, stim=np.zeros(0, dtype=np.float32))]
        enc = np.asarray(self.train["stim_encodings_train"])
        names = self.train_row_names()
        _, first = np.unique(enc, axis=0, return_index=True)
        return [Condition(name=str(names[i]), stim=enc[i]) for i in np.sort(first)]

    # ---- prediction ----

    def _stim_block(self, stim: np.ndarray, n: int) -> np.ndarray:
        if self.cog_only:
            return np.zeros((n, 0), dtype=np.float32)
        return np.tile(np.asarray(stim, dtype=np.float32), (n, 1))

    def predict_coeffs(self, profiles: np.ndarray, condition: Condition,
                       predictor: Optional[CoefficientPredictor] = None) -> np.ndarray:
        profiles = np.atleast_2d(np.asarray(profiles, dtype=np.float64))
        p = predictor or self.predictor
        with contextlib.redirect_stdout(_io.StringIO()):
            return np.asarray(p.predict(profiles, self._stim_block(condition.stim, len(profiles))))

    def _to_voxels(self, maps: np.ndarray) -> np.ndarray:
        """Training-space vectors → voxel vectors (inflates Schaefer ROI values if needed)."""
        if self._projector is not None:
            return self._projector.roi_to_voxel_batch(maps)
        return maps

    def combine(self, weights: np.ndarray, profiles: np.ndarray, condition: Condition,
                predictor: Optional[CoefficientPredictor] = None) -> np.ndarray:
        """Weighted map combinations ``weights @ pred(profiles)`` → (n_combos, n_voxels).

        Every row of ``weights`` must sum to zero (a contrast between profiles), which lets
        all additive constants drop out.
        """
        weights = np.atleast_2d(np.asarray(weights, dtype=np.float64))
        if not np.allclose(weights.sum(axis=1), 0.0, atol=1e-9):
            raise ValueError("combination weights must sum to zero")
        coeffs = self.predict_coeffs(profiles, condition, predictor).astype(np.float64)
        if self._decode_matrix is not None:
            out = (weights @ coeffs) @ np.asarray(self._decode_matrix, dtype=np.float64)
        else:
            out = weights @ np.asarray(self.basis.decode_batch(coeffs), dtype=np.float64)
        return self._to_voxels(out.astype(np.float32))

    def predict_maps(self, profiles: np.ndarray, condition: Condition) -> np.ndarray:
        """Absolute predicted maps (mean-map offset restored) → (n, n_voxels)."""
        coeffs = self.predict_coeffs(profiles, condition)
        maps = np.asarray(self.basis.decode_batch(coeffs))
        if self.mean_offset.enabled and self.mean_offset.has(condition.name):
            maps = maps + self.mean_offset.mean_by_contrast[condition.name]
        return self._to_voxels(maps.astype(np.float32))

    def to_img(self, vec: np.ndarray):
        return self.masker.inverse_transform(np.asarray(vec, dtype=np.float32))

    # ---- validity summary (for the report header) ----

    def validity_summary(self) -> Dict[str, object]:
        """Held-out metrics + the latest LTO evaluation found in the run dir."""
        out: Dict[str, object] = {}
        for k in ("voxel_correlation_mean", "explained_variance_r2"):
            if k in self.metrics:
                out[k] = self.metrics[k]
        ltos = sorted(self.run_dir.glob("lto_eval_*/lto_results_summary.json"),
                      key=lambda p: p.stat().st_mtime)  # most recent last (names aren't time-ordered)
        if ltos:
            s = load_json(ltos[-1])
            out["lto_eval"] = ltos[-1].parent.name
            out["lto_accuracy"] = s.get("overall", {}).get("accuracy")
            out["lto_pairs"] = s.get("overall", {}).get("total_pairs")
            null = s.get("null_model")
            if isinstance(null, dict):
                out["lto_null"] = {k: v for k, v in null.items()
                                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return out
