"""Shared base for held-out evaluators (Leave-Two-Out and Leave-One-Out).

Holds everything both evaluators need identically: adapter loading, correlation
spaces (incl. atlas projection), atlas-space training, per-fold model training, and
single-sample prediction. The subclasses add their own fold/pair logic, statistics,
and result serialisation.

Reads its configuration from explicit attributes the subclass sets before calling
``load_dataset`` — `self.correlation_configs`, `self.train_atlas_cfg`,
`self.p_threshold`, `self.p_threshold_mode` — rather than a specific Args dataclass,
so both `LTOArgs` and `LOOArgs` fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

from cog2bold.bases import create_basis
from cog2bold.config import CorrelationConfig, RunConfig
from cog2bold.datasets import (
    ActivationMapSample,
    DatasetAdapter,
    create_adapter,
    filter_samples_by_contrast,
)
from cog2bold.predictors import create_predictor
from cog2bold.pipelines.mean_offset import MeanMapOffset
from cog2bold.utils.atlas_projector import AtlasProjector
from cog2bold.utils.cog_features import cog_stats_from_unique_pairs
from cog2bold.utils.thresholding import threshold_zmap_attenuate, threshold_zmap_zero
from cog2bold.workspace import cache_root


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened maps; 0.0 for degenerate input."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(r) if np.isfinite(r) else 0.0


# Similarity metrics shared by all held-out evaluators. Both are reduced to a single
# "is this match better?" question via metric_is_better / metric_delta.
EVAL_METRICS = ["correlation", "mse"]


def metric_is_better(metric: str, a: float, b: float) -> bool:
    """Is value `a` a *better* match than `b` under `metric`?

    correlation -> higher is better (a > b);  mse -> lower is better (a < b).
    """
    return a > b if metric == "correlation" else a < b


def metric_delta(metric: str, model: float, baseline: float) -> float:
    """Signed improvement of model over baseline (positive = model is better)."""
    return (model - baseline) if metric == "correlation" else (baseline - model)


class BaseLeaveOutEvaluator:
    """Common machinery for held-out evaluators.

    Subclasses must set, before calling ``load_dataset()``:
        self.correlation_configs : List[CorrelationConfig]
        self.train_atlas_cfg     : Optional[CorrelationConfig]
        self.p_threshold         : Optional[float]
        self.p_threshold_mode    : str  ("zero" | "attenuate")
    """

    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.adapter: DatasetAdapter = create_adapter(
            run_config.data_config.adapter_name,
            data_root=run_config.data_root,
            cog_base_cols=run_config.data_config.cog_base_cols,
            **(run_config.data_config.adapter_kwargs or {}),
        )
        self.samples: List[ActivationMapSample] = []
        self.cog_std: Optional[np.ndarray] = None
        # per-fold mean-map offset (reset in _train_on_subset); disabled placeholder by default
        self._mean_offset = MeanMapOffset(False)
        # Space the *metrics* are computed in — "map" (absolute) or "deviation" (minus the
        # per-contrast training mean map). Independent of how the model was trained: a
        # map-space run can be scored in deviation space and vice versa. Subclasses that
        # support it set this from their Args before load_dataset().
        self.score_space: str = "map"
        self._score_offset = MeanMapOffset(False)

        # Atlas state (populated in load_dataset)
        self._atlas_projectors: Dict[int, AtlasProjector] = {}
        self._roi_subset_cache: Dict[str, List[int]] = {}
        self._train_atlas_projector: Optional[AtlasProjector] = None
        self._precomputed_roi_maps: Optional[Dict[int, np.ndarray]] = None

    # ---- correlation-config helpers ----

    def _resolve_correlation_configs(
        self, explicit: Optional[List[CorrelationConfig]],
    ) -> List[CorrelationConfig]:
        if explicit is not None:
            return explicit
        if self.run_config.eval_correlation_configs:
            return self.run_config.eval_correlation_configs
        return [CorrelationConfig(name="voxel", space=0)]

    def _enforce_correlation_config_compatibility(self) -> None:
        """When training in atlas space, drop correlation configs that can't apply.

        With ``train_atlas_cfg`` set, predictions are ROI vectors of size
        ``atlas_n_rois``; voxel-space (space=0) or a different atlas size would crash
        or compare maps in incompatible spaces. Filter those out; raise if none remain.
        """
        if self.train_atlas_cfg is None:
            return
        compatible: List[CorrelationConfig] = []
        rejected: List[str] = []
        for cfg in self.correlation_configs:
            if cfg.space == 0:
                rejected.append(
                    f"  - dropping {cfg.name!r}: voxel-space correlation is incompatible "
                    f"with train_atlas_cfg={self.train_atlas_cfg.name!r}"
                )
                continue
            if cfg.atlas_n_rois != self.train_atlas_cfg.atlas_n_rois:
                rejected.append(
                    f"  - dropping {cfg.name!r}: atlas_n_rois={cfg.atlas_n_rois} "
                    f"!= train_atlas_cfg.atlas_n_rois={self.train_atlas_cfg.atlas_n_rois}"
                )
                continue
            compatible.append(cfg)
        if not compatible:
            raise ValueError(
                "No correlation configs are compatible with atlas-space training.\n"
                f"train_atlas_cfg = {self.train_atlas_cfg.to_dict()}\n"
                + "\n".join(rejected)
                + "\nFix: set train_atlas_cfg=None, or supply correlation_configs "
                "with matching atlas_n_rois (and space in {1, 2})."
            )
        if rejected:
            print("[eval] filtering correlation configs for atlas-space training:")
            for line in rejected:
                print(line)
        self.correlation_configs = compatible

    # ---- data loading + atlas projection ----

    def load_dataset(self) -> List[ActivationMapSample]:
        dc = self.run_config.data_config
        ss_filter = None
        if dc.subject_session_filter is not None:
            ssf_path = Path(dc.subject_session_filter)
            if not ssf_path.is_absolute():
                ssf_path = self.run_config.data_root / ssf_path
            data = json.loads(ssf_path.read_text())
            ss_filter = {(p["subject"], p["session"]) for p in data["included"]}

        cache_dir = cache_root() / self.run_config.data_config.adapter_name
        samples = self.adapter.materialize(
            cog_base_cols=dc.cog_base_cols,
            stim_encoding_mode=dc.stim_encoding,
            contrast_pairs=dc.contrast_pairs,
            subject_session_filter=ss_filter,
            cache_dir=cache_dir,
        )
        if dc.single_contrast is not None:
            samples = filter_samples_by_contrast(samples, dc.single_contrast, self.adapter)
        self.samples = samples
        _, self.cog_std = cog_stats_from_unique_pairs(samples)

        # Atlas projectors for evaluation correlation spaces (+ training projection)
        atlas_sizes = sorted({c.atlas_n_rois for c in self.correlation_configs if c.space in {1, 2}})
        if self.train_atlas_cfg is not None and self.train_atlas_cfg.atlas_n_rois not in atlas_sizes:
            atlas_sizes.append(self.train_atlas_cfg.atlas_n_rois)
        if atlas_sizes:
            masker = self.adapter.get_masker()
            for n_rois in atlas_sizes:
                self._atlas_projectors[n_rois] = AtlasProjector(masker=masker, n_rois=n_rois)
            for cfg in self.correlation_configs:
                if cfg.space == 2 and cfg.roi_name_patterns:
                    proj = self._atlas_projectors[cfg.atlas_n_rois]
                    self._roi_subset_cache[cfg.name] = proj.roi_indices_matching(cfg.roi_name_patterns)
                    print(
                        f"  ROI subset {cfg.name!r}: {len(self._roi_subset_cache[cfg.name])} "
                        f"of {len(proj.labels)} ROIs match {cfg.roi_name_patterns}"
                    )

        if self.train_atlas_cfg is not None:
            self._train_atlas_projector = self._atlas_projectors[self.train_atlas_cfg.atlas_n_rois]
            print(f"[eval] pre-projecting all maps to {self.train_atlas_cfg.name}")
            self._precomputed_roi_maps = {
                i: self._train_atlas_projector.voxel_to_roi(samples[i].map)
                for i in range(len(samples))
            }
        return samples

    def _get_map(self, idx: int) -> np.ndarray:
        if self._precomputed_roi_maps is not None:
            return self._precomputed_roi_maps[idx]
        return self.samples[idx].map

    def _project_pair(self, m1: np.ndarray, m2: np.ndarray, cfg: CorrelationConfig):
        """Project two maps into the correlation-config's space → (v1, v2)."""
        # Maps already in atlas space (training projection): compare directly
        if self._precomputed_roi_maps is not None:
            v1, v2 = m1, m2
            if cfg.space == 2:
                idx = self._roi_subset_cache.get(cfg.name)
                if idx:
                    v1, v2 = v1[idx], v2[idx]
            return v1, v2
        if cfg.space == 0:
            return m1, m2
        proj = self._atlas_projectors[cfg.atlas_n_rois]
        v1 = proj.voxel_to_roi(m1)
        v2 = proj.voxel_to_roi(m2)
        if cfg.space == 2:
            idx = self._roi_subset_cache.get(cfg.name, [])
            if idx:
                v1, v2 = v1[idx], v2[idx]
        return v1, v2

    def _correlation_for(self, m1: np.ndarray, m2: np.ndarray, cfg: CorrelationConfig) -> float:
        v1, v2 = self._project_pair(m1, m2, cfg)
        return safe_corr(v1, v2)

    def _metric_value(self, m1: np.ndarray, m2: np.ndarray, cfg: CorrelationConfig,
                      metric: str = "correlation") -> float:
        """Similarity between two maps in `cfg`'s space, under the given metric.

        correlation -> Pearson r (higher = more similar)
        mse         -> mean squared error (lower = more similar)
        """
        v1, v2 = self._project_pair(m1, m2, cfg)
        if metric == "correlation":
            return safe_corr(v1, v2)
        if metric == "mse":
            a = np.asarray(v1).ravel().astype(np.float64)
            b = np.asarray(v2).ravel().astype(np.float64)
            return float(np.mean((a - b) ** 2))
        raise ValueError(f"Unknown metric {metric!r}")

    # ---- contrast grouping ----

    def _group_by_contrast(self) -> Dict[str, List[int]]:
        """Map readable stimulus/contrast name -> list of sample indices."""
        groups: Dict[Tuple, List[int]] = {}
        for i, s in enumerate(self.samples):
            groups.setdefault(tuple(s.stim_encoding.tolist()), []).append(i)
        readable: Dict[str, List[int]] = {}
        for key, idx_list in groups.items():
            sample0 = self.samples[idx_list[0]]
            if sample0.is_contrast:
                name = sample0.contrast_name
            else:
                stim_idx = int(np.argmax(np.abs(np.asarray(key))))
                name = self.adapter.global_labels[stim_idx]
            readable[name] = idx_list
        return readable

    def _contrast_name_of(self, idx: int) -> str:
        """Readable stimulus/contrast name for a single sample index."""
        return self._contrast_name_of_sample(self.samples[idx])

    def _contrast_name_of_sample(self, s: ActivationMapSample) -> str:
        """Readable stimulus/contrast name for a sample object."""
        if s.is_contrast:
            return s.contrast_name
        stim_idx = int(np.argmax(np.abs(s.stim_encoding)))
        return self.adapter.global_labels[stim_idx]

    # ---- per-fold training + prediction ----

    def _train_on_subset(
        self,
        train_indices: List[int],
        cog_override: Optional[Dict[int, np.ndarray]] = None,
    ):
        train_maps = np.asarray([self._get_map(i) for i in train_indices])
        train_cog = np.asarray([
            cog_override[i] if (cog_override is not None and i in cog_override)
            else self.samples[i].cog_features
            for i in train_indices
        ])
        train_stim = np.asarray([self.samples[i].stim_encoding for i in train_indices])
        if self.run_config.predictor_config.cog_only:
            train_stim = np.zeros((train_stim.shape[0], 0), dtype=train_stim.dtype)

        # Optional mean-map offset: fit basis/predictor on deviations from the per-contrast
        # training mean map; the mean is added back in _predict. Identity when the flag is off.
        train_contrasts = [self._contrast_name_of(i) for i in train_indices]
        self._mean_offset = MeanMapOffset(
            getattr(self.run_config, "mean_map_offset", False)
        ).fit(train_maps, train_contrasts)
        # Scoring-space μ — fitted on the SAME held-out-excluding training maps, so nothing
        # about the two evaluated samples leaks into the μ they are later compared through.
        # Deliberately independent of mean_map_offset: the two knobs answer different
        # questions (what the model models vs. what the metric measures).
        self._score_offset = MeanMapOffset(self.score_space == "deviation").fit(
            train_maps, train_contrasts
        )
        train_maps = self._mean_offset.center(train_maps, train_contrasts)

        basis = create_basis(self.run_config.basis_config)
        kwargs = {}
        if basis.requires_supervision:
            kwargs.update(cog_features=train_cog, stim_encodings=train_stim)
        if self.run_config.basis_config.basis_type == "atlas" or (
            self.run_config.basis_config.basis_type == "hybrid"
            and self.run_config.basis_config.extra_params.get("unsupervised_method") == "atlas"
        ):
            kwargs["masker"] = self.adapter.get_masker()
        basis.fit(train_maps, **kwargs)

        if basis.requires_supervision:
            train_coeffs = basis.encode_batch(
                train_maps, cog_features=train_cog, stim_encodings=train_stim
            )
        else:
            train_coeffs = basis.encode_batch(train_maps)

        self.run_config.predictor_config.output_dim = train_coeffs.shape[1]
        predictor = create_predictor(self.run_config.predictor_config)

        if cog_override is None:
            cog_mean, cog_std = cog_stats_from_unique_pairs([self.samples[i] for i in train_indices])
        else:
            cog_mean = train_cog.mean(axis=0)
            cog_std = np.maximum(train_cog.std(axis=0), 1e-8)
        fit_kwargs = dict(
            cog_features=train_cog,
            stim_encodings=train_stim,
            target_coeffs=train_coeffs,
            cog_mean=cog_mean,
            cog_std=cog_std,
        )
        if self.run_config.predictor_config.loss_mode == "map":
            fit_kwargs.update(target_maps=train_maps, basis_object=basis)
        predictor.fit(**fit_kwargs)
        return basis, predictor

    def _predict(self, sample: ActivationMapSample, basis, predictor) -> np.ndarray:
        cog = sample.cog_features[np.newaxis, :]
        stim = sample.stim_encoding[np.newaxis, :]
        if self.run_config.predictor_config.cog_only:
            stim = np.zeros((1, 0), dtype=stim.dtype)
        coeffs = predictor.predict(cog, stim, warn_unseen=False)
        pred = basis.decode(coeffs[0])
        # add the per-contrast training mean back (identity unless mean_map_offset is on)
        return self._mean_offset.restore(pred, self._contrast_name_of_sample(sample))

    def _to_score_space(self, m: np.ndarray, contrast_name: str) -> np.ndarray:
        """Project a full map into the space the metrics are computed in.

        Identity for ``score_space="map"``. For ``"deviation"`` it subtracts the per-contrast
        training mean map, which must be applied to BOTH sides of every comparison for the
        result to mean anything.
        """
        return self._score_offset.deviate(m, contrast_name)

    def _score_mu_available(self, contrast_name: str) -> bool:
        """Can this contrast be scored? False only when deviation scoring has no μ for it."""
        return self._score_offset.has(contrast_name)

    def _apply_threshold(self, m: np.ndarray) -> np.ndarray:
        if self.p_threshold is None:
            return m
        if self.p_threshold_mode == "zero":
            return threshold_zmap_zero(m, self.p_threshold)
        return threshold_zmap_attenuate(m, self.p_threshold)
