"""Training pipeline.

Bug fixes vs the legacy `basis_framework/training.py`:
  - cog_mean/std computed train-only (#3)
  - compute_contrast uses _get_map() consistently for atlas-space training (#14)
  - decode_batch uses get_decode_matrix() when available, no Python loops in inference
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cog2bold.bases import create_basis
from cog2bold.bases.base import BasisTransform
from cog2bold.config import RunConfig
from cog2bold.datasets import ActivationMapSample, DatasetAdapter, filter_samples_by_contrast
from cog2bold.io import save_json, save_pickle
from cog2bold.metrics import compute_contrast, compute_metrics
from cog2bold.pipelines.mean_offset import MeanMapOffset
from cog2bold.predictors import create_predictor
from cog2bold.predictors.base import CoefficientPredictor
from cog2bold.utils.atlas_projector import AtlasProjector
from cog2bold.utils.cog_features import cog_stats_from_unique_pairs
from cog2bold.utils.reproducibility import seed_everything
from cog2bold.workspace import cache_root


class TrainingPipeline:
    """Run a full train + evaluate + save cycle."""

    def __init__(self, config: RunConfig, adapter: DatasetAdapter):
        self.config = config
        self.adapter = adapter
        self.samples: List[ActivationMapSample] = []
        self.basis: Optional[BasisTransform] = None
        self.predictor: Optional[CoefficientPredictor] = None
        self._atlas_projector: Optional[AtlasProjector] = None
        self._atlas_maps: Optional[Dict[int, np.ndarray]] = None
        # per-contrast mean-map offset (fit in _fit_basis); disabled placeholder until then
        self._mean_offset = MeanMapOffset(getattr(config, "mean_map_offset", False))
        self.results: Dict[str, object] = {}

    @property
    def _cog_only(self) -> bool:
        return bool(self.config.predictor_config.cog_only)

    def _maybe_zero_stim(self, stim_encodings: np.ndarray) -> np.ndarray:
        if self._cog_only:
            return np.zeros((stim_encodings.shape[0], 0), dtype=stim_encodings.dtype)
        return stim_encodings

    def run(self) -> Dict:
        seed_everything(self.config.random_seed)

        print("=" * 70)
        print(f"[Pipeline] run_dir = {self.config.run_dir}")
        print("=" * 70)

        print("\n[1/7] Loading samples via adapter...")
        self._load_samples()

        if self.config.train_atlas_cfg is not None:
            print("\n[~] Pre-projecting maps to atlas space (training in ROI space)...")
            self._setup_atlas_projection()

        print("\n[2/7] Splitting train/test by subject...")
        train_idx, test_idx = self._split_by_subject()
        print(f"  train: {len(train_idx)}  test: {len(test_idx)}")

        print("\n[3/7] Fitting basis...")
        self.basis = self._fit_basis(train_idx)

        print("\n[4/7] Encoding maps to coefficients...")
        coeffs_train, coeffs_test = self._encode_maps(train_idx, test_idx)
        print(f"  coefficient shape: {coeffs_train.shape}")

        print("\n[5/7] Training predictor...")
        self.predictor = self._train_predictor(train_idx, test_idx, coeffs_train, coeffs_test)

        print("\n[6/7] Evaluating...")
        metrics = self._evaluate(test_idx, coeffs_test)

        print("\n[7/7] Saving outputs...")
        self._save_outputs(train_idx, test_idx, coeffs_train, coeffs_test, metrics)

        print("\n" + "=" * 70)
        print(f"[Pipeline] DONE. Results at {self.config.run_dir}")
        print("=" * 70)
        return self.results

    # ---- step 1 ----

    def _load_samples(self) -> None:
        ss_filter = None
        ssf_path = self.config.data_config.subject_session_filter
        if ssf_path is not None:
            import json

            ssf_path = Path(ssf_path)
            if not ssf_path.is_absolute():
                ssf_path = self.config.data_root / ssf_path
            data = json.loads(ssf_path.read_text())
            ss_filter = {(p["subject"], p["session"]) for p in data["included"]}
            print(f"  using filter: {len(ss_filter)} (subject, session) pairs from {ssf_path.name}")

        cache_dir = cache_root() / self.config.data_config.adapter_name
        samples = self.adapter.materialize(
            cog_base_cols=self.config.data_config.cog_base_cols,
            stim_encoding_mode=self.config.data_config.stim_encoding,
            contrast_pairs=self.config.data_config.contrast_pairs,
            subject_session_filter=ss_filter,
            cache_dir=cache_dir,
            force_recompute=self.config.data_config.force_recompute,
        )
        if self.config.data_config.single_contrast is not None:
            samples = filter_samples_by_contrast(
                samples, self.config.data_config.single_contrast, self.adapter,
            )
        self.samples = samples
        print(f"  total samples: {len(self.samples)}")

    # ---- atlas-space training ----

    def _setup_atlas_projection(self) -> None:
        cfg = self.config.train_atlas_cfg
        masker = self.adapter.get_masker()
        self._atlas_projector = AtlasProjector(masker=masker, n_rois=cfg.atlas_n_rois)
        n = len(self.samples)
        print(f"  projecting {n} maps -> Schaefer-{cfg.atlas_n_rois} ROI space")
        self._atlas_maps = {
            i: self._atlas_projector.voxel_to_roi(self.samples[i].map) for i in range(n)
        }
        first = next(iter(self._atlas_maps.values()))
        print(f"  pre-projected map shape: {first.shape}")

    def _get_map(self, idx: int) -> np.ndarray:
        if self._atlas_maps is not None:
            return self._atlas_maps[idx]
        return self.samples[idx].map

    # ---- step 2 ----

    def _split_by_subject(self) -> Tuple[np.ndarray, np.ndarray]:
        subjects = np.asarray([s.subject_id for s in self.samples])
        unique = np.unique(subjects)
        rng = np.random.RandomState(self.config.random_seed)
        rng.shuffle(unique)
        n_train_subj = int(round(len(unique) * self.config.train_fraction))
        train_subjects = set(unique[:n_train_subj])
        train_idx, test_idx = [], []
        for i, s in enumerate(self.samples):
            (train_idx if s.subject_id in train_subjects else test_idx).append(i)
        return np.asarray(train_idx), np.asarray(test_idx)

    # ---- step 3 ----

    def _contrast_names(self, indices) -> List[str]:
        """Readable stimulus/contrast name per sample index (for the mean-map offset)."""
        out = []
        for i in indices:
            s = self.samples[i]
            if s.is_contrast:
                out.append(s.contrast_name)
            else:
                out.append(self.adapter.global_labels[int(np.argmax(np.abs(s.stim_encoding)))])
        return out

    def _collect(self, indices: np.ndarray):
        maps = np.asarray([self._get_map(i) for i in indices])
        if self._mean_offset.enabled:
            maps = self._mean_offset.center(maps, self._contrast_names(indices))
        cog = np.asarray([self.samples[i].cog_features for i in indices])
        stim = self._maybe_zero_stim(np.asarray([self.samples[i].stim_encoding for i in indices]))
        return maps, cog, stim

    def _fit_basis(self, train_idx: np.ndarray) -> BasisTransform:
        # Fit the mean-map offset on RAW train maps first, so subsequent _collect() calls
        # (here and in encode/predictor steps) center consistently. No-op when disabled.
        if self._mean_offset.enabled:
            raw_train_maps = np.asarray([self._get_map(i) for i in train_idx])
            self._mean_offset.fit(raw_train_maps, self._contrast_names(train_idx))
        train_maps, cog_train, stim_train = self._collect(train_idx)
        basis = create_basis(self.config.basis_config)
        kwargs = {}
        if basis.requires_supervision:
            kwargs.update({"cog_features": cog_train, "stim_encodings": stim_train})
        if self.config.basis_config.basis_type == "atlas":
            kwargs["masker"] = self.adapter.get_masker()
        if (
            self.config.basis_config.basis_type == "hybrid"
            and self.config.basis_config.extra_params.get("unsupervised_method") == "atlas"
        ):
            kwargs["masker"] = self.adapter.get_masker()
        basis.fit(train_maps, **kwargs)
        return basis

    # ---- step 4 ----

    def _encode_maps(self, train_idx, test_idx):
        train_maps, cog_train, stim_train = self._collect(train_idx)
        test_maps, cog_test, stim_test = self._collect(test_idx)
        if self.basis.requires_supervision:
            coeffs_train = self.basis.encode_batch(
                train_maps, cog_features=cog_train, stim_encodings=stim_train,
            )
            coeffs_test = self.basis.encode_batch(
                test_maps, cog_features=cog_test, stim_encodings=stim_test,
            )
        else:
            coeffs_train = self.basis.encode_batch(train_maps)
            coeffs_test = self.basis.encode_batch(test_maps)
        return coeffs_train, coeffs_test

    # ---- step 5 ----

    def _train_predictor(self, train_idx, test_idx, coeffs_train, coeffs_test):
        train_maps, cog_train, stim_train = self._collect(train_idx)
        test_maps, cog_test, stim_test = self._collect(test_idx)

        # TRAIN-ONLY cog stats (bug fix #3)
        cog_mean, cog_std = cog_stats_from_unique_pairs([self.samples[i] for i in train_idx])

        self.config.predictor_config.output_dim = coeffs_train.shape[1]
        predictor = create_predictor(self.config.predictor_config)

        kwargs = dict(
            cog_features=cog_train,
            stim_encodings=stim_train,
            target_coeffs=coeffs_train,
            cog_mean=cog_mean,
            cog_std=cog_std,
        )
        if self.config.predictor_config.loss_mode == "map":
            kwargs.update(target_maps=train_maps, basis_object=self.basis)
        kwargs["val_data"] = (cog_test, stim_test, coeffs_test)
        predictor.fit(**kwargs)
        return predictor

    # ---- step 6 ----

    def _evaluate(self, test_idx, coeffs_test_gt):
        cog_test = np.asarray([self.samples[i].cog_features for i in test_idx])
        stim_test = self._maybe_zero_stim(
            np.asarray([self.samples[i].stim_encoding for i in test_idx])
        )
        coeffs_pred = self.predictor.predict(cog_test, stim_test, warn_unseen=False)
        maps_test_gt = np.asarray([self._get_map(i) for i in test_idx])
        maps_test_pred = self.basis.decode_batch(coeffs_pred)
        # add the per-contrast training mean back so predictions are full maps (no-op if disabled)
        if self._mean_offset.enabled:
            tc = self._contrast_names(test_idx)
            maps_test_pred = np.asarray(
                [self._mean_offset.restore(m, c) for m, c in zip(maps_test_pred, tc)]
            )

        metrics = compute_metrics(
            maps_test_gt, maps_test_pred,
            coeffs_gt=coeffs_test_gt, coeffs_pred=coeffs_pred,
        )

        # Stash raw predicted/coefficient arrays for save_outputs
        self._test_artifacts = {
            "coeffs_pred": coeffs_pred,
            "maps_pred": maps_test_pred,
        }

        if self.config.contrast_pairs:
            print(f"  computing {len(self.config.contrast_pairs)} contrasts...")
            metrics["contrasts"] = {}
            label2id = self.adapter.label2id
            for stim_a, stim_b in self.config.contrast_pairs:
                gt, pred = compute_contrast(
                    samples=self.samples,
                    basis=self.basis,
                    predictor=self.predictor,
                    test_idx=test_idx,
                    stim_a=stim_a,
                    stim_b=stim_b,
                    label2id=label2id,
                    map_lookup=self._get_map,
                )
                if gt is not None:
                    metrics["contrasts"][f"{stim_a}_vs_{stim_b}"] = {
                        "map_gt": gt,
                        "map_pred": pred,
                    }

        print(f"\n  voxel-corr (mean): {metrics['voxel_correlation_mean']:.3f}")
        print(f"  R²:                {metrics['explained_variance_r2']:.3f}")
        print(f"  map MSE:           {metrics['map_mse']:.4g}")
        return metrics

    # ---- step 7 ----

    def _save_outputs(self, train_idx, test_idx, coeffs_train, coeffs_test, metrics) -> None:
        run_dir = Path(self.config.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self.config.save(run_dir / "config.json")
        self.basis.save(run_dir / "basis.pkl")
        self.predictor.save(run_dir / "predictor.pkl")
        if self._mean_offset.enabled:
            save_pickle(self._mean_offset.state(), run_dir / "mean_maps_by_contrast.pkl")

        cog_train = np.asarray([self.samples[i].cog_features for i in train_idx])
        cog_test = np.asarray([self.samples[i].cog_features for i in test_idx])
        stim_train = np.asarray([self.samples[i].stim_encoding for i in train_idx])
        stim_test = np.asarray([self.samples[i].stim_encoding for i in test_idx])
        maps_train_gt = np.asarray([self._get_map(i) for i in train_idx])
        maps_test_gt = np.asarray([self._get_map(i) for i in test_idx])

        coeffs_pred = self._test_artifacts["coeffs_pred"]
        maps_test_pred = self._test_artifacts["maps_pred"]

        # Back-project to voxel space for visualisation if we trained in atlas space
        if self._atlas_projector is not None:
            print("  back-projecting atlas-space maps to voxel space for viz...")
            maps_train_gt = self._atlas_projector.roi_to_voxel_batch(maps_train_gt)
            maps_test_gt = self._atlas_projector.roi_to_voxel_batch(maps_test_gt)
            maps_test_pred = self._atlas_projector.roi_to_voxel_batch(maps_test_pred)

        predictions = {
            "train_indices": train_idx,
            "coeffs_gt_train": coeffs_train,
            "maps_gt_train": maps_train_gt,
            "cog_features_train": cog_train,
            "stim_encodings_train": stim_train,
            "subject_ids_train": np.asarray([self.samples[i].subject_id for i in train_idx]),
            "session_ids_train": np.asarray([self.samples[i].session_id for i in train_idx]),
            "stim_ids_train": np.asarray([self.samples[i].stim_id for i in train_idx]),
            "is_contrast_train": np.asarray([self.samples[i].is_contrast for i in train_idx]),
            "contrast_names_train": np.asarray(
                [self.samples[i].contrast_name or "" for i in train_idx]
            ),
            "test_indices": test_idx,
            "coeffs_gt": coeffs_test,
            "coeffs_pred": coeffs_pred,
            "maps_gt": maps_test_gt,
            "maps_pred": maps_test_pred,
            "cog_features": cog_test,
            "stim_encodings": stim_test,
            "stim_ids": np.asarray([self.samples[i].stim_id for i in test_idx]),
            "subject_ids": np.asarray([self.samples[i].subject_id for i in test_idx]),
            "session_ids": np.asarray([self.samples[i].session_id for i in test_idx]),
            "is_contrast": np.asarray([self.samples[i].is_contrast for i in test_idx]),
            "contrast_names": np.asarray(
                [self.samples[i].contrast_name or "" for i in test_idx]
            ),
        }
        save_pickle(predictions, run_dir / "predictions.pkl")

        metrics_clean = {k: v for k, v in metrics.items() if k != "contrasts"}
        save_json(metrics_clean, run_dir / "metrics.json")

        if "contrasts" in metrics:
            cdir = run_dir / "contrasts"
            cdir.mkdir(exist_ok=True)
            for name, maps in metrics["contrasts"].items():
                np.save(cdir / f"{name}_gt.npy", maps["map_gt"])
                np.save(cdir / f"{name}_pred.npy", maps["map_pred"])

        self.results = {
            "run_dir": str(run_dir),
            "metrics": metrics_clean,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "basis_type": self.config.basis_config.basis_type,
            "model_type": self.config.predictor_config.model_type,
        }
