"""From an effect map Δ to a statistic map + threshold — two modes.

**effect** (descriptive): the statistic is Δ itself, in the target maps' units (z-stat units
for zmap targets). The default threshold is a percentile of |Δ| *pooled over every map in the
scan*, so a stronger combination gets more suprathreshold voxels than a weaker one and the
ranking stays meaningful. No significance claim.

**bootstrap** (calibrated): resample training *subjects* with replacement B times, refit the
predictor (the basis is held fixed — this is predictor uncertainty conditional on the basis),
recompute every Δ, and form ``z = Δ_full / SD_boot``. Thresholds: voxel |z| > z_thr, or a
Benjamini–Hochberg cutoff on two-sided normal p-values. Also returned: the fraction of
bootstrap Δs agreeing in sign with Δ_full, and bootstrap distributions of ROI-mean effects.

One refit serves every combination and condition, so the cost is B predictor fits. Linear
predictors are closed-form (milliseconds); MLPs pay their full training loop per refit.
"""

from __future__ import annotations

import contextlib
import copy
import io as _io
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from cog2bold.analysis.effect_atlas.atlas_labels import VoxelRegions, region_means
from cog2bold.analysis.effect_atlas.combinations import Combination, stack
from cog2bold.analysis.effect_atlas.model_io import Condition, LoadedRun
from cog2bold.predictors import create_predictor


# ---------------------------------------------------------------- thresholds


def pooled_percentile_threshold(deltas: Sequence[np.ndarray], percentile: float) -> float:
    allv = np.concatenate([np.abs(np.asarray(d)).ravel() for d in deltas])
    return float(np.percentile(allv, percentile)) if allv.size else 0.0


def z_from_bootstrap(delta: np.ndarray, sd: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 1e-12, delta / sd, 0.0)
    return z.astype(np.float32)


def fdr_z_threshold(z: np.ndarray, q: float) -> float:
    """|z| cutoff controlling the BH false-discovery rate at `q` (two-sided). inf if none survive."""
    p = 2.0 * norm.sf(np.abs(np.asarray(z, dtype=np.float64)))
    order = np.argsort(p)
    m = len(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    if not passed.any():
        return float("inf")
    p_cut = p[order][np.flatnonzero(passed).max()]
    return float(norm.isf(p_cut / 2.0))


# ---------------------------------------------------------------- bootstrap


@dataclass
class BootstrapStats:
    n_boot: int
    sd: Dict[str, np.ndarray]                 # cond -> (n_combos, n_voxels)
    consistency: Dict[str, np.ndarray]        # cond -> (n_combos, n_voxels) sign-agreement
    roi_boot: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    # cond -> {"ho"|"yeo7": (n_boot, n_combos, n_regions)}


def _training_targets(run: LoadedRun):
    """(X, S, coeffs, maps|None) in the space the predictor was trained in."""
    t = run.train
    X = np.asarray(t["cog_features_train"], dtype=np.float64)
    S = np.asarray(t["stim_encodings_train"], dtype=np.float32)
    if run.cog_only:
        S = np.zeros((len(X), 0), dtype=np.float32)
    Y = np.asarray(t["coeffs_gt_train"])
    M = None
    if run.config.predictor_config.loss_mode == "map":
        M = np.asarray(t["maps_gt_train"], dtype=np.float32)
        if run._projector is not None:  # predictions.pkl stores voxel back-projections
            M = run._projector.voxel_to_roi_batch(M)
        if run.mean_offset.enabled:
            M = run.mean_offset.center(M, run.train_row_names())
    return X, S, Y, M


def refit_predictor(run: LoadedRun, rows: np.ndarray, seed: int, data=None):
    """Refit the run's predictor on training rows `rows` (basis and cog normalisation fixed)."""
    X, S, Y, M = data if data is not None else _training_targets(run)
    cfg = copy.deepcopy(run.config.predictor_config)
    cfg.random_seed = int(seed)
    pred = create_predictor(cfg)
    kw = dict(cog_mean=run.cog_mean, cog_std=run.cog_std)
    if M is not None:
        kw.update(target_maps=M[rows], basis_object=run.basis)
    with contextlib.redirect_stdout(_io.StringIO()):
        pred.fit(X[rows], S[rows], Y[rows], **kw)
    return pred


def bootstrap(
    run: LoadedRun,
    combos: List[Combination],
    conditions: List[Condition],
    deltas_full: Dict[str, np.ndarray],
    n_boot: int = 200,
    seed: int = 0,
    regions: Optional[VoxelRegions] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> BootstrapStats:
    data = _training_targets(run)
    subj = np.asarray(run.train["subject_ids_train"])
    uniq = np.unique(subj)
    rows_of = {s: np.flatnonzero(subj == s) for s in uniq}
    rng = np.random.default_rng(seed)
    W, P = stack(combos)

    mean = {c.name: np.zeros_like(deltas_full[c.name], dtype=np.float64) for c in conditions}
    m2 = {c.name: np.zeros_like(deltas_full[c.name], dtype=np.float64) for c in conditions}
    agree = {c.name: np.zeros_like(deltas_full[c.name], dtype=np.float64) for c in conditions}
    roi: Dict[str, Dict[str, List[np.ndarray]]] = {c.name: {"ho": [], "yeo7": []} for c in conditions}

    for b in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_of[s] for s in draw])
        pred = refit_predictor(run, rows, seed=seed + 1 + b, data=data)
        for c in conditions:
            d = run.combine(W, P, c, predictor=pred).astype(np.float64)
            delta = d - mean[c.name]
            mean[c.name] += delta / (b + 1)
            m2[c.name] += delta * (d - mean[c.name])
            agree[c.name] += np.sign(d) == np.sign(deltas_full[c.name])
            if regions is not None:
                roi[c.name]["ho"].append(region_means(d, regions.ho, len(regions.ho_names)))
                roi[c.name]["yeo7"].append(region_means(d, regions.yeo7, len(regions.yeo7_names)))
        if progress:
            progress(b + 1, n_boot)

    denom = max(n_boot - 1, 1)
    return BootstrapStats(
        n_boot=n_boot,
        sd={k: np.sqrt(v / denom).astype(np.float32) for k, v in m2.items()},
        consistency={k: (v / max(n_boot, 1)).astype(np.float32) for k, v in agree.items()},
        roi_boot={k: {a: np.stack(v) for a, v in d.items() if v} for k, d in roi.items()},
    )
