"""Evaluation metrics.

Bug fixes vs the legacy code:
  - voxelwise_correlation is vectorised (no Python per-voxel loop)
  - compute_contrast uses stim_encoding (not stim_id ints), and pulls GT maps from
    the same _get_map(i) helper as predicted maps so atlas-space training stays
    consistent.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def voxelwise_correlation(maps_gt: np.ndarray, maps_pred: np.ndarray) -> np.ndarray:
    """Per-voxel Pearson correlation across samples.

    Args:
        maps_gt:   (n_samples, n_voxels)
        maps_pred: (n_samples, n_voxels)
    Returns:
        (n_voxels,) correlations in [-1, 1] (zeros where a voxel has no variance).
    """
    if maps_gt.shape != maps_pred.shape:
        raise ValueError(f"shape mismatch: {maps_gt.shape} vs {maps_pred.shape}")
    if maps_gt.shape[0] < 2:
        return np.zeros(maps_gt.shape[1], dtype=np.float64)
    gt = maps_gt - maps_gt.mean(axis=0, keepdims=True)
    pr = maps_pred - maps_pred.mean(axis=0, keepdims=True)
    num = (gt * pr).sum(axis=0)
    norm_gt = np.linalg.norm(gt, axis=0)
    norm_pr = np.linalg.norm(pr, axis=0)
    den = norm_gt * norm_pr
    out = np.zeros_like(num)
    nz = den > 1e-12
    out[nz] = num[nz] / den[nz]
    out[~np.isfinite(out)] = 0.0
    return out


def compute_metrics(
    maps_gt: np.ndarray,
    maps_pred: np.ndarray,
    coeffs_gt: Optional[np.ndarray] = None,
    coeffs_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Bundle of map- and coefficient-level metrics."""
    if maps_gt.ndim != 2:
        maps_gt = maps_gt.reshape(maps_gt.shape[0], -1)
    if maps_pred.ndim != 2:
        maps_pred = maps_pred.reshape(maps_pred.shape[0], -1)

    voxel_corr = voxelwise_correlation(maps_gt, maps_pred)

    diff = maps_gt - maps_pred
    map_mse = float((diff ** 2).mean())
    map_mae = float(np.abs(diff).mean())

    # Cosine similarity per sample
    eps = 1e-12
    norm_gt = np.linalg.norm(maps_gt, axis=1)
    norm_pr = np.linalg.norm(maps_pred, axis=1)
    cos = (maps_gt * maps_pred).sum(axis=1) / np.maximum(norm_gt * norm_pr, eps)

    ss_res = float(((maps_gt - maps_pred) ** 2).sum())
    ss_tot = float(((maps_gt - maps_gt.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, eps)

    out = {
        "voxel_correlation_mean": float(np.nanmean(voxel_corr)),
        "voxel_correlation_median": float(np.nanmedian(voxel_corr)),
        "voxel_correlation_std": float(np.nanstd(voxel_corr)),
        "map_mse": map_mse,
        "map_mae": map_mae,
        "cosine_similarity_mean": float(np.nanmean(cos)),
        "cosine_similarity_std": float(np.nanstd(cos)),
        "explained_variance_r2": float(r2),
    }
    if coeffs_gt is not None and coeffs_pred is not None:
        out["coeff_mse"] = float(((coeffs_gt - coeffs_pred) ** 2).mean())
    return out


def compute_contrast(
    samples,
    basis,
    predictor,
    test_idx: np.ndarray,
    stim_a: str,
    stim_b: str,
    label2id: Dict[str, int],
    map_lookup: Callable[[int], np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """GT vs predicted contrast: mean(maps_a) - mean(maps_b).

    Two bug fixes baked in:
      1. We pass `stim_encoding` (one-hot vector) to `predictor.predict`, not `stim_id`.
      2. GT maps come from `map_lookup(i)` which returns whatever space training was in
         (voxel or pre-projected ROI), matching the predicted side.
    """
    a_id = label2id[stim_a]
    b_id = label2id[stim_b]
    a_idx = [i for i in test_idx if (not samples[i].is_contrast) and samples[i].stim_id == a_id]
    b_idx = [i for i in test_idx if (not samples[i].is_contrast) and samples[i].stim_id == b_id]
    if not a_idx or not b_idx:
        return None, None

    n_stim = samples[a_idx[0]].stim_encoding.shape[0]

    def one_hot(stim_id: int) -> np.ndarray:
        v = np.zeros(n_stim, dtype=np.float32)
        v[stim_id] = 1.0
        return v

    cog_a = np.asarray([samples[i].cog_features for i in a_idx])
    enc_a = np.tile(one_hot(a_id), (len(a_idx), 1))
    coeffs_a_pred = predictor.predict(cog_a, enc_a)
    maps_a_pred = basis.decode_batch(coeffs_a_pred)

    cog_b = np.asarray([samples[i].cog_features for i in b_idx])
    enc_b = np.tile(one_hot(b_id), (len(b_idx), 1))
    coeffs_b_pred = predictor.predict(cog_b, enc_b)
    maps_b_pred = basis.decode_batch(coeffs_b_pred)

    maps_a_gt = np.asarray([map_lookup(i) for i in a_idx])
    maps_b_gt = np.asarray([map_lookup(i) for i in b_idx])

    return (
        maps_a_gt.mean(axis=0) - maps_b_gt.mean(axis=0),
        maps_a_pred.mean(axis=0) - maps_b_pred.mean(axis=0),
    )
