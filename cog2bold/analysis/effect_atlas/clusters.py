"""Suprathreshold blobs of a statistic map, reported like a neuroimaging peak table.

Positive (activation) and negative (deactivation) clusters are found separately by
connected-component labelling (26-connectivity by default, as in FSL). For each cluster:

- the peak and up to ``max_subpeaks`` local maxima at least ``min_peak_distance_mm`` apart
  (SPM/nilearn style), in MNI mm through the mask affine,
- the size (voxels, mm³), the |stat|-weighted centre of gravity, the bounding extent,
- the mean Δ inside it and the share of the map's total |Δ| it holds.

The output is a tidy DataFrame (one row per peak; ``peak_type`` = 'peak' | 'sub') plus a
signed label vector (+id / −id per in-mask voxel, 0 elsewhere) for composition and display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage


@dataclass
class ClusterParams:
    connectivity: int = 26            # 6 | 18 | 26
    min_cluster_voxels: int = 10
    min_peak_distance_mm: float = 8.0
    max_subpeaks: int = 2


_STRUCT = {6: 1, 18: 2, 26: 3}


def _to_vol(vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vol = np.zeros(mask.shape, dtype=np.float32)
    vol[mask] = vec
    return vol


def find_clusters(
    stat: np.ndarray,
    threshold: float,
    mask_img,
    delta: Optional[np.ndarray] = None,
    consistency: Optional[np.ndarray] = None,
    params: ClusterParams = ClusterParams(),
) -> Tuple[pd.DataFrame, np.ndarray]:
    mask = np.asarray(mask_img.dataobj).astype(bool)
    affine = mask_img.affine
    vox_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    vox_size = np.abs(np.diag(affine)[:3])
    stat_vol = _to_vol(np.asarray(stat, dtype=np.float32), mask)
    delta = np.asarray(stat if delta is None else delta, dtype=np.float32)
    delta_vol = _to_vol(delta, mask)
    cons_vol = _to_vol(np.asarray(consistency, dtype=np.float32), mask) if consistency is not None else None
    total_abs = float(np.abs(delta).sum()) or 1.0
    struct = ndimage.generate_binary_structure(3, _STRUCT[params.connectivity])

    label_vol = np.zeros(mask.shape, dtype=np.int32)
    rows = []
    next_id = 1
    if not np.isfinite(threshold):
        return pd.DataFrame(rows), label_vol[mask]
    for sign in (1, -1):
        s = sign * stat_vol
        lab, n = ndimage.label((s > threshold) & mask, structure=struct)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        local_max = (s == ndimage.maximum_filter(s, footprint=struct)) & (lab > 0)
        for c in np.argsort(-sizes):
            if sizes[c] < params.min_cluster_voxels:
                continue
            cid = sign * next_id
            next_id += 1
            in_c = lab == (c + 1)
            label_vol[in_c] = cid
            ijk = np.argwhere(in_c)
            mm = np.c_[ijk, np.ones(len(ijk))] @ affine.T[:, :3]
            w = np.abs(stat_vol[in_c])
            cog = (mm * w[:, None]).sum(0) / w.sum()
            dvals = delta_vol[in_c]
            base = dict(
                cluster_id=int(cid),
                sign="activation" if sign > 0 else "deactivation",
                n_voxels=int(sizes[c]),
                volume_mm3=float(sizes[c] * vox_mm3),
                cog_x=float(cog[0]), cog_y=float(cog[1]), cog_z=float(cog[2]),
                extent_x_mm=float(np.ptp(mm[:, 0]) + vox_size[0]),
                extent_y_mm=float(np.ptp(mm[:, 1]) + vox_size[1]),
                extent_z_mm=float(np.ptp(mm[:, 2]) + vox_size[2]),
                mean_delta=float(dvals.mean()),
                abs_delta_share=float(np.abs(dvals).sum() / total_abs),
                sign_consistency=float(cons_vol[in_c].mean()) if cons_vol is not None else np.nan,
            )
            # peaks: local maxima inside the cluster, strongest first, spaced ≥ min distance
            cand = np.argwhere(local_max & in_c)
            cand = cand[np.argsort(-s[tuple(cand.T)])]
            kept = []
            for q in cand:
                q_mm = np.r_[q, 1.0] @ affine.T[:, :3]
                if all(np.linalg.norm(q_mm - k[1]) >= params.min_peak_distance_mm for k in kept):
                    kept.append((q, q_mm))
                if len(kept) > params.max_subpeaks:
                    break
            for n_pk, (q, q_mm) in enumerate(kept):
                rows.append({
                    **base,
                    "peak_type": "peak" if n_pk == 0 else "sub",
                    "x": float(q_mm[0]), "y": float(q_mm[1]), "z": float(q_mm[2]),
                    "peak_stat": float(stat_vol[tuple(q)]),
                    "peak_delta": float(delta_vol[tuple(q)]),
                })
    return pd.DataFrame(rows), label_vol[mask]
