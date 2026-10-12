"""MeanMapOffset — optional "predict the deviation from the population-mean map" transform.

Motivation: with a supervised/hybrid basis (e.g. ridgecol+PCA) the basis has no ``mean_map``, so
``decode()`` produces maps with no shared/common component — good for *relative* discrimination
(LTO) but poor as *absolute* reconstructions (they lose to the population-mean map in LOO). This
helper lets the pipeline model the **deviation** from the per-contrast training mean map and add the
mean back at reconstruction, so predicted maps carry the common component too.

It lives at the map-I/O boundary and is therefore fully general: it needs nothing from the basis or
predictor and works for any current/future model, in voxel space or atlas/ROI space (shape-agnostic).

Usage (pipeline side)::

    off = MeanMapOffset(run_config.mean_map_offset)
    off.fit(train_maps, [contrast_of(i) for i in train_idx])
    basis.fit(off.center(train_maps, contrast_names), ...)      # fit on deviations
    ...
    full_map = off.restore(basis.decode(coeffs), contrast_of(sample))   # add mean back

When ``enabled=False`` every method is an identity pass-through, so callers stay branch-free and
behaviour is bit-for-bit unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


class MeanMapOffset:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.mean_by_contrast: Dict[str, np.ndarray] = {}

    def fit(self, maps: np.ndarray, contrast_names: Sequence[str]) -> "MeanMapOffset":
        """Store the mean map per contrast over the given (training) maps."""
        if not self.enabled:
            return self
        maps = np.asarray(maps)
        by: Dict[str, List[np.ndarray]] = {}
        for m, c in zip(maps, contrast_names):
            by.setdefault(c, []).append(m)
        self.mean_by_contrast = {c: np.mean(v, axis=0) for c, v in by.items()}
        return self

    def center(self, maps: np.ndarray, contrast_names: Sequence[str]) -> np.ndarray:
        """Subtract each map's per-contrast mean. Identity when disabled."""
        if not self.enabled:
            return maps
        maps = np.asarray(maps)
        return np.asarray([m - self.mean_by_contrast[c] for m, c in zip(maps, contrast_names)])

    def restore(self, activation_map: np.ndarray, contrast_name: str) -> np.ndarray:
        """Add the per-contrast mean back to a decoded (deviation) map. Identity when disabled."""
        if not self.enabled:
            return activation_map
        return activation_map + self.mean_by_contrast[contrast_name]

    def deviate(self, activation_map: np.ndarray, contrast_name: str) -> np.ndarray:
        """Subtract the per-contrast mean from a full map — the inverse of ``restore``.

        Used for *scoring* in deviation space: both the ground truth and the prediction go
        through this, so the shared μ that dominates absolute maps cancels out and the
        metric only sees the sample-specific part. Identity when disabled.
        """
        if not self.enabled:
            return activation_map
        return activation_map - self.mean_by_contrast[contrast_name]

    def has(self, contrast_name: str) -> bool:
        """Is a mean map available for this contrast? Always True when disabled (no-op)."""
        return (not self.enabled) or (contrast_name in self.mean_by_contrast)

    # ---- persistence ----
    def state(self) -> dict:
        return {"enabled": self.enabled, "mean_by_contrast": self.mean_by_contrast}

    def load_state(self, state: dict) -> "MeanMapOffset":
        self.enabled = bool(state.get("enabled", False))
        self.mean_by_contrast = state.get("mean_by_contrast", {})
        return self
