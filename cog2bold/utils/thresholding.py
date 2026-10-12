"""Z-map thresholding helpers.

Bug fix from legacy `threshold_map_by_pvalue`: it claimed to "set sub-threshold to
0" but actually multiplied them by 0.1 (soft attenuation). The output dirname was
named `p0p05` — misleading. Here we expose two distinct functions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _two_tailed_p(z: np.ndarray) -> np.ndarray:
    return 2.0 * (1.0 - norm.cdf(np.abs(z)))


def threshold_zmap_zero(z_map: np.ndarray, p_threshold: float) -> np.ndarray:
    """Hard threshold: voxels with two-sided p >= p_threshold become 0."""
    p = _two_tailed_p(z_map)
    out = z_map.copy()
    out[p >= p_threshold] = 0.0
    return out


def threshold_zmap_attenuate(
    z_map: np.ndarray, p_threshold: float, factor: float = 0.1,
) -> np.ndarray:
    """Soft threshold: voxels with two-sided p >= p_threshold are scaled by `factor`.

    This is what the legacy code actually did; kept here for reproducibility but
    explicitly named so it's not confused with hard zeroing.
    """
    p = _two_tailed_p(z_map)
    out = z_map.copy()
    out[p >= p_threshold] = factor * out[p >= p_threshold]
    return out
