"""Cognitive feature normalisation — train-only by construction.

Bug fix: the legacy `get_cog_stats` computed mean/std over the ENTIRE dataset
(train + test), leaking test info into the predictor's normalisation. Here we
require the caller to pass the train sample subset, so leakage is impossible.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def cog_stats_from_unique_pairs(samples: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (cog_mean, cog_std) using one sample per (subject, session) pair.

    Matches the convention from the legacy ridge regression code: averaging over
    samples directly would over-weight subjects who appear in many records.
    """
    seen = {}  # (subj, sess) -> cog_features
    for s in samples:
        key = (s.subject_id, s.session_id)
        seen.setdefault(key, s.cog_features)
    if not seen:
        raise ValueError("No samples provided to cog_stats_from_unique_pairs")
    M = np.asarray(list(seen.values()), dtype=np.float64)
    mean = M.mean(axis=0)
    std = np.maximum(M.std(axis=0), 1e-8)
    return mean, std


def standardised_distance(
    cog_a: np.ndarray, cog_b: np.ndarray, cog_std: np.ndarray
) -> float:
    """L2 distance in standardised (z-score) units."""
    return float(np.linalg.norm((cog_a - cog_b) / cog_std))
