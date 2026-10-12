"""The feature combinations an effect scan queries the model at.

Every combination is a zero-sum weighted set of raw cognitive profiles; its effect map is
``Δ = Σ_t w_t · pred(profile_t)``:

- **single**     ``pred(f high) − pred(f low)``, all other features at the training mean.
- **pairwise**   ``HH − HL − LH + LL`` — the f×g interaction. Identically zero for a linear model.
- **archetype**  ``pred(centroid) − pred(mean)`` for k-means clusters of real training subjects.
- **custom**     ``pred(A) − pred(B)`` for any two profiles (the UI's sliders).

"high"/"low" is ``μ ± k·σ`` for continuous features and 1/0 for binary ones (group one-hots),
with μ, σ the predictor's own train-only normalisation stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations as _pairs
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cog2bold.analysis.effect_atlas.model_io import LoadedRun


@dataclass
class Combination:
    name: str
    kind: str                 # single | pairwise | archetype | custom
    description: str
    profiles: np.ndarray      # (n_terms, n_cog) raw feature values
    weights: np.ndarray       # (n_terms,), sums to 0
    features: Tuple[str, ...] = ()
    meta: Dict[str, object] = field(default_factory=dict)


def _is_binary(col: np.ndarray) -> bool:
    return set(np.unique(col).tolist()) <= {0.0, 1.0}


def feature_levels(run: LoadedRun, k_sd: float = 1.0):
    """(low, high, labels) per feature; labels like ('−1SD', '+1SD') or ('0', '1')."""
    X = run.subject_profiles()
    mu, sd = run.cog_mean, run.cog_std
    low, high, labels = mu.copy(), mu.copy(), []
    for j in range(len(mu)):
        if _is_binary(X[:, j]):
            low[j], high[j] = 0.0, 1.0
            labels.append(("0", "1"))
        else:
            low[j], high[j] = mu[j] - k_sd * sd[j], mu[j] + k_sd * sd[j]
            labels.append((f"−{k_sd:g}SD", f"+{k_sd:g}SD"))
    return low, high, labels


def _select(run: LoadedRun, features: Optional[Sequence[str]]) -> List[int]:
    if not features:
        return list(range(len(run.feature_names)))
    idx = {n: i for i, n in enumerate(run.feature_names)}
    missing = [f for f in features if f not in idx]
    if missing:
        raise KeyError(f"unknown features {missing}; run has {run.feature_names}")
    return [idx[f] for f in features]


def single_feature(run: LoadedRun, k_sd: float = 1.0,
                   features: Optional[Sequence[str]] = None) -> List[Combination]:
    low, high, labels = feature_levels(run, k_sd)
    out = []
    for j in _select(run, features):
        f = run.feature_names[j]
        a, b = run.cog_mean.copy(), run.cog_mean.copy()
        a[j], b[j] = high[j], low[j]
        out.append(Combination(
            name=f, kind="single",
            description=f"{f} {labels[j][1]} vs {labels[j][0]}",
            profiles=np.stack([a, b]), weights=np.array([1.0, -1.0]), features=(f,),
        ))
    return out


def pairwise_interaction(run: LoadedRun, k_sd: float = 1.0,
                         features: Optional[Sequence[str]] = None) -> List[Combination]:
    low, high, labels = feature_levels(run, k_sd)
    out = []
    for i, j in _pairs(_select(run, features), 2):
        fi, fj = run.feature_names[i], run.feature_names[j]
        if labels[i] == labels[j] == ("0", "1"):
            continue  # two one-hot levels: HH is not a real profile
        rows = []
        for vi, vj in ((high, high), (high, low), (low, high), (low, low)):
            p = run.cog_mean.copy()
            p[i], p[j] = vi[i], vj[j]
            rows.append(p)
        out.append(Combination(
            name=f"{fi} × {fj}", kind="pairwise",
            description=f"interaction {fi} × {fj} (HH − HL − LH + LL)",
            profiles=np.stack(rows), weights=np.array([1.0, -1.0, -1.0, 1.0]),
            features=(fi, fj),
        ))
    return out


def archetypes(run: LoadedRun, k: Optional[int] = None, k_range: Tuple[int, int] = (2, 6),
               seed: int = 0) -> List[Combination]:
    """k-means archetypes of the real training subjects, each contrasted with the mean profile.

    ``k=None`` picks k in ``k_range`` by silhouette on the z-scored profiles.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = run.subject_profiles()
    Z = (X - run.cog_mean) / run.cog_std
    ks = [k] if k else list(range(k_range[0], min(k_range[1], len(Z) - 1) + 1))
    best = None
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Z)
        score = silhouette_score(Z, km.labels_) if kk > 1 else -1.0
        if best is None or score > best[0]:
            best = (score, km)
    score, km = best
    out = []
    for c in range(km.n_clusters):
        zc = km.cluster_centers_[c]
        n = int((km.labels_ == c).sum())
        top = np.argsort(-np.abs(zc))[:3]
        sig = ", ".join(f"{run.feature_names[t]} {zc[t]:+.1f}σ" for t in top)
        out.append(Combination(
            name=f"archetype {c + 1}", kind="archetype",
            description=f"archetype {c + 1} (n={n}): {sig} — vs mean profile",
            profiles=np.stack([run.cog_mean + zc * run.cog_std, run.cog_mean]),
            weights=np.array([1.0, -1.0]),
            meta={"n_subjects": n, "z_profile": dict(zip(run.feature_names, zc.round(3).tolist())),
                  "k": km.n_clusters, "silhouette": float(score)},
        ))
    return out


def custom(name: str, a: np.ndarray, b: np.ndarray, description: str = "") -> Combination:
    return Combination(
        name=name, kind="custom", description=description or name,
        profiles=np.stack([np.asarray(a, float), np.asarray(b, float)]),
        weights=np.array([1.0, -1.0]),
    )


def build_combinations(run: LoadedRun, kinds: Sequence[str] = ("single", "pairwise", "archetype"),
                       k_sd: float = 1.0, archetypes_k: Optional[int] = None,
                       features: Optional[Sequence[str]] = None, seed: int = 0,
                       ) -> Tuple[List[Combination], List[str]]:
    """All requested combinations + human-readable notes on anything skipped."""
    combos: List[Combination] = []
    notes: List[str] = []
    if "single" in kinds:
        combos += single_feature(run, k_sd, features)
    if "pairwise" in kinds:
        if run.is_linear:
            notes.append("Pairwise interactions skipped: the predictor is linear, so every "
                         "interaction map is identically zero.")
        else:
            combos += pairwise_interaction(run, k_sd, features)
    if "archetype" in kinds:
        combos += archetypes(run, archetypes_k, seed=seed)
    return combos, notes


def stack(combos: Sequence[Combination]) -> Tuple[np.ndarray, np.ndarray]:
    """All profiles stacked (n_profiles, n_cog) + a block weight matrix (n_combos, n_profiles),
    so one predictor call serves every combination."""
    P = np.concatenate([c.profiles for c in combos], axis=0)
    W = np.zeros((len(combos), len(P)))
    o = 0
    for i, c in enumerate(combos):
        W[i, o:o + len(c.weights)] = c.weights
        o += len(c.weights)
    return W, P
