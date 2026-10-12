"""Calibrated analysis of the LTO accuracy-vs-cognitive-distance curve.

Motivation
----------
`examples/lto_curves_ds000030.py` plots leave-two-out accuracy against a minimum
cognitive-distance threshold `d` and reads the rise against a flat line at 0.25. That
comparison is not sound, for four reasons:

1. **Precision drifts.** `n_pairs` collapses by ~50x across the sweep, and `n_pairs` itself
   overstates precision because a subject recurs in many pairs.
2. **The null location drifts.** True chance is `1/4 - Var(F(W))` where `W` is the hubness
   gap between the two held-out ground-truth maps (`docs/data_informativeness_null_below_
   chance.tex`). Selecting pairs on cognitive distance changes the law of `W`, so chance is
   a function of `d` whose direction is not predictable a priori.
3. **The points are not independent.** The pair sets are strictly nested, so a cumulative
   curve looks smooth and monotone almost by construction.
4. **The x-axis is regime-dependent**, because `cog_std` is computed per adapter-filtered
   cohort.

The fix, implemented here:

* **Measure the null instead of assuming it.** Both the observed accuracy and its null go
  through the identical pair selection at every `d`, so the comparison is apples-to-apples
  by construction — no reasoning about how the composition changed is required.
* **Report `n_eff`, not `n_pairs`.** The permutation spread at each threshold yields
  `n_eff = m(1-m)/sd^2`, the independent-trial equivalent. Generalises the `0.25/null_sd**2`
  estimator already used in `analysis/triplet_test.py`.
* **Make the headline a single trend test** over all pairs at their own distances, so there
  are no thresholds, no nesting and no multiplicity — and test its slope against a
  *permutation* null, because under a drifting chance level the null slope is not zero.

Everything here is post-hoc over the `null_verdicts.npz` written by
`cog2bold.pipelines.lto` with `store_null_verdicts=True`. That works because **a pair's
verdict does not depend on `d`**: training always uses the other N-2 samples, and
`min_cog_distance` only gates which pairs are scored. One evaluation therefore supports any
threshold grid, either binning scheme, and any re-standardisation of the x-axis.

Deliberately NOT provided: a p-value-vs-distance curve. A p-value conflates effect size with
n, and n collapses 50x across the sweep, so such a curve would be dominated by the precision
confound; it would also reintroduce the `1/(B+1)` permutation floor and create a multiplicity
problem over correlated, nested tests. Per-point p-values are returned in the tidy frame
(BH-corrected) for tabulation, and the effect-size axis is `accuracy - null_mean`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cog2bold.analysis.independence_tests import bh_fdr
from cog2bold.pipelines.lto import _summarise_null

# A tiny ridge is applied to the logistic fit. This is NOT for regularisation: with
# `penalty=None` a permutation that happens to be perfectly separable sends the slope to
# infinity, and a handful of divergent fits would corrupt the null distribution. The penalty
# is applied identically to the observed and permuted fits, so the permutation test stays
# exact; it only keeps the statistic finite.
_LOGIT_C = 1e6
_LOGIT_MAX_ITER = 1000


# --------------------------------------------------------------------------- loading


@dataclass
class VerdictTable:
    """One evaluated LTO pair per row, plus the permutation verdict matrix.

    `correct_null` is `(n_permutations, n_pairs)` and is column-aligned with
    `correct_obs` — `lto.LeaveTwoOutClassifier._assert_pair_alignment` guarantees every
    pass scored the same pairs in the same order.
    """

    idx1: np.ndarray            # (n_pairs,) sample index of the first held-out map
    idx2: np.ndarray            # (n_pairs,)
    subj1: np.ndarray           # (n_pairs,) subject id — the clustering unit
    subj2: np.ndarray           # (n_pairs,)
    cog1: np.ndarray            # (n_pairs, n_feat) raw cognitive vectors
    cog2: np.ndarray            # (n_pairs, n_feat)
    cog_distance: np.ndarray    # (n_pairs,) standardised L2, under this run's own cog_std
    cog_std: np.ndarray         # (n_feat,) the standardisation actually used
    contrast: np.ndarray        # (n_pairs,) str
    correct_obs: np.ndarray     # (n_pairs,) bool
    correct_null: np.ndarray    # (n_permutations, n_pairs) bool
    meta: dict

    @property
    def n_pairs(self) -> int:
        return int(self.correct_obs.shape[0])

    @property
    def n_permutations(self) -> int:
        return int(self.correct_null.shape[0]) if self.correct_null.size else 0

    @property
    def p_floor(self) -> float:
        """Smallest p-value the permutation count can express (Phipson & Smyth +1)."""
        return 1.0 / (1.0 + self.n_permutations) if self.n_permutations else float("nan")

    @property
    def n_subjects(self) -> int:
        return int(np.unique(np.concatenate([self.subj1, self.subj2])).size)


def load_verdicts(
    path: Path, metric: str = "correlation", corr_config: Optional[str] = None,
) -> VerdictTable:
    """Load `null_verdicts.npz` for one (metric, correlation-config) combination."""
    path = Path(path)
    if path.is_dir():
        path = path / "null_verdicts.npz"
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    cfg = corr_config or meta.get("primary_corr_config", "voxel")
    key = f"{metric}__{cfg}"
    if f"obs__{key}" not in z:
        available = sorted(k[5:] for k in z.files if k.startswith("obs__"))
        raise KeyError(f"{key!r} not in {path}; available: {available}")
    null = z[f"null__{key}"] if f"null__{key}" in z else np.zeros((0, len(z["idx1"])), bool)
    return VerdictTable(
        idx1=z["idx1"], idx2=z["idx2"], subj1=z["subj1"], subj2=z["subj2"],
        cog1=z["cog1"], cog2=z["cog2"],
        cog_distance=z["cog_distance"], cog_std=z["cog_std"],
        contrast=z["contrast"],
        correct_obs=z[f"obs__{key}"], correct_null=null,
        meta={**meta, "metric": metric, "corr_config": cfg},
    )


def restandardise_distance(
    cog1: np.ndarray, cog2: np.ndarray, cog_std: np.ndarray,
    keep_cols: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Recompute the distance axis under a different standardisation.

    `cog_std` is computed per adapter-filtered cohort, so `d=3` on a CONTROL-only curve is a
    *larger* raw cognitive difference than `d=3` on an all-groups curve (controls vary less,
    so their std is smaller). Passing one cohort-wide `cog_std` here puts every regime on a
    common axis. `keep_cols` selects a feature subset — use it to drop diagnosis one-hot
    columns, which widen the vector for `*_diagoh` regimes and would otherwise make them
    incommensurable with the rest.

    The float32 subtraction order matches `lto._save_null_verdicts`; see the dtype note
    there before "simplifying" this.
    """
    a, b = np.asarray(cog1), np.asarray(cog2)
    s = np.asarray(cog_std, dtype=np.float64)
    if keep_cols is not None:
        keep = list(keep_cols)
        a, b, s = a[:, keep], b[:, keep], s[keep]
    return np.asarray(np.linalg.norm((a - b) / s, axis=-1), dtype=np.float64)


# --------------------------------------------------------------------------- curves


def threshold_masks(
    cog_distance: np.ndarray, thresholds: Sequence[float],
    mode: str = "cumulative", bin_width: float = 1.0,
) -> np.ndarray:
    """(n_thresholds, n_pairs) boolean selection matrix.

    `cumulative` — `d >= t`. Reproduces the historical figure exactly, and answers "if I only
    trust far-apart pairs, how good am I?". The sets are strictly nested, so consecutive
    points share nearly all their data and must not be read as independent evidence.

    `binned` — `t <= d < t + bin_width`. Disjoint, so points are close to independent and
    cannot inherit their shape from the smallest-threshold population. This is the right
    display for the dose-response claim ("accuracy depends on distance").
    """
    d = np.asarray(cog_distance, dtype=np.float64)
    if mode == "cumulative":
        return np.asarray([d >= t for t in thresholds])
    if mode == "binned":
        return np.asarray([(d >= t) & (d < t + bin_width) for t in thresholds])
    raise ValueError(f"Unknown mode {mode!r}; expected 'cumulative' or 'binned'")


def _accuracy_under(masks: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Accuracy per threshold. `correct` is (n_pairs,) or (B, n_pairs); result (T,) or (B, T)."""
    n = masks.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        hits = np.asarray(correct, dtype=np.float64) @ masks.T.astype(np.float64)
        return hits / n


def effective_n(null_mean: np.ndarray, null_sd: np.ndarray) -> np.ndarray:
    """Independent-trial equivalent implied by the permutation spread.

    If the scored pairs were independent Bernoulli(m) trials, the null accuracy would have
    `sd = sqrt(m(1-m)/N)`; inverting gives `N_eff = m(1-m)/sd^2`. Because a subject recurs in
    many pairs — and because the spread also absorbs refit-to-refit variability — `n_eff` is
    typically far below `n_pairs`, and it is `n_eff` that should size an error bar.

    Generalises `analysis/triplet_test.py`'s `n_eff = 0.25/null_sd**2` (which assumes m=0.5).
    """
    m = np.asarray(null_mean, dtype=np.float64)
    sd = np.asarray(null_sd, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 1e-12, m * (1.0 - m) / np.square(sd), np.nan)


def global_band(
    null_curves: np.ndarray, null_mean: np.ndarray, null_sd: np.ndarray, level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sup-norm simultaneous band over the whole curve.

    A pointwise band is exceeded somewhere by chance far more often than `1 - level` when
    there are several thresholds. This standardises each threshold by its own null sd, takes
    the max over thresholds within each permuted curve, and uses the `level` quantile of that
    maximum as one common critical width — so "the curve as a whole rises above its null" is
    a single claim rather than T correlated ones.
    """
    sd = np.where(np.asarray(null_sd) > 1e-12, null_sd, np.nan)
    z = np.abs((null_curves - null_mean) / sd)
    with np.errstate(invalid="ignore"):
        crit = float(np.nanquantile(np.nanmax(z, axis=1), level))
    return null_mean - crit * sd, null_mean + crit * sd


def calibrated_curve(
    vt: VerdictTable,
    thresholds: Sequence[float],
    mode: str = "cumulative",
    bin_width: float = 1.0,
    cog_distance: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """The curve with its measured null: one row per threshold.

    Pass `cog_distance` to override the axis (e.g. a cohort-wide re-standardisation from
    `restandardise_distance`); otherwise the run's own distances are used.

    Columns: `min_cog_dist, mode, accuracy, n_pairs, null_mean, null_sd, null_lo, null_hi,
    global_lo, global_hi, n_eff, delta, emp_p, p_floor, chance` (`chance` is the nominal 0.25,
    retained only for backwards compatibility with existing viewers — never compare to it).
    """
    d = vt.cog_distance if cog_distance is None else np.asarray(cog_distance, dtype=np.float64)
    masks = threshold_masks(d, thresholds, mode=mode, bin_width=bin_width)
    n = masks.sum(axis=1)
    acc = _accuracy_under(masks, vt.correct_obs)

    T = len(thresholds)
    if vt.n_permutations:
        null_curves = _accuracy_under(masks, vt.correct_null)          # (B, T)
        # Reuse the project's null conventions (+1-corrected one-sided p, percentile CI,
        # delta-over-null-mean) rather than re-deriving them here.
        summaries = [
            _summarise_null(float(acc[t]), [float(v) for v in null_curves[:, t]
                                            if np.isfinite(v)])
            for t in range(T)
        ]
        null_mean = np.array([s["mean"] if s["mean"] is not None else np.nan for s in summaries])
        null_sd = np.array([s["std"] if s["std"] is not None else np.nan for s in summaries])
        null_lo = np.array([s["ci95_low"] if s["ci95_low"] is not None else np.nan
                            for s in summaries])
        null_hi = np.array([s["ci95_high"] if s["ci95_high"] is not None else np.nan
                            for s in summaries])
        emp_p = np.array([s["empirical_p_one_sided"] if s["empirical_p_one_sided"] is not None
                          else np.nan for s in summaries])
        delta = np.array([s["delta_over_null_mean"] if s["delta_over_null_mean"] is not None
                          else np.nan for s in summaries])
        g_lo, g_hi = global_band(null_curves, null_mean, null_sd)
        n_eff = effective_n(null_mean, null_sd)
    else:
        nan = np.full(T, np.nan)
        null_mean = null_sd = null_lo = null_hi = emp_p = delta = g_lo = g_hi = n_eff = nan

    df = pd.DataFrame({
        "min_cog_dist": np.asarray(thresholds, dtype=np.float64),
        "mode": mode,
        "accuracy": acc,
        "n_pairs": n,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_lo": null_lo,
        "null_hi": null_hi,
        "global_lo": g_lo,
        "global_hi": g_hi,
        "n_eff": n_eff,
        "delta": delta,
        "emp_p": emp_p,
        "p_floor": vt.p_floor,
        "chance": 0.25,
    })
    df["q_bh"] = bh_fdr(df["emp_p"].to_numpy())
    return df


# --------------------------------------------------------------------------- trend test


def logistic_slope(
    x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
    x_mean: Optional[float] = None, x_sd: Optional[float] = None,
) -> float:
    """Slope of `logit P(correct) = b0 + b1 * x`, in raw units of x.

    `x` is standardised internally for conditioning and the slope converted back, so the
    returned value is per unit of cognitive distance. Since x is identical across
    permutations this rescaling cannot affect any permutation p-value.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y).astype(int).ravel()
    if y.size == 0:
        return float("nan")
    if y.min() == y.max():
        # No variation to explain. A permutation that scored every pair wrong (or every pair
        # right) exhibits no distance trend, and the ridge-penalised MLE for such data is
        # exactly beta1 = 0 -- sklearn simply refuses to fit a single-class target. Record the
        # 0 rather than dropping the permutation: dropping would silently restrict the null to
        # the luckier permutations and inflate its spread. Not hypothetical -- in deviation
        # space 18 of 60 null permutations score zero correct pairs.
        return 0.0
    xr = np.asarray(x, dtype=np.float64).ravel()
    mu = float(np.mean(xr)) if x_mean is None else x_mean
    sd = float(np.std(xr)) if x_sd is None else x_sd
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("nan")
    z = ((xr - mu) / sd).reshape(-1, 1)
    # L2 is the default penalty in every sklearn version; naming it explicitly is deprecated
    # from 1.8 on, so we only set C.
    model = LogisticRegression(C=_LOGIT_C, solver="lbfgs", max_iter=_LOGIT_MAX_ITER)
    model.fit(z, y, sample_weight=sample_weight)
    return float(model.coef_[0][0] / sd)


def rank_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """AUC = P(distance of a correct pair > distance of an incorrect pair); 0.5 = no relation.

    A link-free companion to the logistic slope, computed from the same data at negligible
    cost so the conclusion does not rest on the logit assumption. Equivalent to a
    Mann-Whitney U, normalised.
    """
    y = np.asarray(y).astype(bool).ravel()
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(x, dtype=np.float64).ravel())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _perm_p(observed: float, null: np.ndarray) -> float:
    """One-sided permutation p, +1-corrected — same convention as `lto._summarise_null`."""
    arr = np.asarray(null, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size or not np.isfinite(observed):
        return float("nan")
    return float((1 + int((arr >= observed).sum())) / (1 + arr.size))


def trend_test(
    vt: VerdictTable, cog_distance: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """The headline test: does accuracy increase with cognitive distance?

    Fits `logit P(correct) = b0 + b1 * cog_distance` over every pair at its own distance — no
    thresholds, no bins, no nesting, each pair used exactly once.

    The p-value comes from refitting on each permuted verdict vector, **not** from the fit's
    own standard error. Two independent reasons:

    1. The rows are not independent (a subject recurs in many pairs), so a model-based SE
       would be badly anti-conservative. A permutation test is valid under arbitrary
       dependence, because the null is generated with that same dependence in place.
    2. Under a chance level that drifts with distance, the null slope is not zero. Testing
       against 0 would be exactly the error this whole module exists to avoid; testing
       against the permuted slope distribution absorbs the drift whatever its sign.
    """
    d = vt.cog_distance if cog_distance is None else np.asarray(cog_distance, dtype=np.float64)
    mu, sd = float(np.mean(d)), float(np.std(d))

    beta1 = logistic_slope(d, vt.correct_obs, x_mean=mu, x_sd=sd)
    auc = rank_statistic(d, vt.correct_obs)

    beta1_null = np.array([
        logistic_slope(d, row, x_mean=mu, x_sd=sd) for row in vt.correct_null
    ]) if vt.n_permutations else np.array([])
    auc_null = np.array([
        rank_statistic(d, row) for row in vt.correct_null
    ]) if vt.n_permutations else np.array([])

    return {
        "beta1": beta1,
        "beta1_null_mean": float(np.nanmean(beta1_null)) if beta1_null.size else float("nan"),
        "beta1_null_sd": float(np.nanstd(beta1_null)) if beta1_null.size else float("nan"),
        "beta1_p": _perm_p(beta1, beta1_null),
        "auc": auc,
        "auc_null_mean": float(np.nanmean(auc_null)) if auc_null.size else float("nan"),
        "auc_p": _perm_p(auc, auc_null),
        "p_floor": vt.p_floor,
        "n_pairs": vt.n_pairs,
        "n_subjects": vt.n_subjects,
        "n_permutations": vt.n_permutations,
        "_beta1_null": beta1_null,   # kept for the null-histogram panel
        "_auc_null": auc_null,
    }


# --------------------------------------------------------------------- dependence controls


def disjoint_matchings(
    subj1: np.ndarray, subj2: np.ndarray, n_matchings: int = 20, seed: int = 0,
) -> List[np.ndarray]:
    """Row indices of matchings in which no subject appears twice.

    Greedy over a seeded random pair order — the same construction
    `analysis/triplet_test.disjoint_triplet_wins` uses to obtain genuinely independent
    trials. Each matching holds at most `n_subjects // 2` pairs, so any single one is small;
    the point is the distribution over many of them.
    """
    rng = np.random.default_rng(seed)
    n = len(subj1)
    out: List[np.ndarray] = []
    for _ in range(n_matchings):
        order = rng.permutation(n)
        used: set = set()
        keep: List[int] = []
        for i in order:
            a, b = int(subj1[i]), int(subj2[i])
            if a in used or b in used:
                continue
            used.update((a, b))
            keep.append(int(i))
        out.append(np.sort(np.asarray(keep, dtype=np.int64)))
    return out


def disjoint_matching_slopes(
    vt: VerdictTable, n_matchings: int = 20, seed: int = 0,
    cog_distance: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Refit the trend on subject-disjoint subsets: is the slope an artifact of reuse?

    Every row within a matching is an independent trial, so this removes the dependence
    entirely rather than correcting for it. Each matching also gets its own permutation
    p-value, computed from the null verdicts restricted to the same rows.
    """
    d = vt.cog_distance if cog_distance is None else np.asarray(cog_distance, dtype=np.float64)
    rows = []
    for k, sel in enumerate(disjoint_matchings(vt.subj1, vt.subj2, n_matchings, seed)):
        if sel.size < 8:
            continue
        ds = d[sel]
        mu, sd = float(np.mean(ds)), float(np.std(ds))
        b = logistic_slope(ds, vt.correct_obs[sel], x_mean=mu, x_sd=sd)
        b_null = np.array([
            logistic_slope(ds, row[sel], x_mean=mu, x_sd=sd) for row in vt.correct_null
        ]) if vt.n_permutations else np.array([])
        acc = float(np.mean(vt.correct_obs[sel]))
        acc_null = (np.asarray([np.mean(row[sel]) for row in vt.correct_null])
                    if vt.n_permutations else np.array([]))
        rows.append({
            "matching": k,
            "n_pairs": int(sel.size),
            "accuracy": acc,
            "null_mean": float(np.mean(acc_null)) if acc_null.size else float("nan"),
            "acc_p": _perm_p(acc, acc_null),
            "beta1": b,
            "beta1_p": _perm_p(b, b_null),
        })
    return pd.DataFrame(rows)


def subject_bootstrap_slope_ci(
    vt: VerdictTable, n_boot: int = 500, seed: int = 0, level: float = 0.95,
    cog_distance: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Percentile CI for the trend slope from a subject-level (vertex) bootstrap.

    Resamples *subjects* with replacement and reweights each pair by `count(u) * count(v)`,
    the induced multiplicity — the standard vertex bootstrap for dyadic data, where
    resampling rows directly would ignore that rows sharing a subject move together.

    CI only. The p-value stays permutation-based (see `trend_test`); the bootstrap is here to
    size the uncertainty on the effect, not to test it.
    """
    d = vt.cog_distance if cog_distance is None else np.asarray(cog_distance, dtype=np.float64)
    mu, sd = float(np.mean(d)), float(np.std(d))
    subjects = np.unique(np.concatenate([vt.subj1, vt.subj2]))
    pos = {int(s): k for k, s in enumerate(subjects)}
    a = np.array([pos[int(s)] for s in vt.subj1])
    b = np.array([pos[int(s)] for s in vt.subj2])

    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        counts = rng.multinomial(subjects.size, np.full(subjects.size, 1.0 / subjects.size))
        w = counts[a].astype(np.float64) * counts[b].astype(np.float64)
        if w.sum() <= 0:
            continue
        keep = w > 0
        if keep.sum() < 8 or len(np.unique(vt.correct_obs[keep])) < 2:
            continue
        slopes.append(
            logistic_slope(d[keep], vt.correct_obs[keep], sample_weight=w[keep],
                           x_mean=mu, x_sd=sd)
        )
    arr = np.asarray(slopes, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {"beta1_ci_lo": float("nan"), "beta1_ci_hi": float("nan"), "n_boot_ok": 0}
    lo, hi = np.quantile(arr, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"beta1_ci_lo": float(lo), "beta1_ci_hi": float(hi), "n_boot_ok": int(arr.size)}


__all__ = [
    "VerdictTable",
    "load_verdicts",
    "restandardise_distance",
    "threshold_masks",
    "effective_n",
    "global_band",
    "calibrated_curve",
    "logistic_slope",
    "rank_statistic",
    "trend_test",
    "disjoint_matchings",
    "disjoint_matching_slopes",
    "subject_bootstrap_slope_ci",
]
