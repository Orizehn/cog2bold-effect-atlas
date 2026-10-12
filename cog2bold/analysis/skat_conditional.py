"""Conditional forward selection for SKAT: which cognitive features carry INDEPENDENT map signal.

The problem
-----------
``independence_tests.skat_score_test`` tests one cognitive feature at a time against the map
kernel. When several features measure the same construct they all come out significant, and a
count of significant rows badly overstates how much has been discovered. In ds000030 this is not
hypothetical: ``ds_totalraw`` and ``lns_totalraw`` correlate at 0.71 (they are the two WAIS
subtests that get summed into the Working Memory Index), every pair in the curated battery sits at
0.40-0.71, and the first principal component explains 60% of the variance.

The procedure
-------------
Forward stepwise conditional selection. Take the feature with the strongest association, put it in
SKAT's covariate matrix ``X0``, re-test everything else, repeat until nothing survives. Because
SKAT's outcome is the cognitive score, putting feature A in ``X0`` residualises the OUTCOME: the
test then asks whether the maps explain the part of feature B that A does not.

This is the same stepwise logic as GCTA-COJO in statistical genetics (Yang et al. 2012), where
correlated markers in linkage disequilibrium each reach significance individually. The analogy is
structural, not exact -- COJO conditions on another *predictor*, here we condition on the *outcome*
side -- but the failure mode it fixes is identical. The tighter precedent is conditional SKAT in
sequencing studies, where a gene-based test is routinely conditioned on a known associated variant
by adding it to the covariate matrix.

Why the naive stopping rule is not enough -- and why it needs no permutations
----------------------------------------------------------------------------
Quoting the winning feature's own p-value is wrong, because you did not name that feature in
advance: you searched. Measured on 26 correlated features at n=74 with nothing related to the maps,
**the best candidate's p-value falls below 0.05 about 40% of the time.**

But fixing this does NOT require abandoning SKAT's analytic null, which is the whole reason SKAT was
adopted over dCor. There are two distinct nulls and only the first belongs to SKAT:

  1. "Is this one named feature related to the maps?" -- SKAT's exact mixture-of-chi-squares,
     evaluated by Imhof inversion. No permutation, no 1/(B+1) floor.
  2. "Would the best of m searched candidates look this good by luck?" -- an ordinary
     multiple-comparisons question ABOUT THE SEARCH, applied to the p-values that (1) already
     gives analytically.

So the default stopping rule is a closed-form Sidak correction over the *effective* number of
independent candidates (``effective_n_tests``, from the candidate correlation matrix -- the maps
never enter it). On ds000030/pamret this reproduces a 10,000-permutation answer:

    CONTROL (n=74):  best raw p 0.013 -> Sidak 0.071 ; permutation 0.074 ; Bonferroni x7 0.091
    pooled  (n=186): best raw p 3.9e-5 -> Sidak 2.2e-4 ; permutation 8.0e-4 (floored at 1e-4)

Note the second row: the permutation step p is resolution-limited, the analytic one is not. A
Westfall-Young step-down min-P rule is still available via ``stopping="permutation"`` as a
cross-check that assumes nothing about how the candidates correlate.

What the output means
---------------------
The selected set answers "how many statistically independent channels are there", NOT "which
distinct constructs exist". With A = g + e_A and B = g + e_B two noisy measures of one latent g,
regressing A out of B leaves ``g * s2/(1+s2)`` where ``s2 = Var(e)``: reliable indicators of one
construct annihilate each other, unreliable ones do not. Mutual annihilation is the signature of
redundancy, not of absent signal, and the procedure is conservative about total signal by
construction.

For a claim you want to defend, prefer ``skat_conditional_anchored``: pre-specify what goes in
``X0`` and run one test whose p-value is valid as printed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm, rankdata

from cog2bold.analysis.independence_tests import skat_score_test

__all__ = [
    "ScreenResult",
    "SelectResult",
    "screen_then_select",
    "skat_excess",
    "skat_null_mean",
    "skat_screen",
    "skat_select",
    "FeatureMatrix",
    "ForwardStep",
    "ForwardResult",
    "build_feature_matrix",
    "effective_n_tests",
    "leverage",
    "q_vector",
    "rank_normalize_columns",
    "residualize",
    "skat_conditional_anchored",
    "skat_conditional_forward",
    "skat_stability_selection",
    "step_pvalue_analytic",
]

# Leverage thresholds, from the stress test in docs/stat_tests_report/ section 6.5: SKAT's null is
# broken not by skewness or kurtosis but by ONE observation dominating the residual sum of squares.
LEVERAGE_WARN = 0.2
LEVERAGE_REFUSE = 0.5


# =====================================================================================
# Core primitives
# =====================================================================================


def residualize(F: np.ndarray, X0: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Project the columns of F off the covariates.

    Returns (R, P0, c) with R = P0 @ F, P0 = I - X0 X0^+ and c = rank(X0). An intercept is
    prepended when absent, matching ``skat_score_test``'s convention so the two agree exactly.
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim == 1:
        F = F[:, None]
    n = F.shape[0]
    if X0 is None:
        X0 = np.ones((n, 1))
    else:
        X0 = np.asarray(X0, dtype=np.float64)
        if X0.ndim == 1:
            X0 = X0[:, None]
        if not np.allclose(X0[:, 0], 1.0):
            X0 = np.hstack([np.ones((n, 1)), X0])
    P0 = np.eye(n) - X0 @ np.linalg.pinv(X0)
    c = int(np.linalg.matrix_rank(X0))
    return P0 @ F, P0, c


def q_vector(R: np.ndarray, K: np.ndarray, dof: int) -> np.ndarray:
    """SKAT's scale-free statistic Q for EVERY column of the residual matrix R at once.

        Q_j = dof * (R_j' K R_j) / (R_j' R_j)

    This is the whole efficiency story of the module. At a given step every candidate feature
    shares the same ``P0``, the same ``c`` and the same ``K``, so ``Q(.)`` is literally the same
    function of the outcome for all of them and their analytic nulls coincide. Ranking candidates
    by Q is therefore identical to ranking them by p-value, and only the winner needs an Imhof
    inversion. Verified against a loop over ``skat_score_test``: Q agrees to 7e-14 and the argmax
    matches the argmin of the p-values exactly (Spearman 1.000).

    The practical consequence: 10,000 permutations across 26 features take 0.2 s, where a
    per-feature Imhof loop would take ~9 minutes.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 1:
        R = R[:, None]
    num = np.einsum("ij,ij->j", K @ R, R)
    den = np.einsum("ij,ij->j", R, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = dof * num / den
    return np.where(den > 0, Q, -np.inf)


def leverage(R: np.ndarray) -> np.ndarray:
    """Per column, the share of the residual sum of squares carried by a single subject.

    ``max_i r_i^2 / sum_j r_j^2``. This, not kurtosis, is the diagnostic for when SKAT's analytic
    null breaks: at leverage > 0.9 the realised type-I error reaches 0.11 against a nominal 0.05.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 1:
        R = R[:, None]
    ss = np.einsum("ij,ij->j", R, R)
    mx = (R ** 2).max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = mx / ss
    return np.where(ss > 0, lev, 1.0)


def rank_normalize_columns(F: np.ndarray) -> np.ndarray:
    """Inverse-normal (van der Waerden) transform of each column.

    The fix for a high-leverage feature: it bounds the influence of any single subject while
    preserving the ordering, so the Gaussian-error assumption behind the analytic null holds.
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim == 1:
        F = F[:, None]
    n = F.shape[0]
    out = np.empty_like(F)
    for j in range(F.shape[1]):
        out[:, j] = norm.ppf((rankdata(F[:, j]) - 0.5) / n)
    return out


# =====================================================================================
# Step-level multiplicity correction
# =====================================================================================


def effective_n_tests(R: np.ndarray) -> float:
    """Effective number of independent candidates, from their correlation matrix.

    Cheverud-Nyholt / Li-Ji: with eigenvalues ``ev`` of the m x m correlation matrix of the
    candidates,

        M_eff = 1 + (m - 1) * (1 - Var(ev) / m).

    m identical columns give M_eff = 1; m orthogonal columns give M_eff = m. On the ds000030
    curated battery the seven candidates are worth M_eff = 5.65 independent tests, so a plain
    Bonferroni factor of 7 is needlessly conservative.

    Computed from the CANDIDATES ONLY -- it never touches the maps or the kernel, so it introduces
    no dependence on the outcome being tested.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 1 or R.shape[1] == 1:
        return 1.0
    sd = R.std(axis=0)
    keep = sd > 0
    if keep.sum() < 2:
        return float(max(int(keep.sum()), 1))
    C = np.corrcoef(R[:, keep], rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    ev = np.linalg.eigvalsh(0.5 * (C + C.T))
    m = ev.size
    meff = 1.0 + (m - 1.0) * (1.0 - float(np.var(ev, ddof=1)) / m)
    return float(np.clip(meff, 1.0, m))


def step_pvalue_analytic(p_min: float, m_eff: float, method: str = "sidak") -> float:
    """Probability that the BEST of ~m_eff candidates would reach ``p_min`` by chance.

    This is the whole multiplicity correction, in closed form:

        Sidak      : 1 - (1 - p_min)^m_eff        (exact under independence)
        Bonferroni : min(1, p_min * m_eff)        (an upper bound under ANY dependence)

    Why no permutation is needed. There are two different nulls in play and only the first is
    SKAT's:

      1. "Is this one named feature related to the maps?" -- SKAT's exact chi-square mixture,
         evaluated by Imhof inversion. No permutation, no resolution floor.
      2. "Would the best of m searched candidates look this good by luck?" -- an ordinary
         multiple-comparisons question ABOUT THE SEARCH, applied to the p-values that (1) already
         supplies analytically.

    Treating (2) as something that needs a permutation null throws away exactly the property SKAT
    was adopted for. Worse, it reintroduces the 1/(B+1) floor: on ds000030/pamret pooled the
    permutation step p bottoms out at 8e-4 against a floor of 1e-4, while the analytic route gives
    a genuine 2.2e-4.

    Using ``m_eff`` rather than the raw count keeps Sidak sharp when the candidates are correlated:
    on the CONTROL cell this gives 0.071, against 0.074 from 10,000 permutations and 0.091 from a
    plain Bonferroni factor of 7.
    """
    p_min = float(np.clip(p_min, 0.0, 1.0))
    m_eff = max(float(m_eff), 1.0)
    if method == "bonferroni":
        return float(min(1.0, p_min * m_eff))
    if method == "sidak":
        # log1p form stays accurate for the tiny p_min that strong effects produce
        return float(-np.expm1(m_eff * np.log1p(-p_min)))
    raise ValueError(f"Unknown analytic step method {method!r} (use 'sidak' or 'bonferroni')")


# =====================================================================================
# Westfall-Young step-down permutation null (optional cross-check)
# =====================================================================================


def _permutation_q(
    R: np.ndarray, K: np.ndarray, P0: np.ndarray, dof: int, n_perm: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_perm, p) matrix of Q values under the null, by permuting the null-model residuals.

    What is actually shuffled is **which cognitive score belongs to which subject**. Both K and r
    are indexed by subject, so reordering the rows of R re-deals the scores while leaving the map
    geometry identical -- it destroys the brain-behaviour pairing and nothing else.

    Two details matter and both are deliberate:

    1. **One shared permutation per replicate**, applied to every candidate simultaneously. That
       preserves the correlation structure among the features, which is exactly what lets the
       min-P statistic account for their dependence rather than assuming independence.
    2. **Re-project after permuting** (``P0 @ R[pi]``). A permuted residual is no longer orthogonal
       to the covariates, so it has to be pushed back into the residual space to remain a draw from
       the null the statistic is defined on.

    Permuting residuals rather than the raw outcome is the scheme already validated for the
    covariate-adjusted case in ``docs/stat_tests_report/`` section 6.3.

    This is an OPT-IN cross-check (``stopping="permutation"``); the default route is analytic. See
    ``step_pvalue_analytic`` for why.
    """
    n = R.shape[0]
    out = np.empty((n_perm, R.shape[1]), dtype=np.float64)
    for b in range(n_perm):
        out[b] = q_vector(P0 @ R[rng.permutation(n)], K, dof)
    return out


def _step_down_pvalues(Qo: np.ndarray, Qb: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Per-feature permutation p, the observed min-P, and the step's FWER-controlled p-value.

    The min-P null is read off the SAME (n_perm, p) matrix by ranking within each column -- no
    second permutation loop. Column ranking converts each replicate's Q into that feature's own
    permutation p-value, which puts features with different residual distributions on a common
    scale before the minimum is taken. That is why min-P is preferred to a plain max-Q here: the
    permutation null of Q depends on the feature's own residual multiset, so a raw maximum favours
    whichever feature happens to be heaviest-tailed.
    """
    n_perm = Qb.shape[0]
    pj = (1.0 + (Qb >= Qo[None, :]).sum(axis=0)) / (n_perm + 1.0)
    # rank 0 = largest Q within the column
    order = np.argsort(np.argsort(-Qb, axis=0), axis=0)
    pjb = (order + 1.0) / (n_perm + 1.0)
    minp_null = pjb.min(axis=1)
    minp_obs = float(pj.min())
    p_step = float((1.0 + (minp_null <= minp_obs).sum()) / (n_perm + 1.0))
    return pj, minp_obs, p_step


# =====================================================================================
# Result containers
# =====================================================================================


@dataclass
class ForwardStep:
    """One accepted step of the forward path.

    Attributes
    ----------
    p_step
        The multiplicity-corrected p-value for this step: the chance that the best of the
        candidates searched here would look this good by luck. **This is the column to read.**
        Analytic (Sidak over ``m_eff``) by default; permutation min-P if requested.
    p_perm
        Westfall-Young permutation step p-value, when ``stopping="permutation"`` was used or a
        cross-check was requested; NaN otherwise.
    p_selected_raw
        The winning feature's own conditional SKAT p-value, uncorrected. Selection-biased -- this is
        the number that must NOT be reported as significant (a 0.40 false-positive rate over 26
        correlated candidates). Kept for interpretation and for comparison with the marginal scan.
    m_eff
        Effective number of independent candidates at this step (``effective_n_tests``).
    """

    step: int
    selected: str
    Q: float
    p_step: float
    p_selected_raw: float
    m_eff: float
    n_candidates: int
    df_resid: int
    max_leverage: float
    p_perm: float = float("nan")


@dataclass
class ForwardResult:
    steps: List[ForwardStep] = field(default_factory=list)
    selected: List[str] = field(default_factory=list)
    stopped_because: str = ""
    # the step that FAILED, i.e. why the path ended. Reported even when nothing was selected --
    # "4 features are individually significant but the best of them only reaches a corrected
    # p of 0.079" is the result, and it needs these numbers, not a prose string.
    stop_p_step: float = float("nan")
    stop_p_raw: float = float("nan")
    stop_m_eff: float = float("nan")
    stop_feature: str = ""
    joint: Dict[str, float] = field(default_factory=dict)
    stability: Optional["object"] = None      # pandas DataFrame when requested
    marginal: Dict[str, float] = field(default_factory=dict)
    dropped_subjects: int = 0
    config: Dict[str, object] = field(default_factory=dict)

    def to_frame(self):
        """Selection path as a DataFrame, corrected p-value first."""
        import pandas as pd

        return pd.DataFrame([{
            "step": s.step, "selected": s.selected, "Q": s.Q,
            "p_step_corrected": s.p_step,
            "p_selected_raw_biased": s.p_selected_raw,
            "m_eff": s.m_eff,
            "p_step_permutation": s.p_perm,
            "p_marginal": self.marginal.get(s.selected, np.nan),
            "p_joint": self.joint.get(s.selected, np.nan),
            "n_candidates": s.n_candidates, "df_resid": s.df_resid,
            "max_leverage": s.max_leverage,
        } for s in self.steps])


# =====================================================================================
# The forward procedure
# =====================================================================================


def skat_conditional_forward(
    F: np.ndarray,
    K: np.ndarray,
    *,
    feature_names: Sequence[str],
    X0_base: Optional[np.ndarray] = None,
    stopping: str = "sidak",
    alpha: float = 0.05,
    max_steps: int = 10,
    rank_normalize: bool = False,
    seed: int = 0,
    n_perm: int = 10_000,
    cross_check_permutation: bool = False,
    stability_subsamples: int = 0,
    stability_fraction: float = 0.5,
    stability_path_len: Optional[int] = None,
    verbose: bool = True,
) -> ForwardResult:
    """Forward stepwise conditional SKAT with a multiplicity-corrected stopping rule.

    Parameters
    ----------
    F : (n, p)
        Cognitive features, one column each, on a COMMON complete-case subject set. Unequal
        coverage across columns would make "most significant" incomparable between features --
        use ``build_feature_matrix`` to enforce this.
    K : (n, n)
        PSD map-similarity kernel, e.g. ``map_kernel_matrix(M, kind="correlation")``.
    X0_base : (n, c), optional
        Covariates held fixed across all steps (amplitude, typicality, diagnosis dummies). An
        intercept is added automatically. Use the same adjustment set as the marginal scan's
        ``skat_adj_p`` so the two are comparable.
    stopping : {"sidak", "bonferroni", "permutation"}
        How the step's multiplicity is handled. **The default is fully analytic**, which preserves
        SKAT's defining advantage -- no permutations and no p-value floor -- because correcting for
        having searched m candidates is an ordinary multiple-comparisons problem applied to SKAT's
        own exact p-values, not a new null that has to be simulated. See
        ``step_pvalue_analytic``. ``"permutation"`` runs the Westfall-Young step-down min-P rule
        instead; it makes no assumption about how the candidates correlate, but it is slower and
        floors at ``1/(n_perm+1)``.
    n_perm, cross_check_permutation
        Permutation budget, and whether to also compute the permutation step p-value when the
        analytic route is in charge (recorded in ``p_perm``, never used for the decision).
    rank_normalize
        Inverse-normal transform every feature first. Recommended when any feature trips the
        leverage guard.
    stability_subsamples
        If > 0, also run stability selection (subsampling without replacement) and attach the
        per-feature selection frequencies. This resamples subjects but is NOT a null and yields no
        p-value -- it answers "would I have picked the same feature with different subjects?".

    Returns
    -------
    ForwardResult
        ``.steps`` is the path; read ``p_step``. ``.joint`` holds the backward/joint pass.
        ``.marginal`` holds each feature's unconditional p, so the drop from marginal to
        conditional is visible.
    """
    if stopping not in ("sidak", "bonferroni", "permutation"):
        raise ValueError(f"stopping must be 'sidak', 'bonferroni' or 'permutation', got {stopping!r}")
    F = np.asarray(F, dtype=np.float64)
    if F.ndim == 1:
        F = F[:, None]
    K = np.asarray(K, dtype=np.float64)
    names = list(feature_names)
    n, p = F.shape
    if len(names) != p:
        raise ValueError(f"feature_names has {len(names)} entries for {p} columns")
    if K.shape != (n, n):
        raise ValueError(f"K is {K.shape}, expected ({n}, {n})")
    if n < 8:
        raise ValueError(f"need at least 8 subjects, got {n}")

    if rank_normalize:
        F = rank_normalize_columns(F)

    # Constant columns carry no information and would divide by zero.
    keep = [j for j in range(p) if np.std(F[:, j]) > 0]
    if len(keep) < p and verbose:
        dropped = [names[j] for j in range(p) if j not in keep]
        print(f"[skat-cond] dropping {len(dropped)} constant feature(s): {dropped[:5]}")
    F, names = F[:, keep], [names[j] for j in keep]
    p = F.shape[1]
    if p == 0:
        return ForwardResult(stopped_because="no non-constant features")

    rng = np.random.default_rng(seed)
    use_perm = stopping == "permutation"
    result = ForwardResult(config={
        "n": n, "p": p, "stopping": stopping, "alpha": alpha, "max_steps": max_steps,
        "rank_normalize": rank_normalize, "seed": seed,
        "n_perm": n_perm if (use_perm or cross_check_permutation) else 0,
    })

    # Marginal (unconditional) p-values, for the "what redundancy costs" comparison.
    for j, nm in enumerate(names):
        result.marginal[nm] = float(skat_score_test(F[:, j], K, X0=X0_base)["p_value"])

    selected_idx: List[int] = []
    X0 = X0_base

    for step in range(1, max_steps + 1):
        cand = [j for j in range(p) if j not in selected_idx]
        if not cand:
            result.stopped_because = "all features selected"
            break

        R, P0, c = residualize(F[:, cand], X0)
        dof = max(n - c, 1)
        if dof <= 2:
            result.stopped_because = f"out of residual df (df_resid={dof})"
            break

        lev = leverage(R)
        Qo = q_vector(R, K, dof)
        if not np.isfinite(Qo).any():
            result.stopped_because = "all candidates degenerate"
            break

        # Rank candidates by Q -- identical to ranking by p, since they share P0, c and K.
        win_local = int(np.argmax(Qo))
        m_eff = effective_n_tests(R)

        # The winner's own CONDITIONAL p-value, from SKAT's exact null. Uncorrected.
        p_raw = float(skat_score_test(F[:, cand[win_local]], K, X0=X0)["p_value"])

        p_perm = float("nan")
        if use_perm or cross_check_permutation:
            Qb = _permutation_q(R, K, P0, dof, n_perm, rng)
            pj, _, p_wy = _step_down_pvalues(Qo, Qb)
            p_perm = p_wy
            if use_perm:
                win_local = int(np.argmin(pj))
                p_raw = float(skat_score_test(F[:, cand[win_local]], K, X0=X0)["p_value"])

        p_step = p_perm if use_perm else step_pvalue_analytic(p_raw, m_eff, method=stopping)

        if p_step >= alpha:
            result.stop_p_step, result.stop_p_raw, result.stop_m_eff = p_step, p_raw, m_eff
            result.stop_feature = names[cand[win_local]]
            result.stopped_because = (
                f"step {step}: corrected p={p_step:.4g} >= alpha={alpha} "
                f"(best raw p={p_raw:.3g}, m_eff={m_eff:.2f})")
            if verbose:
                print(f"[skat-cond] step {step}: stop, corrected p={p_step:.4g} "
                      f"(raw {p_raw:.3g} x m_eff {m_eff:.2f})")
            break

        win = cand[win_local]
        win_lev = float(lev[win_local])
        if win_lev > LEVERAGE_REFUSE:
            result.stopped_because = (
                f"step {step}: winner {names[win]!r} has leverage {win_lev:.2f} > "
                f"{LEVERAGE_REFUSE} -- the analytic null is unreliable; "
                "re-run with rank_normalize=True")
            if verbose:
                print(f"[skat-cond] {result.stopped_because}")
            break
        if win_lev > LEVERAGE_WARN:
            warnings.warn(
                f"{names[win]!r} has leverage {win_lev:.2f} (> {LEVERAGE_WARN}): one subject "
                "dominates its residual sum of squares, which inflates SKAT's type-I error. "
                "Consider rank_normalize=True.",
                RuntimeWarning, stacklevel=2,
            )

        result.steps.append(ForwardStep(
            step=step, selected=names[win], Q=float(Qo[win_local]), p_step=p_step,
            p_selected_raw=p_raw, m_eff=m_eff, n_candidates=len(cand), df_resid=dof,
            max_leverage=win_lev, p_perm=p_perm,
        ))
        if verbose:
            print(f"[skat-cond] step {step}: + {names[win]:<22s} Q={Qo[win_local]:8.2f} "
                  f"raw p={p_raw:.3g} -> corrected p={p_step:.4g} "
                  f"(m_eff={m_eff:.2f} of {len(cand)})")

        selected_idx.append(win)
        col = F[:, win][:, None]
        X0 = col if X0 is None else np.hstack([np.asarray(X0, dtype=np.float64).reshape(n, -1),
                                               col])
    else:
        result.stopped_because = f"hit max_steps={max_steps}"

    result.selected = [names[j] for j in selected_idx]

    # ---- backward / joint pass: COJO's "joint" step -------------------------------------
    # Re-test each selected feature with all the OTHERS in X0. An early pick can be made
    # redundant by something selected later; this is what catches it.
    if len(selected_idx) > 1:
        for j in selected_idx:
            others = [k for k in selected_idx if k != j]
            parts = [] if X0_base is None else [np.asarray(X0_base, dtype=np.float64).reshape(n, -1)]
            parts.append(F[:, others])
            result.joint[names[j]] = float(
                skat_score_test(F[:, j], K, X0=np.hstack(parts))["p_value"])
        dropped = [nm for nm, pv in result.joint.items() if pv >= alpha]
        if dropped and verbose:
            print(f"[skat-cond] joint pass: {dropped} no longer clear alpha={alpha}")
    elif len(selected_idx) == 1:
        result.joint[names[selected_idx[0]]] = result.steps[0].p_selected_raw

    # ---- stability selection -------------------------------------------------------------
    if stability_subsamples > 0:
        path_len = stability_path_len or max(len(selected_idx), 1)
        result.stability = skat_stability_selection(
            F, K, feature_names=names, path_len=path_len,
            n_subsamples=stability_subsamples, fraction=stability_fraction,
            X0_base=X0_base, seed=seed, verbose=verbose,
        )

    return result


# =====================================================================================
# Stability selection
# =====================================================================================


def skat_stability_selection(
    F: np.ndarray,
    K: np.ndarray,
    *,
    feature_names: Sequence[str],
    path_len: int,
    n_subsamples: int = 200,
    fraction: float = 0.5,
    X0_base: Optional[np.ndarray] = None,
    seed: int = 0,
    verbose: bool = True,
):
    """How often each feature is selected across subsamples of the subjects.

    A single greedy path is not reportable when two candidates are nearly tied: with
    ``r(ds_totalraw, lns_totalraw) = 0.71``, which one wins step 1 is close to a coin flip, and
    everything downstream hangs off that coin. Selection frequencies communicate the redundancy
    instead of arbitrating it -- two features splitting ~50/50 is the readable signature of "one
    construct, two instruments".

    Subsampling is WITHOUT replacement (Meinshausen & Buhlmann 2010). Bootstrapping with
    replacement would duplicate subjects, which breaks exchangeability and puts exact ties into K.

    Runs a fixed-length path rather than the permutation stopping rule -- the frequencies, not the
    per-subsample stopping decisions, are the output, and this keeps the cost linear.
    """
    import pandas as pd

    F = np.asarray(F, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    names = list(feature_names)
    n, p = F.shape
    m = max(int(round(fraction * n)), 8)
    if m >= n:
        raise ValueError(f"fraction={fraction} leaves no subjects out (m={m}, n={n})")

    rng = np.random.default_rng(seed + 991)
    counts = np.zeros(p, dtype=np.int64)
    first = np.zeros(p, dtype=np.int64)
    n_ok = 0

    for _ in range(n_subsamples):
        sub = rng.choice(n, size=m, replace=False)
        Fs, Ks = F[sub], K[np.ix_(sub, sub)]
        X0s = (None if X0_base is None
               else np.asarray(X0_base, dtype=np.float64).reshape(n, -1)[sub])
        if np.any(Fs.std(axis=0) <= 0):
            continue
        sel: List[int] = []
        X0 = X0s
        for step in range(path_len):
            cand = [j for j in range(p) if j not in sel]
            if not cand:
                break
            R, _, c = residualize(Fs[:, cand], X0)
            dof = max(m - c, 1)
            if dof <= 2:
                break
            Q = q_vector(R, Ks, dof)
            if not np.isfinite(Q).any():
                break
            win = cand[int(np.argmax(Q))]
            sel.append(win)
            col = Fs[:, win][:, None]
            X0 = col if X0 is None else np.hstack([np.asarray(X0).reshape(m, -1), col])
        if not sel:
            continue
        n_ok += 1
        counts[sel] += 1
        first[sel[0]] += 1

    denom = max(n_ok, 1)
    df = pd.DataFrame({
        "feature": names,
        "selection_freq": counts / denom,
        "first_pick_freq": first / denom,
        "n_subsamples": n_ok,
        "subsample_n": m,
        "path_len": path_len,
    }).sort_values("selection_freq", ascending=False).reset_index(drop=True)
    if verbose:
        top = df.head(5)
        print(f"[skat-cond] stability over {n_ok} subsamples of {m}/{n} subjects:")
        for _, r in top.iterrows():
            print(f"             {r.feature:<24s} selected {r.selection_freq:5.1%}  "
                  f"first {r.first_pick_freq:5.1%}")
    return df


# =====================================================================================
# Pre-specified (anchored) conditional test
# =====================================================================================


def skat_conditional_anchored(
    y: np.ndarray,
    K: np.ndarray,
    *,
    conditioning: Optional[np.ndarray] = None,
    X0_base: Optional[np.ndarray] = None,
    conditioning_names: Optional[Sequence[str]] = None,
    method: str = "imhof",
) -> Dict[str, object]:
    """One pre-specified conditional test. **The p-value is valid exactly as printed.**

    Nothing is selected, so there is no selection bias to correct: state in advance which features
    go into ``X0``, run once, report. This is the right form for a hypothesis you want to defend.

    The motivating case is the scap specificity control. Comparing ``ssp_totalraw``'s p-value to
    ``voc_totalraw``'s cannot separate them when the two share 60% of their variance; asking
    whether the maps explain ``ssp_totalraw`` *after* ``voc_totalraw`` and ``mr_totalraw`` have
    been removed from it is the question the design actually intended.

    Returns the marginal and conditional statistics side by side, plus the variance of the target
    that survives conditioning -- if that is tiny, a null conditional result means "redundant", not
    "unrelated", and the report should say so.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    K = np.asarray(K, dtype=np.float64)

    marginal = skat_score_test(y, K, X0=X0_base, method=method)
    if conditioning is None or np.size(conditioning) == 0:
        out = {
            "Q_marginal": marginal["Q"], "p_marginal": marginal["p_value"],
            "Q_conditional": marginal["Q"], "p_conditional": marginal["p_value"],
            "variance_retained": 1.0, "n": n, "conditioning": [],
            "df_resid_marginal": marginal["df_resid"], "df_resid_conditional": marginal["df_resid"],
        }
        return out

    C = np.asarray(conditioning, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    parts = [] if X0_base is None else [np.asarray(X0_base, dtype=np.float64).reshape(n, -1)]
    parts.append(C)
    X0_cond = np.hstack(parts)
    conditional = skat_score_test(y, K, X0=X0_cond, method=method)

    r_marg, _, _ = residualize(y[:, None], X0_base)
    r_cond, _, _ = residualize(y[:, None], X0_cond)
    ss_marg = float(np.sum(r_marg ** 2))
    retained = float(np.sum(r_cond ** 2)) / ss_marg if ss_marg > 0 else np.nan

    return {
        "Q_marginal": marginal["Q"], "p_marginal": marginal["p_value"],
        "Q_conditional": conditional["Q"], "p_conditional": conditional["p_value"],
        "variance_retained": retained,
        "n": n,
        "conditioning": list(conditioning_names or []),
        "df_resid_marginal": marginal["df_resid"],
        "df_resid_conditional": conditional["df_resid"],
        "max_leverage_conditional": float(leverage(r_cond)[0]),
    }


# =====================================================================================
# Building a common-subject feature matrix
# =====================================================================================


@dataclass
class FeatureMatrix:
    """Aligned cognitive features and maps on one complete-case subject set."""

    F: np.ndarray                # (n, p)
    M: np.ndarray                # (n, d)
    feature_names: List[str]
    subject_ids: List[int]
    dropped_subjects: int
    dropped_features: List[str]

    @property
    def n(self) -> int:
        return self.F.shape[0]

    @property
    def p(self) -> int:
        return self.F.shape[1]


def build_feature_matrix(
    adapter,
    samples,
    features: Sequence[str],
    *,
    collapse_to_subject: bool = True,
    verbose: bool = True,
) -> FeatureMatrix:
    """Stack features and maps on the subjects that have ALL requested features.

    A common subject set is not a nicety here. The forward procedure compares candidates by their
    Q values, and that comparison is only meaningful when they share ``n``, ``P0`` and ``K`` --
    which is also what makes ``q_vector`` valid. Features are therefore intersected, not
    individually subsetted as in the marginal scan.

    Reuses ``feature_informativeness._build_feature_override`` so missing-value handling matches
    the rest of the project exactly.
    """
    from cog2bold.analysis.feature_informativeness import _build_feature_override

    idx = list(range(len(samples)))
    if collapse_to_subject:
        # Repeated sessions break the iid assumption behind eps ~ N(0, s2 I).
        seen, keep = set(), []
        for i in idx:
            sid = samples[i].subject_id
            if sid in seen:
                continue
            seen.add(sid)
            keep.append(i)
        if len(keep) != len(idx) and verbose:
            print(f"[skat-cond] collapsed {len(idx)} -> {len(keep)} (one session per subject)")
        idx = keep

    overrides: Dict[str, Dict[int, np.ndarray]] = {}
    dropped_features: List[str] = []
    for feat in features:
        ov, _ = _build_feature_override(adapter, samples, feat)
        if ov:
            overrides[feat] = ov
        else:
            dropped_features.append(feat)
    if not overrides:
        raise ValueError("no requested feature could be resolved from the adapter")

    idx_set = set(idx)
    common = idx_set.intersection(*[set(ov) for ov in overrides.values()])
    rows = sorted(common)
    if len(rows) < 8:
        raise ValueError(
            f"only {len(rows)} subjects have all {len(overrides)} features; "
            "drop the sparsest features or test them separately")
    dropped = len(idx) - len(rows)
    if dropped and verbose:
        print(f"[skat-cond] complete-case intersection: {len(idx)} -> {len(rows)} subjects "
              f"({dropped} dropped)")
    if dropped_features and verbose:
        print(f"[skat-cond] unresolvable features skipped: {dropped_features}")

    names = list(overrides.keys())
    F = np.asarray([[float(overrides[f][i][0]) for f in names] for i in rows], dtype=np.float64)
    M = np.asarray([samples[i].map for i in rows], dtype=np.float64)
    subject_ids = [samples[i].subject_id for i in rows]
    return FeatureMatrix(F=F, M=M, feature_names=names, subject_ids=subject_ids,
                         dropped_subjects=dropped, dropped_features=dropped_features)


# =====================================================================================
# The two-stage method: SCREEN (inference) then SELECT (description)
# =====================================================================================
#
# This is the recommended entry point. It separates the two jobs that the plain forward path
# mixes together:
#
#   Stage 1 -- SCREEN. Regular (marginal) SKAT on every candidate, same covariates, BH-FDR across
#              candidates. Nothing has been selected before these tests run, so SKAT's exact null
#              holds and the q-values are valid. Output: the LEGAL SET.
#   Stage 2 -- SELECT. Within the legal set, pick a non-redundant subset. This stage makes NO
#              significance claim -- legality was settled in Stage 1 -- so the "X0 was chosen from
#              the data" problem of the forward path's later steps simply does not arise.
#
# The researcher can steer both stages by hand: choose which candidates to screen, and FORCE
# features of scientific interest into the selection; the remaining ones are then chosen
# conditionally on those.


def skat_null_mean(K: np.ndarray, X0: Optional[np.ndarray]) -> float:
    """Exact null mean of SKAT's ratio statistic: E0[Q] = tr(P0 K P0).

    Under H0, r = P0 eps with eps spherical, so r/||r|| is uniform on the unit sphere of the
    (n-c)-dimensional residual space. Then E[r'Kr / r'r] = tr(P0 K P0)/(n-c), and multiplying by
    the (n-c) in Q's definition gives tr(P0 K P0). No approximation and no estimate of sigma^2.
    """
    n = K.shape[0]
    _, P0, _ = residualize(np.zeros((n, 1)), X0)
    return float(np.trace(P0 @ K @ P0))


def skat_excess(y: np.ndarray, K: np.ndarray, X0: Optional[np.ndarray]) -> Tuple[float, float]:
    """(Q, Q - E0[Q]) -- how far a feature's statistic sits above what pure noise would give."""
    R, _, c = residualize(y, X0)
    Q = float(q_vector(R, K, max(K.shape[0] - c, 1))[0])
    return Q, Q - skat_null_mean(K, X0)


@dataclass
class ScreenResult:
    """Stage 1 output. ``table`` has one row per candidate; ``legal`` lists those with q < alpha."""

    table: "object"                # pandas DataFrame: feature, Q, excess, p, q, legal
    legal: List[str]
    alpha: float


def skat_screen(
    F: np.ndarray,
    K: np.ndarray,
    *,
    feature_names: Sequence[str],
    X0_base: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> ScreenResult:
    """Stage 1: regular SKAT on every candidate + Benjamini-Hochberg across candidates.

    The candidate list is the researcher's choice -- pass only the features you have a reason to
    ask about. Fewer, better-motivated candidates means a milder FDR correction.

    Failing the screen means *not detectably related at this sample size*, not "no information".
    """
    import pandas as pd

    from cog2bold.analysis.independence_tests import bh_fdr

    F = np.asarray(F, dtype=np.float64)
    names = list(feature_names)
    rows = []
    for j, nm in enumerate(names):
        if np.std(F[:, j]) <= 0:
            rows.append({"feature": nm, "Q": np.nan, "excess": np.nan, "p": np.nan})
            continue
        res = skat_score_test(F[:, j], K, X0=X0_base)
        rows.append({"feature": nm, "Q": res["Q"], "excess": res["Q"] - res["null_mean"],
                     "p": res["p_value"]})
    df = pd.DataFrame(rows)
    df["q"] = bh_fdr(df["p"].to_numpy())
    df["legal"] = df["q"] < alpha
    df = df.sort_values("p").reset_index(drop=True)
    return ScreenResult(table=df, legal=list(df.loc[df.legal, "feature"]), alpha=alpha)


@dataclass
class SelectResult:
    """Stage 2 output.

    ``selected`` is the final subset, forced features first. ``path`` records every round of the
    greedy search, INCLUDING the round that ended it, with each remaining candidate's unique-signal
    fraction -- so the reason the search stopped is always on the record.
    """

    selected: List[str]
    forced: List[str]
    path: "object"                 # DataFrame: round, feature, conditioned_on, Q, excess, u, chosen
    stability: Optional["object"] = None
    theta: float = 0.5
    stopped_because: str = ""


def _select_rounds(F, K, names, legal_idx, forced_idx, X0_base, theta, max_features):
    """Greedy search on the unique-signal fraction. Returns (chosen_idx, rows, stop_reason)."""
    n = F.shape[0]

    def x0_with(idx):
        parts = [] if X0_base is None else [np.asarray(X0_base, dtype=np.float64).reshape(n, -1)]
        if idx:
            parts.append(F[:, idx])
        return np.hstack(parts) if parts else None

    # the denominator: each candidate's excess under the base covariates only
    base_excess = {j: skat_excess(F[:, j], K, X0_base)[1] for j in legal_idx}

    chosen = list(forced_idx)
    rows, stop = [], ""
    rnd = 0
    while True:
        cand = [j for j in legal_idx if j not in chosen]
        if not cand:
            stop = "every legal feature was selected"
            break
        if max_features is not None and len(chosen) >= max_features:
            stop = f"reached max_features={max_features}"
            break
        rnd += 1
        X0 = x0_with(chosen)
        scored = []
        for j in cand:
            Q, ex = skat_excess(F[:, j], K, X0)
            den = base_excess[j]
            u = ex / den if den > 0 else np.nan
            scored.append((j, Q, ex, u))
        # rank by unique fraction; ties (always the case in round 1 with nothing forced, where
        # every u = 1) go to the larger excess -- at a shared X0 that is the smaller p-value
        scored.sort(key=lambda t: (-np.nan_to_num(t[3], nan=-np.inf), -t[2]))
        best = scored[0]
        accept = np.isfinite(best[3]) and best[3] >= theta
        cond = "+".join(names[k] for k in chosen) or "(base covariates only)"
        for j, Q, ex, u in scored:
            rows.append({"round": rnd, "feature": names[j], "conditioned_on": cond,
                         "Q": Q, "excess": ex, "excess_base": base_excess[j], "u": u,
                         "chosen": bool(accept and j == best[0])})
        if not accept:
            stop = (f"round {rnd}: best remaining {names[best[0]]!r} keeps only "
                    f"u={best[3]:.2f} < theta={theta} of its signal")
            break
        chosen.append(best[0])
    return chosen, rows, stop


def skat_select(
    F: np.ndarray,
    K: np.ndarray,
    *,
    feature_names: Sequence[str],
    legal: Sequence[str],
    forced: Sequence[str] = (),
    X0_base: Optional[np.ndarray] = None,
    theta: float = 0.5,
    max_features: Optional[int] = None,
    stability_subsamples: int = 0,
    stability_fraction: float = 0.5,
    seed: int = 0,
) -> SelectResult:
    """Stage 2: a non-redundant subset of the legal set, steered by the researcher.

    Criterion -- the UNIQUE-SIGNAL FRACTION of candidate j given the already-chosen set S:

        u_j(S) = ( Q_j|S - E0[Q_j|S] ) / ( Q_j - E0[Q_j] )

    numerator: how far j's statistic sits above pure noise AFTER S has been removed from j;
    denominator: the same with only the base covariates. So u = 1 means none of j's map signal is
    shared with S, u = 0 means all of it is, and u > 1 means conditioning SHARPENED it (S was
    masking it). E0[Q] = tr(P0 K P0) is exact (``skat_null_mean``), so u needs no sigma^2 and no
    distributional approximation.

    The search is greedy: add the candidate with the largest u, as long as u >= theta. With the
    default theta = 0.5 a feature is admitted only if at least half its map signal is its own.

    ``forced`` features are placed in S before the search starts -- this is how a researcher says
    "I care about spatial span; choose the rest given that". A forced feature need not be legal;
    if it is not, that is recorded but respected.

    No p-value is produced or needed: legality was established in Stage 1.
    """
    import pandas as pd

    F = np.asarray(F, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    names = list(feature_names)
    unknown = [f for f in list(legal) + list(forced) if f not in names]
    if unknown:
        raise ValueError(f"unknown feature(s): {unknown}")
    legal_idx = [names.index(f) for f in legal]
    forced_idx = [names.index(f) for f in forced]
    if not_legal := [f for f in forced if f not in legal]:
        warnings.warn(f"forced feature(s) {not_legal} did not pass the Stage 1 screen; "
                      "kept because the researcher asked for them", RuntimeWarning, stacklevel=2)

    chosen, rows, stop = _select_rounds(F, K, names, legal_idx, forced_idx, X0_base, theta,
                                        max_features)
    result = SelectResult(selected=[names[j] for j in chosen], forced=list(forced),
                          path=pd.DataFrame(rows), theta=theta, stopped_because=stop)

    if stability_subsamples > 0:
        n = F.shape[0]
        m = max(int(round(stability_fraction * n)), 8)
        rng = np.random.default_rng(seed + 7177)
        counts = np.zeros(len(names))
        ok = 0
        for _ in range(stability_subsamples):
            sub = rng.choice(n, size=m, replace=False)
            Fs, Ks = F[sub], K[np.ix_(sub, sub)]
            X0s = None if X0_base is None else np.asarray(X0_base, float).reshape(n, -1)[sub]
            if np.any(Fs[:, legal_idx + forced_idx].std(axis=0) <= 0):
                continue
            ch, _, _ = _select_rounds(Fs, Ks, names, legal_idx, forced_idx, X0s, theta,
                                      max_features)
            counts[ch] += 1
            ok += 1
        result.stability = pd.DataFrame({
            "feature": [names[j] for j in sorted(set(legal_idx) | set(forced_idx))],
        })
        result.stability["selection_freq"] = [counts[names.index(f)] / max(ok, 1)
                                               for f in result.stability.feature]
        result.stability["forced"] = result.stability.feature.isin(list(forced))
        result.stability["n_subsamples"] = ok
        result.stability["subsample_n"] = m
        result.stability = result.stability.sort_values(
            "selection_freq", ascending=False).reset_index(drop=True)
    return result


def screen_then_select(
    F: np.ndarray,
    K: np.ndarray,
    *,
    feature_names: Sequence[str],
    X0_base: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    forced: Sequence[str] = (),
    theta: float = 0.5,
    stability_subsamples: int = 200,
    seed: int = 0,
) -> Tuple[ScreenResult, SelectResult]:
    """The full two-stage method in one call. See ``skat_screen`` and ``skat_select``."""
    screen = skat_screen(F, K, feature_names=feature_names, X0_base=X0_base, alpha=alpha)
    select = skat_select(F, K, feature_names=feature_names, legal=screen.legal, forced=forced,
                         X0_base=X0_base, theta=theta,
                         stability_subsamples=stability_subsamples, seed=seed)
    return screen, select
