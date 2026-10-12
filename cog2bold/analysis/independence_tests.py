"""Independence tests with ANALYTIC null distributions (no permutation needed).

Motivation
----------
The existing informativeness scan (``feature_informativeness.py``) tests whether a cognitive
feature is related to the activation maps using a 2x2 super-pair classifier whose null is
*estimated* by shuffling the cog<->map correspondence. That works, but the H0 distribution is only
known empirically, the smallest attainable p-value is 1/(n_perm+1), and every feature costs a full
permutation sweep.

This module implements tests whose null distribution is **known analytically**:

1. ``dcor_t_test``  — distance-correlation t-test of independence in high dimension
   (Szekely & Rizzo 2013). Bias-corrected (U-centered) distance correlation; under H0 a
   transformation is Student-t distributed as the dimension grows. Detects *any* dependence
   (dCor = 0 iff independent), not just linear.

2. ``skat_score_test`` — kernel variance-component score test (SKAT; Wu et al. 2011).
   The cognitive score is the OUTCOME and the maps enter through an n x n PSD kernel. Under H0 the
   statistic is exactly a mixture of chi-squares, evaluated by Imhof (1961) numerical inversion
   (near-exact) with Liu et al. (2009) moment matching as a cross-check. Crucially it accepts
   COVARIATES, so nuisance (diagnosis group, map amplitude / data quality) is adjusted for
   analytically rather than via a restricted shuffle scope.

Map similarity: correlation vs Euclidean
---------------------------------------
Euclidean distance between maps is magnitude-sensitive, and in task-fMRI the overall map amplitude
is heavily contaminated by data quality / SNR (and, for performance-defined contrasts, by trial
counts). Correlation-based similarity removes that channel. To stay inside the rigorous Euclidean
dCor theory we do NOT feed a correlation matrix as a "distance"; instead each map is centred and
scaled to unit norm, after which

    || m_hat_i - m_hat_j ||  =  sqrt( 2 * (1 - corr(m_i, m_j)) )

i.e. ordinary Euclidean distance on the unit sphere, a monotone function of the correlation. This
keeps dCor's "zero iff independent" characterisation. (Using ``1 - r`` directly would be the
*squared* distance; squared Euclidean distance is of negative but not *strong* negative type, so
dCov degenerates to detecting only linear structure.)

Note both distances and the double/U-centering are invariant to the common mean map, so these tests
operate in "deviation space" by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2, ncx2
from scipy.stats import t as student_t

# =====================================================================================
# Map similarity
# =====================================================================================


def center_maps(M: np.ndarray) -> np.ndarray:
    """Subtract each map's own mean voxel value (what Pearson correlation does)."""
    M = np.asarray(M, dtype=np.float64)
    return M - M.mean(axis=1, keepdims=True)


def normalize_maps(M: np.ndarray) -> np.ndarray:
    """Per-map centre + unit norm, so the Gram matrix IS the correlation matrix."""
    Mc = center_maps(M)
    nrm = np.linalg.norm(Mc, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return Mc / nrm


def pairwise_euclidean(A: np.ndarray) -> np.ndarray:
    """n x n Euclidean distance matrix of the rows of A (via one Gram product)."""
    A = np.asarray(A, dtype=np.float64)
    G = A @ A.T
    sq = np.diag(G).copy()
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D


def map_distance_matrix(M: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Distance matrix between maps.

    metric="correlation": sqrt(2(1-r)) — Euclidean on unit-normalised maps (magnitude-free).
    metric="euclidean":   plain Euclidean on per-map-centred maps (magnitude-sensitive).
    """
    if metric == "correlation":
        return pairwise_euclidean(normalize_maps(M))
    if metric == "euclidean":
        return pairwise_euclidean(center_maps(M))
    raise ValueError(f"Unknown map metric {metric!r} (use 'correlation' or 'euclidean')")


def map_kernel_matrix(M: np.ndarray, kind: str = "correlation",
                      sigma_mult: float = 1.0, distance_metric: str = "correlation"):
    """PSD n x n similarity kernel between maps, for the SKAT-style test.

    kind="correlation": the subject-by-subject map correlation matrix (magnitude-free).
    kind="linear":      Gram of per-map-centred maps (magnitude-sensitive).
    kind="gaussian":    RBF exp(-d^2 / 2 sigma^2) with sigma = median pairwise distance times
                        `sigma_mult` (the median heuristic).

    The first two are LINEAR kernels, so SKAT then tests only monotone association. The Gaussian
    kernel is universal, so the implied f(Y) ranges over a rich function class and non-monotone
    dependence becomes visible. The null is unaffected: the derivation of Q/sigma^2 ~ sum_j lambda_j
    chi2_1 needs only that K be a FIXED PSD matrix, and exp(-d^2/2s^2) is PSD (Schoenberg) while the
    median-heuristic bandwidth depends on the maps only -- never on the outcome y.

    Returns the kernel; for kind="gaussian" returns (K, sigma) so the bandwidth can be recorded.
    """
    if kind == "correlation":
        Mn = normalize_maps(M)
        K = Mn @ Mn.T
    elif kind == "linear":
        Mc = center_maps(M)
        K = Mc @ Mc.T
        K /= max(float(np.trace(K)) / K.shape[0], 1e-300)  # scale for conditioning
    elif kind == "gaussian":
        D = map_distance_matrix(M, metric=distance_metric)
        n = D.shape[0]
        off = D[np.triu_indices(n, 1)]
        sigma = float(np.median(off)) * float(sigma_mult) if off.size else 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        K = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
        return 0.5 * (K + K.T), sigma
    else:
        raise ValueError(f"Unknown kernel kind {kind!r}")
    return 0.5 * (K + K.T)  # enforce exact symmetry


def quality_covariates(M: np.ndarray) -> Dict[str, np.ndarray]:
    """Scalar per-subject data-quality summaries of the maps.

    amplitude  = ||map - mean(map)||           (for z-stat maps this is ~ an SNR measure)
    typicality = mean correlation to all other subjects' maps
    Both are strong nuisance candidates: they can correlate with cognition for reasons that have
    nothing to do with spatial pattern (motion, trial counts, scan quality).
    """
    Mc = center_maps(M)
    amp = np.linalg.norm(Mc, axis=1)
    Mn = normalize_maps(M)
    R = Mn @ Mn.T
    np.fill_diagonal(R, np.nan)
    typ = np.nanmean(R, axis=1)
    return {"amplitude": amp, "typicality": typ}


# =====================================================================================
# 1. Distance correlation t-test  (Szekely & Rizzo 2013)
# =====================================================================================


def u_center(D: np.ndarray) -> np.ndarray:
    """U-centered distance matrix (gives an UNBIASED dCov^2 estimator; note the (n-2))."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n < 4:
        raise ValueError("U-centering needs n >= 4")
    rs = D.sum(axis=1)
    U = (D
         - rs[:, None] / (n - 2)
         - rs[None, :] / (n - 2)
         + D.sum() / ((n - 1) * (n - 2)))
    np.fill_diagonal(U, 0.0)
    return U


def u_inner(A: np.ndarray, B: np.ndarray) -> float:
    """U-statistic inner product (A.B); diagonals are zero so a full sum is fine."""
    n = A.shape[0]
    return float((A * B).sum() / (n * (n - 3)))


def dcor_t_test(Dx: np.ndarray, Dy: np.ndarray) -> Dict[str, float]:
    """Szekely & Rizzo (2013) t-test of independence in high dimension.

    Dx, Dy: n x n Euclidean distance matrices for the two variables.
    Returns bias-corrected dCor (R_star, an effect size), the t statistic, df and one-sided p.

    The t null is asymptotic in the DIMENSION (well suited to d >> n). It is only approximate when
    one side is low-dimensional (e.g. a single cognitive score) — validate with
    ``permutation_calibration``.
    """
    n = Dx.shape[0]
    A, B = u_center(Dx), u_center(Dy)
    aa, bb = u_inner(A, A), u_inner(B, B)
    if aa <= 0 or bb <= 0:
        return {"dcor_ustat": 0.0, "R_star": 0.0, "T": 0.0, "df": float(n * (n - 3) / 2 - 1),
                "p_value": 1.0}
    ab = u_inner(A, B)
    R = ab / np.sqrt(aa * bb)
    R = float(np.clip(R, -0.999999999, 0.999999999))
    v = n * (n - 3) / 2.0
    T = float(np.sqrt(v - 1.0) * R / np.sqrt(1.0 - R * R))
    return {"dcor_ustat": float(ab), "R_star": R, "T": T, "df": float(v - 1),
            "p_value": float(student_t.sf(T, df=v - 1.0))}


# =====================================================================================
# 2. Mixture-of-chi-squares tail probabilities
# =====================================================================================


def imhof_sf(
    q: float,
    lambdas: Sequence[float],
    abs_tol: float = 1e-10,
    max_panels: int = 4000,
    return_info: bool = False,
):
    """P(sum_j lambda_j * chi2_1 > q) by Imhof (1961) numerical inversion.

    P(Q > q) = 1/2 + (1/pi) * integral_0^inf sin(theta(u)) / (u * rho(u)) du
      theta(u) = 0.5 * sum_j arctan(lambda_j u)  -  0.5 * q * u
      rho(u)   = prod_j (1 + lambda_j^2 u^2)^(1/4)

    The integrand oscillates with half-period pi/(q/2) and decays like u^(-1-k/2) for k positive
    eigenvalues, so we integrate over oscillation-aligned panels and bound the truncated tail via
    (1 + l^2 u^2)^(-1/4) <= (l u)^(-1/2). When k is small the decay is too slow to brute-force; that
    case is either exact (all eigenvalues equal -> scaled chi-square) or falls back to Liu.
    """
    import warnings

    lam = np.asarray([float(l) for l in lambdas], dtype=np.float64)
    amax = float(np.abs(lam).max()) if lam.size else 0.0
    lam = lam[np.abs(lam) > amax * 1e-12] if amax > 0 else lam[:0]
    info = {"method": "imhof", "tail_bound": 0.0, "n_panels": 0}
    if lam.size == 0:
        p = float(q < 0)  # Q is identically 0
        return (p, info) if return_info else p
    k = int(lam.size)
    # exact when all eigenvalues coincide and are positive: Q = lam * chi2_k
    if np.all(lam > 0) and np.allclose(lam, lam[0], rtol=1e-9, atol=0.0):
        info["method"] = "exact_scaled_chi2"
        p = float(chi2.sf(q / lam[0], df=k)) if q > 0 else 1.0
        return (p, info) if return_info else p

    lam_sum = float(lam.sum())

    def _log_rho(u: float) -> float:
        return 0.25 * float(np.log1p((lam * u) ** 2).sum())

    def _integrand(u: float) -> float:
        if u <= 0.0:
            return 0.5 * (lam_sum - q)  # limit of sin(theta)/u as u -> 0
        theta = 0.5 * float(np.arctan(lam * u).sum()) - 0.5 * q * u
        return float(np.sin(theta) * np.exp(-_log_rho(u)) / u)

    def _envelope(u: float) -> float:
        """|integrand| <= exp(-log_rho(u))/u; used directly rather than via the loose
        prod|lam|^{-1/2} bound, which overflows once there are thousands of eigenvalues
        (e.g. the n^2 outer-product weights of the spectral dCor null)."""
        if u <= 0.0:
            return np.inf
        return float(np.exp(-_log_rho(u)) / u)

    # Truncation point: walk up until the envelope itself is negligible. This is tight and
    # numerically safe for any number of eigenvalues.
    U = 1.0 / max(float(np.abs(lam).max()), 1e-12)
    for _ in range(400):
        if _envelope(U) <= abs_tol:
            break
        U *= 1.6
    w = 0.5 * abs(q)
    width = (np.pi / w) if w > 0 else U
    if width > 0 and U / width > max_panels:
        U = max_panels * width
    tail = _envelope(U) * U / np.pi  # crude remaining-mass bound beyond U
    if tail > 1e-4:  # decay too slow to resolve numerically -> moment match instead
        info.update(method="liu_fallback", tail_bound=tail)
        p = liu_sf(q, lam) if np.all(lam > 0) else float("nan")
        return (p, info) if return_info else p

    # panel edges: geometric near 0 (arctan transition) + uniform (q-driven oscillation)
    small = min(width, 1.0 / max(float(np.abs(lam).max()), 1e-12))
    edges = [0.0] + list(np.geomspace(small * 1e-6, small, 16))
    if w > 0:
        n_unif = int(np.ceil((U - small) / width))
        if n_unif > 0:
            edges += list(small + width * np.arange(1, n_unif + 1))
    else:
        edges += list(np.geomspace(small * 2, U, 64))
    edges = sorted({min(e, U) for e in edges})

    total = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # accuracy is governed by our own tail bound
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= lo:
                continue
            val, _ = quad(_integrand, lo, hi, limit=200, epsabs=1e-13, epsrel=1e-10)
            total += val
    info.update(tail_bound=tail, n_panels=len(edges) - 1)
    p = float(min(max(0.5 + total / np.pi, 0.0), 1.0))
    return (p, info) if return_info else p


def liu_sf(q: float, lambdas: Sequence[float]) -> float:
    """P(sum lambda_j chi2_1 > q) by Liu et al. (2009) moment matching (fast cross-check)."""
    lam = np.asarray([float(l) for l in lambdas], dtype=np.float64)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 1.0
    c1, c2 = lam.sum(), (lam ** 2).sum()
    c3, c4 = (lam ** 3).sum(), (lam ** 4).sum()
    if c2 <= 0:
        return 1.0
    s1 = c3 / c2 ** 1.5
    s2 = c4 / (c2 ** 2)
    muQ, sigmaQ = c1, np.sqrt(2.0 * c2)
    if s1 ** 2 > s2:
        a = 1.0 / (s1 - np.sqrt(s1 ** 2 - s2))
        delta = s1 * a ** 3 - a ** 2
        dof = a ** 2 - 2.0 * delta
    else:
        a = 1.0 / s1 if s1 > 0 else np.inf
        delta = 0.0
        dof = c2 ** 3 / c3 ** 2 if c3 > 0 else 1.0
    muX, sigmaX = dof + delta, np.sqrt(2.0) * a
    z = (q - muQ) / sigmaQ * sigmaX + muX
    if delta > 0:
        return float(ncx2.sf(z, df=max(dof, 1e-8), nc=delta))
    return float(chi2.sf(z, df=max(dof, 1e-8)))


def mixture_chi2_sf(q: float, lambdas: Sequence[float], method: str = "imhof") -> float:
    if method == "imhof":
        try:
            return imhof_sf(q, lambdas)
        except Exception:  # noqa: BLE001 — fall back rather than fail a whole sweep
            return liu_sf(q, lambdas)
    if method == "liu":
        return liu_sf(q, lambdas)
    raise ValueError(f"Unknown method {method!r}")


def dcor_spectral_test(Dx: np.ndarray, Dy: np.ndarray, method: str = "imhof",
                       max_weights: int = 400) -> Dict[str, float]:
    """DIAGNOSTIC ONLY -- the naive spectral plug-in null for dCov. **Do not use its p-value.**

    Theory (Szekely, Rizzo & Bakirov 2007; equivalently the HSIC null of Gretton et al. 2008, since
    dCov is HSIC with distance-induced kernels): for FIXED dimensions,

        n * dCov^2_n  ->d  sum_k sum_l  lambda_k * eta_l * chi2_1(k,l)

    with lambda_k, eta_l the eigenvalues of the two double-centred distance matrices. That is the
    right *shape* (right-skewed, heavy-tailed) and explains why the Szekely-Rizzo t-approximation
    fails in the tail. But the obvious finite-sample plug-in -- take the sample eigenvalues, form all
    n^2 products, treat them as independent chi2_1 weights -- is NOT VALID, and we measured why:

        weights sum      = 11.70   vs permutation mean     11.86   (mean is fine)
        2 * sum(weights^2) = 1.297 vs permutation variance   0.0797 (variance ~16x too big)

    The reason is that the n^2 eigenvector overlaps (u_k . v_l)^2 are *not* independent: U^T V is
    orthogonal, so each row and column of squared overlaps sums to 1. Treating them as n^2 free
    chi2_1 variables massively over-states the randomness. Empirically this gives
    P(p<0.10) = 0.575 instead of 0.10 and produced a false positive (p=0.043 where the permutation
    p was 0.555).

    Kept only to document the negative result and to expose the weights for inspection. Use
    ``dcor_moment_test`` (validated) or a full permutation test instead.
    """
    Dx = np.asarray(Dx, dtype=np.float64)
    Dy = np.asarray(Dy, dtype=np.float64)
    n = Dx.shape[0]
    if n < 4:
        return {"dcov2": float("nan"), "p_value": float("nan"), "n_weights": 0,
                "valid": False}

    def _dcenter(D: np.ndarray) -> np.ndarray:
        return D - D.mean(axis=1, keepdims=True) - D.mean(axis=0, keepdims=True) + D.mean()

    A, B = _dcenter(Dx), _dcenter(Dy)
    stat = float((A * B).sum() / n)
    la = np.linalg.eigvalsh(0.5 * (A + A.T)) / n
    et = np.linalg.eigvalsh(0.5 * (B + B.T)) / n
    if la.size == 0 or et.size == 0:
        return {"dcov2": stat, "p_value": float("nan"), "n_weights": 0, "valid": False}
    w = np.outer(la, et).ravel()
    order = np.argsort(np.abs(w))[::-1]
    w = w[order][:min(w.size, max_weights)]
    return {
        "dcov2": stat,
        "p_value": float(mixture_chi2_sf(stat, w, method=method)),   # NOT calibrated
        "n_weights": int(w.size),
        "valid": False,
    }


def dcor_moment_test(
    Dx: np.ndarray, Dy: np.ndarray, n_perm: int = 200, seed: int = 0,
) -> Dict[str, float]:
    """dCov test with a CONTINUOUS p-value: Gamma fitted to a short permutation null.

    Motivation: the Szekely-Rizzo t-approximation is invalid in the tail (its premise is that BOTH
    dimensions grow; our cognitive side is a scalar), while a plain permutation p-value is floored at
    1/(P+1) -- which is exactly the resolution problem that motivated leaving permutation tests in the
    first place. The naive spectral plug-in is not valid either (see ``dcor_spectral_test``).

    This is the practical middle road, and the same device Gretton et al. (2008) use for HSIC: the
    null of dCov^2 is a non-negative, right-skewed, roughly Gamma-shaped quantity, so estimate only
    its FIRST TWO MOMENTS (cheap and stable -- a few hundred permutations suffice) and read the tail
    off the fitted Gamma:

        shape = mean^2 / var ,   scale = var / mean ,   p = P( Gamma(shape, scale) > stat ) .

    Cost is ~1/5 of a 1000-permutation test, and the p-value is continuous -- no 1/(P+1) floor.
    """
    Dx = np.asarray(Dx, dtype=np.float64)
    Dy = np.asarray(Dy, dtype=np.float64)
    n = Dx.shape[0]
    if n < 8:
        return {"dcov2": float("nan"), "p_value": float("nan"), "null_mean": float("nan"),
                "null_var": float("nan"), "n_perm": 0}

    def _dcenter(D: np.ndarray) -> np.ndarray:
        return D - D.mean(axis=1, keepdims=True) - D.mean(axis=0, keepdims=True) + D.mean()

    B = _dcenter(Dy)
    stat = float((_dcenter(Dx) * B).sum() / n)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        pi = rng.permutation(n)
        null[k] = float((_dcenter(Dx[np.ix_(pi, pi)]) * B).sum() / n)
    m, v = float(null.mean()), float(null.var(ddof=1))
    if not np.isfinite(m) or not np.isfinite(v) or v <= 0 or m <= 0:
        return {"dcov2": stat, "p_value": float("nan"), "null_mean": m, "null_var": v,
                "n_perm": int(n_perm)}
    shape, scale = m * m / v, v / m
    from scipy.stats import gamma as _gamma
    return {
        "dcov2": stat,
        "p_value": float(_gamma.sf(stat, a=shape, scale=scale)),
        "null_mean": m,
        "null_var": v,
        "z_score": float((stat - m) / np.sqrt(v)),
        "n_perm": int(n_perm),
    }


def dcor_permutation_test(
    Dx: np.ndarray, Dy: np.ndarray, n_perm: int = 20000, seed: int = 0,
) -> Dict[str, float]:
    """dCor with an EXACT permutation null -- the recommended p-value for dCor.

    Every parametric shortcut we tried fails in the tail (see the report): the Szekely-Rizzo
    t-approximation is 48-400x off, the naive spectral chi-square plug-in has ~16x too much variance,
    and a Gamma moment-match is calibrated at alpha~0.05 but still ~300x off far out. Permutation is
    the only reference that is right, and for dCor it is CHEAP:

    permuting the subjects permutes the distance matrix, and U-centering commutes with that
    permutation, so `u_center(Dx[pi][:,pi]) = u_center(Dx)[pi][:,pi]`. The centred matrices are
    therefore computed ONCE and each permutation costs a single fancy-indexed multiply -- O(n^2), no
    distance recomputation. 20,000 permutations run in well under a second at n<=250, giving a floor
    of 5e-5 (and 200,000 gives 5e-6), far below anything BH over a few dozen features needs.
    """
    Dx = np.asarray(Dx, dtype=np.float64)
    Dy = np.asarray(Dy, dtype=np.float64)
    n = Dx.shape[0]
    if n < 4:
        return {"R_star": float("nan"), "p_value": float("nan"), "n_perm": 0,
                "p_floor": float("nan")}
    A, B = u_center(Dx), u_center(Dy)
    aa, bb = u_inner(A, A), u_inner(B, B)
    if aa <= 0 or bb <= 0:
        # degenerate: the feature (or the maps) is constant over these subjects
        return {"R_star": 0.0, "p_value": 1.0, "n_perm": 0, "p_floor": float("nan")}
    denom = np.sqrt(aa * bb)
    obs = u_inner(A, B) / denom
    rng = np.random.default_rng(seed)
    # aa and bb are permutation-invariant, so only the cross term moves
    ge = 0
    for _ in range(n_perm):
        pi = rng.permutation(n)
        if u_inner(A[np.ix_(pi, pi)], B) / denom >= obs:
            ge += 1
    return {
        "R_star": float(obs),
        "p_value": float((1 + ge) / (1 + n_perm)),
        "n_perm": int(n_perm),
        "p_floor": float(1.0 / (1 + n_perm)),
    }


# =====================================================================================
# 3. SKAT-style kernel variance-component score test
# =====================================================================================


def skat_score_test(
    y: np.ndarray,
    K: np.ndarray,
    X0: Optional[np.ndarray] = None,
    method: str = "imhof",
) -> Dict[str, float]:
    """Variance-component score test that the map-kernel K explains the outcome y.

    y  : (n,)    outcome — here the COGNITIVE score.
    K  : (n, n)  PSD similarity between maps.
    X0 : (n, c)  covariates (an intercept is added if absent) — e.g. diagnosis dummies,
                 map amplitude / typicality. The test then asks whether map *pattern*
                 similarity explains cognition BEYOND these.

    Exact null via a ratio of quadratic forms. With r = P0 y, P0 = I - X0 X0^+, and
    y = X0 b + eps, eps ~ N(0, s2 I) under H0, the scale-free statistic

        Q = (n - c) * r'K r / r'r

    satisfies  {Q > q}  <=>  {r' [ (n-c) K - q I ] r > 0}.  So

        P(Q > q_obs) = P( sum_j mu_j * chi2_1 > 0 ),   mu_j = eigvals( P0 [ (n-c)K - q_obs I ] P0 )

    where the mu_j have mixed signs and s2 cancels entirely — no variance estimate is needed.
    (Plugging an estimated s2 into a plain quadratic form badly under-disperses the statistic,
    because r'Kr and r'r are strongly correlated; that route is ~7x too conservative here.)
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    K = np.asarray(K, dtype=np.float64)
    if X0 is None:
        X0 = np.ones((n, 1))
    else:
        X0 = np.asarray(X0, dtype=np.float64)
        if X0.ndim == 1:
            X0 = X0[:, None]
        if not np.allclose(X0[:, 0], 1.0):
            X0 = np.hstack([np.ones((n, 1)), X0])
    # residual projection (pinv handles any rank deficiency in the covariates)
    P0 = np.eye(n) - X0 @ np.linalg.pinv(X0)
    c = int(np.linalg.matrix_rank(X0))
    r = P0 @ y
    rr = float(r @ r)
    dof_resid = max(n - c, 1)
    if rr <= 0:
        return {"Q": 0.0, "p_value": 1.0, "n_lambda": 0, "df_resid": dof_resid,
                "null_mean": 0.0}
    Q = dof_resid * float(r @ K @ r) / rr

    A = P0 @ (dof_resid * K - Q * np.eye(n)) @ P0
    A = 0.5 * (A + A.T)
    mu = np.linalg.eigvalsh(A)
    p = mixture_chi2_sf(0.0, mu, method=method)

    # null mean of Q, for interpretation/comparability across features
    PKP = P0 @ K @ P0
    lam0 = np.linalg.eigvalsh(0.5 * (PKP + PKP.T))
    lam0 = lam0[lam0 > max(float(lam0.max()), 0.0) * 1e-10] if lam0.size else lam0
    return {
        "Q": Q,
        "p_value": float(p),
        "n_lambda": int(np.sum(np.abs(mu) > np.abs(mu).max() * 1e-12)) if mu.size else 0,
        "df_resid": int(dof_resid),
        "null_mean": float(lam0.sum()) if lam0.size else 0.0,
    }


# =====================================================================================
# 4. Multiple testing + calibration
# =====================================================================================


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values (NaNs preserved)."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(m) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    res = np.empty(m)
    res[order] = q
    out[ok] = res
    return out


def permutation_calibration(
    y: np.ndarray,
    Dy_or_K: np.ndarray,
    kind: str = "dcor",
    X0: Optional[np.ndarray] = None,
    n_perm: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Check whether the ANALYTIC null is calibrated, by permuting the outcome.

    Reports the empirical p for the observed statistic and, more importantly, the realised
    type-I error rate of the analytic test at level `alpha` (should be ~alpha) plus the mean/sd
    of the analytic t statistic under permutation (should be ~0 / ~1 for the dCor t-test).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    rng = np.random.default_rng(seed)

    def analytic(yy: np.ndarray) -> Tuple[float, float]:
        if kind == "dcor":
            res = dcor_t_test(pairwise_euclidean(yy[:, None]), Dy_or_K)
            return res["p_value"], res["T"]
        res = skat_score_test(yy, Dy_or_K, X0=X0)
        return res["p_value"], res["Q"]

    p_obs, s_obs = analytic(y)
    ps, ss = [], []
    for _ in range(n_perm):
        pp, sss = analytic(y[rng.permutation(n)])
        ps.append(pp)
        ss.append(sss)
    ps = np.asarray(ps)
    ss = np.asarray(ss)
    return {
        "p_analytic": float(p_obs),
        "p_permutation": float((1 + int((ss >= s_obs).sum())) / (1 + n_perm)),
        "type1_at_alpha": float((ps < alpha).mean()),
        "alpha": float(alpha),
        "null_stat_mean": float(ss.mean()),
        "null_stat_sd": float(ss.std(ddof=1)),
        "n_perm": int(n_perm),
    }


# =====================================================================================
# 5. Orchestration over a dataset
# =====================================================================================


@dataclass
class IndependenceTestArgs:
    """Inputs for an analytic-null independence scan (no trained model needed)."""

    adapter_name: Optional[str] = None
    data_root: Optional[Path] = None
    run_dir: Optional[Path] = None          # borrow dataset settings from a run's config.json
    single_contrast: Optional[object] = None
    stim_encoding: str = "contrast"
    contrast_pairs: Optional[List[Tuple[str, str]]] = None
    adapter_kwargs: Optional[dict] = None

    cog_base_cols_to_scan: Optional[List[str]] = None
    scan_all_numeric: bool = False

    # "correlation" (magnitude-free, recommended) and/or "euclidean" (magnitude-sensitive)
    map_metrics: List[str] = field(default_factory=lambda: ["correlation", "euclidean"])
    # also run SKAT with a Gaussian (RBF) kernel -> sensitive to NON-MONOTONE dependence, and the
    # exact chi2-mixture null still holds because it only needs a fixed PSD K
    skat_gaussian: bool = True
    gaussian_sigma_mult: float = 1.0
    # permutation p-value for dCor (its t-approximation is invalid in the tail); 0 disables
    dcor_perm: int = 999
    # adjust for map amplitude + typicality (data quality) in the SKAT test
    include_quality_covariates: bool = True
    # adjust for diagnosis group (if the adapter exposes it)
    include_group_covariates: bool = True
    # one row per subject (repeated sessions break the iid assumption)
    collapse_to_subject: bool = True

    n_calibration_permutations: int = 0     # 0 = skip; ~500 is a good check
    calibration_features: Optional[List[str]] = None  # None -> first feature
    seed: int = 0
    output_dir: Optional[Path] = None


def _group_dummies(adapter, subject_ids: Sequence[int]) -> Optional[np.ndarray]:
    """One-hot (first level dropped) diagnosis covariates, if the adapter exposes them."""
    diag_by_sid = getattr(adapter, "_diag_by_sid", None)
    if not diag_by_sid:
        return None
    labels = [diag_by_sid.get(int(s)) for s in subject_ids]
    levels = sorted({l for l in labels if l is not None})
    if len(levels) < 2:
        return None
    return np.asarray(
        [[1.0 if l == lev else 0.0 for lev in levels[1:]] for l in labels], dtype=np.float64
    )


def run_independence_tests(args: IndependenceTestArgs) -> Dict[str, object]:
    """Run the analytic-null independence tests for every feature x map-metric."""
    import pandas as pd

    from cog2bold.config import RunConfig
    from cog2bold.datasets import create_adapter, filter_samples_by_contrast
    from cog2bold.io import save_json
    from cog2bold.workspace import cache_root
    from cog2bold.analysis.feature_informativeness import (
        _build_feature_override,
        _enumerate_numeric_cog_cols,
    )

    adapter_name = args.adapter_name
    data_root = args.data_root
    single_contrast = args.single_contrast
    stim_encoding = args.stim_encoding
    contrast_pairs = args.contrast_pairs
    adapter_kwargs = dict(args.adapter_kwargs or {})
    cog_cols = args.cog_base_cols_to_scan

    if args.run_dir is not None:
        rc = RunConfig.load(Path(args.run_dir) / "config.json")
        adapter_name = adapter_name or rc.data_config.adapter_name
        data_root = data_root or rc.data_root
        stim_encoding = stim_encoding or rc.data_config.stim_encoding
        contrast_pairs = contrast_pairs or rc.data_config.contrast_pairs
        if single_contrast is None:
            single_contrast = rc.data_config.single_contrast
        if not adapter_kwargs and rc.data_config.adapter_kwargs:
            adapter_kwargs = dict(rc.data_config.adapter_kwargs)
        if cog_cols is None and not args.scan_all_numeric:
            cog_cols = list(rc.data_config.cog_base_cols)
    if adapter_name is None:
        raise ValueError("adapter_name is required (or pass run_dir).")
    if data_root is None:
        from cog2bold.datasets.registry import _REGISTRY
        cls = _REGISTRY.get(adapter_name)
        data_root = cls.default_data_root() if cls is not None else None
        if data_root is None:
            raise ValueError(f"data_root missing and {adapter_name!r} has no default_data_root().")

    adapter = create_adapter(adapter_name, data_root=Path(data_root), cog_base_cols=[],
                             **adapter_kwargs)
    samples = adapter.materialize(
        cog_base_cols=[], stim_encoding_mode=stim_encoding, contrast_pairs=contrast_pairs,
        subject_session_filter=None, cache_dir=cache_root() / adapter_name,
    )
    if single_contrast is not None:
        samples = filter_samples_by_contrast(samples, single_contrast, adapter)
    print(f"[indep] {adapter_name}: {len(samples)} maps materialized")

    if cog_cols is None:
        cog_cols = _enumerate_numeric_cog_cols(adapter)
    print(f"[indep] scanning {len(cog_cols)} features: {list(cog_cols)}")

    # group samples by contrast so each contrast is tested separately
    by_contrast: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        key = s.contrast_name if s.is_contrast else adapter.global_labels[s.stim_id]
        by_contrast.setdefault(key, []).append(i)

    rows: List[dict] = []
    calibration: Dict[str, dict] = {}
    calib_feats = args.calibration_features or ([cog_cols[0]] if cog_cols else [])

    # Per-feature values do NOT depend on the contrast, so build them once (this used to run
    # inside the contrast loop, repeating the same work for every contrast).
    overrides: Dict[str, dict] = {}
    for feat in cog_cols:
        ov, _ = _build_feature_override(adapter, samples, feat)
        if ov:
            overrides[feat] = ov
    print(f"[indep] resolved values for {len(overrides)}/{len(cog_cols)} features")

    for contrast, idx_all in sorted(by_contrast.items()):
        idx = list(idx_all)
        if args.collapse_to_subject:
            seen, keep = set(), []
            for i in idx:
                sid = samples[i].subject_id
                if sid in seen:
                    continue
                seen.add(sid)
                keep.append(i)
            if len(keep) != len(idx):
                print(f"[indep] {contrast}: collapsed {len(idx)} -> {len(keep)} "
                      "(one session per subject; repeated sessions break independence)")
            idx = keep
        if len(idx) < 8:
            print(f"[indep] {contrast}: only {len(idx)} subjects — skipping")
            continue

        M = np.asarray([samples[i].map for i in idx], dtype=np.float64)
        subject_ids = [samples[i].subject_id for i in idx]
        qual = quality_covariates(M)
        groups = _group_dummies(adapter, subject_ids) if args.include_group_covariates else None
        print(f"[indep] {contrast}: n={len(idx)} subjects, d={M.shape[1]} voxels")

        # precompute per-metric distance + kernel once (shared across all features)
        prep = {}
        for metric in args.map_metrics:
            entry = {
                "D": map_distance_matrix(M, metric=metric),
                "K": map_kernel_matrix(M, kind="correlation" if metric == "correlation"
                                       else "linear"),
            }
            if args.skat_gaussian:
                entry["Kg"], entry["sigma"] = map_kernel_matrix(
                    M, kind="gaussian", sigma_mult=args.gaussian_sigma_mult,
                    distance_metric=metric)
            prep[metric] = entry

        for feat in cog_cols:
            override = overrides.get(feat)
            if not override:
                continue
            pos = [j for j, i in enumerate(idx) if i in override]
            if len(pos) < 8:
                continue
            y = np.asarray([float(override[idx[j]][0]) for j in pos])
            if np.std(y) <= 0:
                continue
            sel = np.asarray(pos, dtype=int)

            # covariates for the adjusted SKAT test
            cov_parts, cov_names = [], []
            if args.include_quality_covariates:
                cov_parts += [qual["amplitude"][sel][:, None], qual["typicality"][sel][:, None]]
                cov_names += ["amplitude", "typicality"]
            if groups is not None:
                cov_parts.append(groups[sel])
                cov_names.append("group")
            X0_adj = np.hstack(cov_parts) if cov_parts else None

            for metric in args.map_metrics:
                D = prep[metric]["D"][np.ix_(sel, sel)]
                K = prep[metric]["K"][np.ix_(sel, sel)]
                Dx = pairwise_euclidean(y[:, None])
                dc = dcor_t_test(Dx, D)
                sk = skat_score_test(y, K, X0=None)
                sk_adj = (skat_score_test(y, K, X0=X0_adj) if X0_adj is not None
                          else {"Q": np.nan, "p_value": np.nan})
                # dCor's t-approximation is invalid in the tail -> also give a permutation p
                dc_pm = (dcor_permutation_test(Dx, D, n_perm=args.dcor_perm, seed=args.seed)
                         if args.dcor_perm else {})
                # SKAT with a Gaussian kernel: sensitive to NON-MONOTONE dependence, same exact null
                if args.skat_gaussian:
                    Kg = prep[metric]["Kg"][np.ix_(sel, sel)]
                    skg = skat_score_test(y, Kg, X0=None)
                    skg_adj = (skat_score_test(y, Kg, X0=X0_adj) if X0_adj is not None
                               else {"Q": np.nan, "p_value": np.nan})
                    gsigma = prep[metric]["sigma"]
                else:
                    skg = skg_adj = {"Q": np.nan, "p_value": np.nan}
                    gsigma = np.nan
                rows.append({
                    "contrast": contrast, "feature": feat, "map_metric": metric,
                    "n": len(pos),
                    "dcor_R_star": dc["R_star"], "dcor_T": dc["T"], "dcor_p": dc["p_value"],
                    "dcor_perm_p": dc_pm.get("p_value", np.nan),
                    "dcor_perm_floor": dc_pm.get("p_floor", np.nan),
                    "skat_Q": sk["Q"], "skat_p": sk["p_value"],
                    "skat_null_mean": sk["null_mean"],
                    "skat_adj_Q": sk_adj["Q"], "skat_adj_p": sk_adj["p_value"],
                    "skat_gauss_Q": skg["Q"], "skat_gauss_p": skg["p_value"],
                    "skat_gauss_adj_Q": skg_adj["Q"], "skat_gauss_adj_p": skg_adj["p_value"],
                    "gauss_sigma": gsigma,
                    "covariates": "+".join(cov_names) if cov_names else "",
                })

            if args.n_calibration_permutations and feat in calib_feats:
                m0 = args.map_metrics[0]
                calibration[f"{contrast}|{feat}|dcor|{m0}"] = permutation_calibration(
                    y, prep[m0]["D"][np.ix_(sel, sel)], kind="dcor",
                    n_perm=args.n_calibration_permutations, seed=args.seed)
                calibration[f"{contrast}|{feat}|skat|{m0}"] = permutation_calibration(
                    y, prep[m0]["K"][np.ix_(sel, sel)], kind="skat", X0=None,
                    n_perm=args.n_calibration_permutations, seed=args.seed)

    if not rows:
        print("[indep] no results produced.")
        return {"results": [], "calibration": calibration}

    df = pd.DataFrame(rows)
    # BH-FDR across features, within (contrast, map_metric)
    for col, qcol in [("dcor_p", "dcor_q"), ("dcor_perm_p", "dcor_perm_q"),
                      ("skat_p", "skat_q"), ("skat_adj_p", "skat_adj_q"),
                      ("skat_gauss_p", "skat_gauss_q"),
                      ("skat_gauss_adj_p", "skat_gauss_adj_q")]:
        df[qcol] = np.nan
        for (_, _), sub in df.groupby(["contrast", "map_metric"]):
            df.loc[sub.index, qcol] = bh_fdr(sub[col].to_numpy())

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "independence_tests.csv", index=False)
        save_json({"args": {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()},
                   "calibration": calibration},
                  out_dir / "independence_tests_summary.json")
        print(f"[indep] saved {out_dir / 'independence_tests.csv'}")
    return {"results": df, "calibration": calibration}
