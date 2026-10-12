"""Joint plausibility region of the cognitive inputs (the model's training support).

Per-feature min/max boxes let a user build profiles that are in range on every slider yet
jointly unlike any training subject (e.g. very high vocabulary with very low matrix reasoning
when the two correlate). The predictor was never validated there, so its maps are unreliable.

Model: a Gaussian copula. Each continuous feature is mapped to normal scores through its
(piecewise-linear) empirical CDF, then a Ledoit-Wolf-shrunk Gaussian is fitted to the latent
vectors. The plausible region is the Mahalanobis ellipsoid ``d²(u) <= tau``, with ``tau`` the
empirical ``quantile`` of the training subjects' own leave-one-out d² (NOT a fixed "2 SD":
under a Gaussian d² ~ chi²_p, so for p=10 the median real subject already sits at d ≈ 3.2).

Binary / constant columns (<= 2 unique values, e.g. one-hot group dummies) are left out of the
ellipsoid and never constrained.

Because the transform is monotone per feature, "move one slider with the others fixed" stays a
1-D problem in latent space, where d² is a parabola in u_k — so each slider's feasible range is
a closed-form interval (``feasible_interval``), centred on the conditional mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

# radial projection lands this fraction of the way to the boundary (in d, not d²), so the
# projected point keeps a non-degenerate feasible interval on every slider
_PROJECT_MARGIN = 0.999


@dataclass
class _Marginal:
    """Monotone piecewise-linear map raw value <-> normal score, linear tails beyond the data."""

    knots: np.ndarray   # unique sorted training values
    scores: np.ndarray  # Φ⁻¹(mid-rank / (n+1)) at each knot, strictly increasing
    slope: float        # tail slope d(score)/d(raw), from the full data range

    @classmethod
    def fit(cls, col: np.ndarray) -> "_Marginal":
        v, counts = np.unique(col, return_counts=True)
        below = np.concatenate([[0], np.cumsum(counts)[:-1]])
        mid_rank = below + (counts + 1) / 2.0
        scores = norm.ppf(mid_rank / (len(col) + 1))
        slope = (scores[-1] - scores[0]) / (v[-1] - v[0])
        return cls(knots=v, scores=scores, slope=float(slope))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        u = np.interp(x, self.knots, self.scores)
        u = np.where(x < self.knots[0], self.scores[0] + self.slope * (x - self.knots[0]), u)
        return np.where(x > self.knots[-1], self.scores[-1] + self.slope * (x - self.knots[-1]), u)

    def inverse(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        x = np.interp(u, self.scores, self.knots)
        x = np.where(u < self.scores[0], self.knots[0] + (u - self.scores[0]) / self.slope, x)
        return np.where(u > self.scores[-1], self.knots[-1] + (u - self.scores[-1]) / self.slope, x)


def _fit_gaussian(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lw = LedoitWolf().fit(U)
    return lw.location_, lw.covariance_


def _d2(U: np.ndarray, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
    D = np.atleast_2d(U) - mu
    return np.einsum("ij,jk,ik->i", D, P, D)


@dataclass
class CogSupport:
    n_features: int
    cont_idx: np.ndarray          # indices of the constrained (continuous) features
    marginals: List[_Marginal]    # one per cont_idx entry
    mu: np.ndarray                # latent mean
    cov: np.ndarray               # latent shrunk covariance
    P: np.ndarray                 # its inverse (precision)
    tau: float                    # d² limit
    quantile: float
    train_d2: np.ndarray          # leave-one-out d² of each training subject (typicality reference)
    X_train: np.ndarray           # raw training profiles, one row per subject

    # ---------------------------------------------------------------- fitting ----

    @classmethod
    def fit(cls, X: np.ndarray, quantile: float = 0.95) -> "CogSupport":
        """Fit on raw training profiles, ONE ROW PER SUBJECT (n_subjects, n_features)."""
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        cont = np.array([j for j in range(p) if len(np.unique(X[:, j])) > 2], dtype=int)
        marginals = [_Marginal.fit(X[:, j]) for j in cont]
        if cont.size == 0 or n < 3:
            return cls(p, np.zeros(0, int), [], np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0)),
                       np.inf, quantile, np.zeros(n), X)
        U = np.column_stack([m.forward(X[:, j]) for m, j in zip(marginals, cont)])
        mu, cov = _fit_gaussian(U)
        P = np.linalg.inv(cov)
        # leave-one-out d², so tau isn't optimistic from scoring subjects against their own fit
        loo = np.empty(n)
        keep = np.ones(n, bool)
        for i in range(n):
            keep[i] = False
            mu_i, cov_i = _fit_gaussian(U[keep])
            loo[i] = _d2(U[i], mu_i, np.linalg.inv(cov_i))[0]
            keep[i] = True
        tau = float(np.quantile(loo, quantile))
        return cls(p, cont, marginals, mu, cov, P, tau, quantile, loo, X)

    # -------------------------------------------------------------- transforms ----

    @property
    def is_trivial(self) -> bool:
        """No continuous features -> nothing to constrain."""
        return self.cont_idx.size == 0

    def to_latent(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.array([m.forward(x[j]) for m, j in zip(self.marginals, self.cont_idx)])

    def from_latent(self, u: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Write latent coords ``u`` back into a copy of raw profile ``base``."""
        x = np.asarray(base, dtype=np.float64).copy()
        for k, (m, j) in enumerate(zip(self.marginals, self.cont_idx)):
            x[j] = m.inverse(u[k])
        return x

    def center(self) -> np.ndarray:
        """Raw profile at the latent mean (binary features at their training mean)."""
        return self.from_latent(self.mu, self.X_train.mean(axis=0))

    # ----------------------------------------------------------------- queries ----

    def mahalanobis2(self, x: np.ndarray) -> float:
        if self.is_trivial:
            return 0.0
        return float(_d2(self.to_latent(x), self.mu, self.P)[0])

    def is_inside(self, x: np.ndarray) -> bool:
        return self.mahalanobis2(x) <= self.tau * (1 + 1e-9)

    def percentile(self, x: np.ndarray) -> float:
        """% of training subjects that are more typical (smaller d²) than ``x``."""
        if self.is_trivial:
            return 0.0
        return float(100.0 * np.mean(self.train_d2 < self.mahalanobis2(x)))

    def feasible_interval(self, x: np.ndarray, j: int) -> Optional[Tuple[float, float]]:
        """Raw-unit range of feature ``j`` that keeps ``x`` inside, others held fixed.

        ``None`` for unconstrained (binary/constant) features. If the other features alone
        already exceed the limit, the interval collapses to the conditional mean.
        """
        hits = np.flatnonzero(self.cont_idx == j)
        if hits.size == 0:
            return None
        k = int(hits[0])
        u = self.to_latent(x)
        Pkk = self.P[k, k]
        others = np.arange(len(u)) != k
        c = self.mu[k] - self.P[k, others] @ (u[others] - self.mu[others]) / Pkk
        u_c = u.copy()
        u_c[k] = c
        rest = float(_d2(u_c, self.mu, self.P)[0])
        half = np.sqrt(max(self.tau - rest, 0.0) / Pkk)
        m = self.marginals[k]
        return float(m.inverse(c - half)), float(m.inverse(c + half))

    # ------------------------------------------------------------- projections ----

    def project(self, x: np.ndarray) -> np.ndarray:
        """Pull ``x`` inside the region; identity if already inside.

        Radial shrink toward the latent mean = the Mahalanobis-closest boundary point (the
        ellipsoid is a sphere in whitened coordinates). Binary features are untouched.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.is_trivial:
            return x.copy()
        u = self.to_latent(x)
        d2 = float(_d2(u, self.mu, self.P)[0])
        if d2 <= self.tau:
            return x.copy()
        s = _PROJECT_MARGIN * np.sqrt(self.tau / d2)
        return self.from_latent(self.mu + s * (u - self.mu), x)

    def conditional_profile(self, j: int, value: float, base: np.ndarray) -> np.ndarray:
        """Realistic profile with feature ``j`` set to ``value``: the others at E[x₋ⱼ | xⱼ].

        Continuous ``j``: Gaussian conditional mean in latent space. Binary ``j``: mean of the
        training subjects in that group. The result is projected inside the region, which may
        also pull ``j`` itself back if ``value`` is marginally implausible.
        """
        x = np.asarray(base, dtype=np.float64).copy()
        hits = np.flatnonzero(self.cont_idx == j)
        if hits.size:
            k = int(hits[0])
            u_j = float(self.marginals[k].forward(value))
            u = self.mu + self.cov[:, k] / self.cov[k, k] * (u_j - self.mu[k])
            x = self.from_latent(u, x)
        else:
            grp = self.X_train[np.isclose(self.X_train[:, j], value)]
            if len(grp):
                x[self.cont_idx] = grp[:, self.cont_idx].mean(axis=0)
        x[j] = value
        return self.project(x)
