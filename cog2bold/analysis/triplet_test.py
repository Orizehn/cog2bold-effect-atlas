"""Triplet test: is the cognitively-closer subject also closer in map space?

For a triplet of subjects (A, B, C) chosen so that

    |x_A - x_B| <= a   (B is cognitively CLOSE to A)
    |x_A - x_C| >= b   (C is cognitively FAR from A)

count a "win" iff  d(A,B) < d(A,C)  in map space.  Under H0 (cognition independent of the maps)
B and C are exchangeable, so P(win) = 1/2 exactly.  H1: P(win) > 1/2.  Sweeping the separation
ratio b/a gives a dose-response curve, which is far more convincing than any single p-value.

TWO VALID VERSIONS (a naive binomial over all triplets is invalid -- the triplets share subjects,
and the error is ~150 orders of magnitude):

  1. permutation  -- use EVERY valid triplet, and get the null by shuffling the cognitive scores
                     across subjects and rebuilding the triplets.  Uses the data efficiently
                     (worth ~270-1300 independent trials here) but the p-value is floored at
                     1/(P+1).
  2. disjoint     -- keep triplets so that no subject is reused.  The trials are then genuinely
                     independent and the null is EXACTLY Binomial(N, 1/2), so the p-value is
                     closed-form with no floor -- but only ~n/3 triplets survive, so it is
                     under-powered (power ~0.3 at a true rate of 0.60).

Both are reported; see docs/stat_tests_report for the full analysis.

The accuracy over all triplets is computed EXACTLY (no sampling, no cap) in O(n^2 log n) via
sorting + searchsorted, which is what makes the permutation null affordable (~1.3 ms per pass).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest


# =====================================================================================
# core statistics
# =====================================================================================


def exact_triplet_wins(z: np.ndarray, D: np.ndarray, a: float, b: float) -> Tuple[int, int]:
    """(wins, total) over EVERY valid triplet -- exact, no sampling.

    z : (n,) standardised cognitive scores.  D : (n, n) map distances.
    For each anchor A let u = D[A, near(A)] and v = D[A, far(A)]; the number of winning triplets
    anchored at A is #{(i,j) : u_i < v_j}, obtained by sorting v once and binary-searching u.
    """
    n = z.shape[0]
    dz = np.abs(z[:, None] - z[None, :])
    near = dz <= a
    np.fill_diagonal(near, False)          # A cannot pair with itself
    far = dz >= b
    wins = total = 0
    for A in range(n):
        u = D[A, near[A]]
        v = D[A, far[A]]
        if u.size == 0 or v.size == 0:
            continue
        vs = np.sort(v)
        # strict inequality d(A,B) < d(A,C): count v_j strictly greater than each u_i
        wins += int((v.size - np.searchsorted(vs, u, side="right")).sum())
        total += u.size * v.size
    return wins, total


def disjoint_triplet_wins(
    z: np.ndarray, D: np.ndarray, a: float, b: float, seed: int = 0,
) -> Tuple[int, int]:
    """(wins, N) over a set of triplets in which NO SUBJECT IS REUSED.

    Greedy: visit anchors in a random (seeded) order and, if the anchor is still unused, claim one
    unused near partner and one unused far partner. The resulting trials are independent, so the
    null is exactly Binomial(N, 1/2). At most floor(n/3) triplets can exist.
    """
    n = z.shape[0]
    dz = np.abs(z[:, None] - z[None, :])
    near = dz <= a
    np.fill_diagonal(near, False)
    far = dz >= b
    rng = np.random.default_rng(seed)
    used = np.zeros(n, dtype=bool)
    wins = N = 0
    for A in rng.permutation(n):
        if used[A]:
            continue
        cand_b = np.flatnonzero(near[A] & ~used)
        cand_c = np.flatnonzero(far[A] & ~used)
        if cand_b.size == 0 or cand_c.size == 0:
            continue
        B = int(rng.choice(cand_b))
        cand_c = cand_c[cand_c != B]
        if cand_c.size == 0:
            continue
        C = int(rng.choice(cand_c))
        used[[A, B, C]] = True
        wins += int(D[A, B] < D[A, C])
        N += 1
    return wins, N


def triplet_permutation_test(
    z: np.ndarray, D: np.ndarray, a: float, b: float, n_perm: int = 1000, seed: int = 0,
) -> Dict[str, float]:
    """All-triplet accuracy with a permutation null (shuffle cognition, rebuild the triplets)."""
    w, t = exact_triplet_wins(z, D, a, b)
    if t == 0:
        return {"acc": float("nan"), "n_triplets": 0, "p_value": float("nan"),
                "null_mean": float("nan"), "null_sd": float("nan"), "z_score": float("nan"),
                "n_perm": 0, "n_eff": float("nan")}
    acc = w / t
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        zp = z[rng.permutation(n)]
        wk, tk = exact_triplet_wins(zp, D, a, b)
        null[k] = wk / tk if tk else np.nan
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"acc": acc, "n_triplets": t, "p_value": float("nan"),
                "null_mean": float("nan"), "null_sd": float("nan"),
                "z_score": float("nan"), "n_perm": 0, "n_eff": float("nan")}
    sd = float(null.std(ddof=1))
    return {
        "acc": float(acc),
        "n_triplets": int(t),
        "p_value": float((1 + int((null >= acc).sum())) / (1 + null.size)),
        "null_mean": float(null.mean()),
        "null_sd": sd,
        # standardised effect: how many null SDs above chance (readable below the p-value floor)
        "z_score": float((acc - null.mean()) / sd) if sd > 0 else float("nan"),
        "n_perm": int(null.size),
        # independent-trial equivalent of the dependent triplets: SD = 0.5/sqrt(N_eff)
        "n_eff": float(0.25 / sd ** 2) if sd > 0 else float("nan"),
    }


def disjoint_binomial_test(
    z: np.ndarray, D: np.ndarray, a: float, b: float, seed: int = 0,
) -> Dict[str, float]:
    """Disjoint-triplet accuracy with the EXACT Binomial(N, 1/2) null (no permutations)."""
    w, N = disjoint_triplet_wins(z, D, a, b, seed=seed)
    if N < 3:
        return {"acc": float("nan"), "N": int(N), "wins": int(w), "p_value": float("nan")}
    return {
        "acc": w / N,
        "N": int(N),
        "wins": int(w),
        "p_value": float(binomtest(w, N, 0.5, alternative="greater").pvalue),
    }


# =====================================================================================
# orchestration
# =====================================================================================


@dataclass
class TripletTestArgs:
    adapter_name: Optional[str] = None
    data_root: Optional[Path] = None
    single_contrast: Optional[object] = None
    stim_encoding: str = "contrast"
    adapter_kwargs: Optional[dict] = None
    cog_base_cols_to_scan: Optional[List[str]] = None

    a_sd: float = 0.25                       # |x_A - x_B| <= a, in SD units
    ratios: List[float] = field(default_factory=lambda: [2.0, 4.0, 6.0, 8.0])   # b/a
    map_metrics: List[str] = field(default_factory=lambda: ["correlation"])
    n_perm: int = 1000
    collapse_to_subject: bool = True
    seed: int = 0
    output_dir: Optional[Path] = None


def run_triplet_tests(args: TripletTestArgs) -> Dict[str, object]:
    """Both triplet versions for every feature x contrast x b/a of one dataset/task/group."""
    import pandas as pd

    from cog2bold.analysis.feature_informativeness import _build_feature_override
    from cog2bold.analysis.independence_tests import map_distance_matrix
    from cog2bold.datasets import create_adapter, filter_samples_by_contrast
    from cog2bold.workspace import cache_root

    if not args.adapter_name:
        raise ValueError("adapter_name is required")
    adapter = create_adapter(args.adapter_name, data_root=Path(args.data_root),
                             cog_base_cols=[], **(args.adapter_kwargs or {}))
    samples = adapter.materialize(
        cog_base_cols=[], stim_encoding_mode=args.stim_encoding,
        cache_dir=cache_root() / getattr(adapter, "name", args.adapter_name),
    )
    if args.single_contrast is not None:
        samples = filter_samples_by_contrast(samples, args.single_contrast, adapter)

    by_contrast: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        key = s.contrast_name if s.is_contrast else adapter.global_labels[s.stim_id]
        by_contrast.setdefault(key, []).append(i)

    feats = list(args.cog_base_cols_to_scan or [])
    overrides = {}
    for f in feats:
        ov, _ = _build_feature_override(adapter, samples, f)
        if ov:
            overrides[f] = ov
    print(f"[triplet] {args.adapter_name}: {len(samples)} maps, "
          f"{len(overrides)}/{len(feats)} features resolved")

    rows: List[dict] = []
    for contrast, idx_all in sorted(by_contrast.items()):
        idx = list(idx_all)
        if args.collapse_to_subject:
            seen, keep = set(), []
            for i in idx:
                if samples[i].subject_id in seen:
                    continue
                seen.add(samples[i].subject_id)
                keep.append(i)
            idx = keep
        if len(idx) < 12:                      # need enough subjects to form triplets
            print(f"[triplet] {contrast}: only {len(idx)} subjects — skipping")
            continue
        M = np.asarray([samples[i].map for i in idx], dtype=np.float64)
        Ds = {m: map_distance_matrix(M, metric=m) for m in args.map_metrics}
        print(f"[triplet] {contrast}: n={len(idx)}")

        for feat, ov in overrides.items():
            pos = [j for j, i in enumerate(idx) if i in ov]
            if len(pos) < 12:
                continue
            y = np.asarray([float(ov[idx[j]][0]) for j in pos])
            if y.std() <= 0:
                continue
            z = (y - y.mean()) / y.std()
            sel = np.asarray(pos, dtype=int)
            for metric, Dfull in Ds.items():
                D = Dfull[np.ix_(sel, sel)]
                for ratio in args.ratios:
                    b = args.a_sd * float(ratio)
                    pm = triplet_permutation_test(z, D, args.a_sd, b,
                                                  n_perm=args.n_perm, seed=args.seed)
                    dj = disjoint_binomial_test(z, D, args.a_sd, b, seed=args.seed)
                    rows.append({
                        "contrast": contrast, "feature": feat, "map_metric": metric,
                        "b_over_a": float(ratio), "a_sd": args.a_sd, "n_subjects": len(pos),
                        "n_triplets_available": pm["n_triplets"],
                        "perm_acc": pm["acc"], "perm_p": pm["p_value"],
                        "perm_null_mean": pm["null_mean"], "perm_null_sd": pm["null_sd"],
                        "perm_z": pm["z_score"], "perm_n_perm": pm["n_perm"],
                        "perm_n_eff": pm["n_eff"],
                        "disj_N": dj["N"], "disj_wins": dj["wins"],
                        "disj_acc": dj["acc"], "disj_p": dj["p_value"],
                    })
    if not rows:
        return {"results": pd.DataFrame()}
    df = pd.DataFrame(rows)
    if args.output_dir is not None:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "triplet_tests.csv", index=False)
        print(f"[triplet] saved {out / 'triplet_tests.csv'} ({len(df)} rows)")
    return {"results": df}
