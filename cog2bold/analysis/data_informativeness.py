"""Data informativeness analysis (oracle leave-two-out test).

Tests whether GT activation maps are informative about cognitive features WITHOUT
any model in the loop. For each pair of "inner pairs" `(A, B)` and `(C, D)`:

    A & B are cognitively close (inner pair 1)
    C & D are cognitively close (inner pair 2)
    Map A=pred_1, B=GT_1, C=pred_2, D=GT_2.
    Correct if r(A,B) > r(A,D) AND r(C,D) > r(C,B).

Inner pairs are built greedily (smallest within-pair distance). The pairing_mode
controls how inner pairs are combined into super-pairs:
    'nearest'  : super-pairs with smallest cross-pair distance (hardest)
    'furthest' : super-pairs with largest cross-pair distance  (easiest / upper bound)
    'random'   : random super-pair assembly

Empirical null model
--------------------
With ``run_null=True`` and ``n_null_permutations>0``, the analysis is re-run
N times with the cog↔map correspondence shuffled. Each permutation gets its
own freshly-built inner pairs and super-pairs (since pairing depends on cog
distance), so the null answers: "could a random cog assignment still find
informative super-pairs?". The summary JSON gains a ``null_model`` block with
empirical p-values, null mean / 95% CI, and per-permutation distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from cog2bold.config import CorrelationConfig, RunConfig
from cog2bold.datasets import (
    ActivationMapSample,
    DatasetAdapter,
    create_adapter,
    filter_samples_by_contrast,
)
from cog2bold.io import save_json
from cog2bold.pipelines._eval_base import EVAL_METRICS, metric_is_better
from cog2bold.pipelines.lto import _safe_corr, _summarise_null
from cog2bold.utils.atlas_projector import AtlasProjector
from cog2bold.utils.cog_features import cog_stats_from_unique_pairs
from cog2bold.workspace import cache_root


@dataclass
class DataInformativenessArgs:
    run_dir: Path
    output_dir: Optional[Path] = None
    max_super_pairs_per_contrast: int = 100
    pairing_modes: List[str] = field(default_factory=lambda: ["nearest", "furthest", "random"])
    min_cog_distance: Optional[float] = None
    unique_samples: bool = False
    save_maps: bool = True
    correlation_configs: Optional[List[CorrelationConfig]] = None
    seed: int = 0

    # Empirical null model — see module docstring.
    run_null: bool = False
    n_null_permutations: int = 0
    null_seed: int = 1
    # "within_contrast" | "global" | "within_contrast_session"
    null_shuffle_scope: str = "within_contrast"
    store_null_distribution: bool = True


# ---------- helpers: cog/session lookup with optional override ----------


def _cog_of(
    samples: List[ActivationMapSample],
    idx: int,
    cog_override: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    if cog_override is not None and idx in cog_override:
        return cog_override[idx]
    return samples[idx].cog_features


# ---------- inner-pair construction ----------


def _pairwise_cog_distances(Z: np.ndarray, cog_std: np.ndarray) -> np.ndarray:
    """Full (m, m) standardised-L2 distance matrix between the rows of Z.

    Computed row-wise with `np.linalg.norm(..., axis=-1)` so the arithmetic is IDENTICAL to the
    per-pair `np.linalg.norm((a - b) / cog_std)` it replaces: same subtract -> divide -> norm
    order and the same norm implementation. That matters because integer-valued test scores
    produce many EXACT distance ties, and which tie wins decides the pairing.

    Cost: m numpy calls instead of ~m^2/4 Python-level ones (the dominant cost of a null
    permutation sweep, since the pairing is redone for every permutation).
    """
    m = Z.shape[0]
    D = np.empty((m, m), dtype=np.float64)
    for a in range(m):
        D[a] = np.linalg.norm((Z[a] - Z) / cog_std, axis=-1)
    return D


def _build_inner_pairs(
    indices: List[int],
    samples: List[ActivationMapSample],
    cog_std: np.ndarray,
    unique_samples: bool,
    cog_override: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Greedy nearest-neighbour pairing within a contrast group.

    Vectorised over the distance computation only; the anchor order, the
    first-strict-minimum tie-breaking and the returned distances are unchanged.
    """
    idx = list(indices)
    if len(idx) < 2:
        return [], []
    # NOTE: do NOT coerce the dtype here. Feature overrides are float32, so the original
    # per-pair `(a - b) / cog_std` subtracted in float32 before promoting to float64; casting
    # to float64 first changes the low bits and, with ~1.6M exact ties in this data, flips
    # tie-breaks and changes the pairing.
    Z = np.asarray([_cog_of(samples, i, cog_override) for i in idx])
    D = _pairwise_cog_distances(Z, cog_std)

    avail = list(range(len(idx)))          # positions into idx / D, in the original order
    pairs: List[Tuple[int, int]] = []
    dists: List[float] = []
    while len(avail) >= 2:
        a = avail.pop(0)
        cand = np.asarray(avail, dtype=np.intp)
        # argmin returns the FIRST minimum, i.e. the earliest candidate in `avail` order —
        # exactly what `if d < best_d` over the same scan order picked.
        k = int(np.argmin(D[a, cand]))
        b = avail.pop(k)
        pairs.append((idx[a], idx[b]))
        dists.append(float(D[a, b]))
    return pairs, dists


def _assemble_super_pairs(
    inner_pairs: List[Tuple[int, int]],
    inner_dists: List[float],
    samples: List[ActivationMapSample],
    cog_std: np.ndarray,
    mode: str,
    max_pairs: int,
    rng: np.random.Generator,
    cog_override: Optional[Dict[int, np.ndarray]] = None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if len(inner_pairs) < 2:
        return []
    centroids = np.asarray([
        0.5 * (_cog_of(samples, i, cog_override) + _cog_of(samples, j, cog_override))
        for (i, j) in inner_pairs
    ])
    # same values as the previous per-pair loop, computed in one vectorised sweep
    # (centroids passed through unchanged — see the dtype note in _build_inner_pairs)
    cross = _pairwise_cog_distances(centroids, cog_std)
    pairs_by_cross_dist: List[Tuple[float, int, int]] = []
    for a in range(len(inner_pairs)):
        for b in range(a + 1, len(inner_pairs)):
            pairs_by_cross_dist.append((cross[a, b], a, b))
    if mode == "nearest":
        pairs_by_cross_dist.sort(key=lambda x: x[0])
    elif mode == "furthest":
        pairs_by_cross_dist.sort(key=lambda x: -x[0])
    elif mode == "random":
        rng.shuffle(pairs_by_cross_dist)
    else:
        raise ValueError(f"Unknown pairing mode {mode!r}")
    return [(inner_pairs[a], inner_pairs[b]) for (_, a, b) in pairs_by_cross_dist[:max_pairs]]


# ---------- scoring helper (factored so null doesn't rebuild projectors) ----------


def _make_score_fn(
    corr_configs: List[CorrelationConfig], proj_by_n: Dict[int, AtlasProjector],
) -> Callable[[np.ndarray, np.ndarray, CorrelationConfig], Dict[str, float]]:
    """Return score(m1, m2, cfg) -> {"correlation": r, "mse": mse}, projecting once."""

    def score(m1: np.ndarray, m2: np.ndarray, cfg: CorrelationConfig) -> Dict[str, float]:
        if cfg.space == 0:
            v1, v2 = m1, m2
        else:
            proj = proj_by_n[cfg.atlas_n_rois]
            v1 = proj.voxel_to_roi(m1)
            v2 = proj.voxel_to_roi(m2)
            if cfg.space == 2 and cfg.roi_name_patterns:
                idx = proj.roi_indices_matching(cfg.roi_name_patterns)
                if idx:
                    v1, v2 = v1[idx], v2[idx]
        a = np.asarray(v1).ravel().astype(np.float64)
        b = np.asarray(v2).ravel().astype(np.float64)
        return {"correlation": _safe_corr(a, b), "mse": float(np.mean((a - b) ** 2))}

    return score


# ---------- one full evaluation pass (observed or null permutation) ----------


def _evaluate_one_pass(
    samples: List[ActivationMapSample],
    groups: Dict[str, List[int]],
    cog_std: np.ndarray,
    corr_configs: List[CorrelationConfig],
    score: Callable[[np.ndarray, np.ndarray, CorrelationConfig], Dict[str, float]],
    pairing_modes: List[str],
    max_super_pairs_per_contrast: int,
    unique_samples: bool,
    seed: int,
    cog_override: Optional[Dict[int, np.ndarray]] = None,
    collect_rows: bool = False,
    save_dir: Optional[Path] = None,
    starting_sp_id: int = 0,
    show_progress: bool = True,
) -> Tuple[Dict[str, Dict], List[dict], int]:
    """Run all `pairing_modes` × all groups under one (possibly shuffled) cog mapping.

    Both similarity metrics (correlation + MSE) are scored per super-pair in a
    single sweep. Returns:
        results_by_mode  — per-mode summary; carries a legacy correlation block at
                           the top level plus a `by_metric` dict (correlation, mse)
        rows             — per-super-pair detailed entries (only when collect_rows=True)
        next_sp_id       — running id, so successive passes don't collide
    """
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    sp_id = starting_sp_id
    results_by_mode: Dict[str, Dict] = {}
    primary = corr_configs[0].name

    for mode in pairing_modes:
        mode_total = 0
        mode_correct = {m: 0 for m in EVAL_METRICS}
        per_contrast: Dict[str, Dict] = {}
        # per_cfg_counts[metric][cfg] = {total, correct}
        per_cfg_counts = {
            m: {c.name: {"total": 0, "correct": 0} for c in corr_configs}
            for m in EVAL_METRICS
        }
        for cname, idx_list in groups.items():
            if len(idx_list) < 4:
                continue
            inner, dists = _build_inner_pairs(
                idx_list, samples, cog_std, unique_samples, cog_override=cog_override,
            )
            if len(inner) < 2:
                continue
            super_pairs = _assemble_super_pairs(
                inner, dists, samples, cog_std, mode,
                max_super_pairs_per_contrast, rng, cog_override=cog_override,
            )
            if show_progress:
                print(f"[{mode:>9}] {cname}: {len(super_pairs)} super-pairs")

            n_correct_c = {m: 0 for m in EVAL_METRICS}
            per_cfg_correct_c = {
                m: {c.name: 0 for c in corr_configs} for m in EVAL_METRICS
            }
            iterator = (tqdm(super_pairs, desc=f"  {mode}/{cname}") if show_progress
                        else super_pairs)
            for (P1, P2) in iterator:
                a, b = P1
                c, d = P2
                A, B = samples[a].map, samples[b].map
                C, D = samples[c].map, samples[d].map

                # per_cfg[name][metric] = {v11,v12,v21,v22, correct}
                per_cfg: Dict[str, Dict[str, dict]] = {}
                for cfg in corr_configs:
                    s11 = score(A, B, cfg); s12 = score(A, D, cfg)
                    s21 = score(C, B, cfg); s22 = score(C, D, cfg)
                    per_cfg[cfg.name] = {}
                    for m in EVAL_METRICS:
                        v11, v12, v21, v22 = s11[m], s12[m], s21[m], s22[m]
                        ok = metric_is_better(m, v11, v12) and metric_is_better(m, v22, v21)
                        per_cfg[cfg.name][m] = {
                            "v11": v11, "v12": v12, "v21": v21, "v22": v22, "correct": ok,
                        }
                        per_cfg_counts[m][cfg.name]["total"] += 1
                        if ok:
                            per_cfg_counts[m][cfg.name]["correct"] += 1
                            per_cfg_correct_c[m][cfg.name] += 1

                mode_total += 1
                for m in EVAL_METRICS:
                    if per_cfg[primary][m]["correct"]:
                        mode_correct[m] += 1
                        n_correct_c[m] += 1

                if collect_rows:
                    a_cog = _cog_of(samples, a, cog_override)
                    b_cog = _cog_of(samples, b, cog_override)
                    c_cog = _cog_of(samples, c, cog_override)
                    d_cog = _cog_of(samples, d, cog_override)
                    pc = per_cfg[primary]["correlation"]
                    pm = per_cfg[primary]["mse"]
                    row = {
                        "super_pair_id": sp_id,
                        "mode": mode,
                        "contrast": cname,
                        "subj_a": samples[a].subject_id, "ses_a": samples[a].session_id, "idx_a": a,
                        "subj_b": samples[b].subject_id, "ses_b": samples[b].session_id, "idx_b": b,
                        "subj_c": samples[c].subject_id, "ses_c": samples[c].session_id, "idx_c": c,
                        "subj_d": samples[d].subject_id, "ses_d": samples[d].session_id, "idx_d": d,
                        "cog_a": a_cog.tolist(),
                        "cog_b": b_cog.tolist(),
                        "cog_c": c_cog.tolist(),
                        "cog_d": d_cog.tolist(),
                        "cog_dist_pair1": float(np.linalg.norm((a_cog - b_cog) / cog_std)),
                        "cog_dist_pair2": float(np.linalg.norm((c_cog - d_cog) / cog_std)),
                        "cog_dist_cross": float(np.linalg.norm(
                            (0.5 * (a_cog + b_cog) - 0.5 * (c_cog + d_cog)) / cog_std
                        )),
                        # primary correlation (legacy columns)
                        "r11": pc["v11"], "r12": pc["v12"],
                        "r21": pc["v21"], "r22": pc["v22"],
                        "correct": pc["correct"],
                        # primary MSE
                        "mse11": pm["v11"], "mse12": pm["v12"],
                        "mse21": pm["v21"], "mse22": pm["v22"],
                        "correct_mse": pm["correct"],
                    }
                    for cfg_name, pc_metrics in per_cfg.items():
                        cc = pc_metrics["correlation"]
                        mm = pc_metrics["mse"]
                        for k in ("v11", "v12", "v21", "v22"):
                            rk = "r" + k[1:]  # v11 -> r11
                            row[f"{rk}_{cfg_name}"] = cc[k]
                            row[f"mse{k[1:]}_{cfg_name}"] = mm[k]
                        row[f"correct_{cfg_name}"] = cc["correct"]
                        row[f"correct_mse_{cfg_name}"] = mm["correct"]
                    rows.append(row)

                    if save_dir is not None:
                        sp_dir = save_dir / "maps" / mode / f"super_pair_{sp_id:05d}"
                        sp_dir.mkdir(parents=True, exist_ok=True)
                        np.save(sp_dir / "map_a.npy", A)
                        np.save(sp_dir / "map_b.npy", B)
                        np.save(sp_dir / "map_c.npy", C)
                        np.save(sp_dir / "map_d.npy", D)
                sp_id += 1

            n = len(super_pairs)
            per_contrast[cname] = {
                "n_super_pairs": int(n),
                # legacy correlation primary
                "n_correct": int(n_correct_c["correlation"]),
                "accuracy": (n_correct_c["correlation"] / n) if n else 0.0,
                "by_corr_config": {
                    cfg.name: {
                        "n_correct": int(per_cfg_correct_c["correlation"][cfg.name]),
                        "accuracy": (per_cfg_correct_c["correlation"][cfg.name] / n) if n else 0.0,
                    }
                    for cfg in corr_configs
                },
                "by_metric": {
                    m: {
                        "n_correct": int(n_correct_c[m]),
                        "accuracy": (n_correct_c[m] / n) if n else 0.0,
                        "by_corr_config": {
                            cfg.name: {
                                "n_correct": int(per_cfg_correct_c[m][cfg.name]),
                                "accuracy": (per_cfg_correct_c[m][cfg.name] / n) if n else 0.0,
                            }
                            for cfg in corr_configs
                        },
                    }
                    for m in EVAL_METRICS
                },
            }

        def _metric_summary(m: str) -> dict:
            return {
                "total": int(mode_total),
                "correct": int(mode_correct[m]),
                "accuracy": (mode_correct[m] / mode_total) if mode_total else 0.0,
                "by_corr_config": {
                    cfg.name: {
                        "correct": int(per_cfg_counts[m][cfg.name]["correct"]),
                        "total": int(per_cfg_counts[m][cfg.name]["total"]),
                        "accuracy": (
                            per_cfg_counts[m][cfg.name]["correct"]
                            / per_cfg_counts[m][cfg.name]["total"]
                        ) if per_cfg_counts[m][cfg.name]["total"] else 0.0,
                    }
                    for cfg in corr_configs
                },
                "per_contrast": {
                    cname: {
                        "n_super_pairs": stats["n_super_pairs"],
                        "n_correct": stats["by_metric"][m]["n_correct"],
                        "accuracy": stats["by_metric"][m]["accuracy"],
                        "by_corr_config": stats["by_metric"][m]["by_corr_config"],
                    }
                    for cname, stats in per_contrast.items()
                },
            }

        corr_summary = _metric_summary("correlation")
        results_by_mode[mode] = {
            # legacy correlation block at the top level (back-compat)
            "total": corr_summary["total"],
            "correct": corr_summary["correct"],
            "accuracy": corr_summary["accuracy"],
            "by_corr_config": corr_summary["by_corr_config"],
            "per_contrast": per_contrast,
            # both metrics, toggled in the viewer
            "by_metric": {m: _metric_summary(m) for m in EVAL_METRICS},
        }
    return results_by_mode, rows, sp_id


# ---------- null permutation builders ----------


def _make_cog_override_for_info(
    samples: List[ActivationMapSample],
    groups: Dict[str, List[int]],
    rng: np.random.Generator,
    scope: str,
) -> Dict[int, np.ndarray]:
    """Shuffle the cog↔map correspondence; scope mirrors the LTO null."""
    override: Dict[int, np.ndarray] = {}
    if scope == "global":
        all_idx = sorted({i for ids in groups.values() for i in ids})
        donors = np.asarray(all_idx, dtype=int).copy()
        rng.shuffle(donors)
        for tgt, src in zip(all_idx, donors):
            override[int(tgt)] = samples[int(src)].cog_features
        return override
    if scope == "within_contrast":
        for _, ids in groups.items():
            donors = np.asarray(ids, dtype=int).copy()
            rng.shuffle(donors)
            for tgt, src in zip(ids, donors):
                override[int(tgt)] = samples[int(src)].cog_features
        return override
    if scope == "within_contrast_session":
        # bucket by (contrast, session_id)
        for cname, ids in groups.items():
            buckets: Dict[int, List[int]] = {}
            for i in ids:
                buckets.setdefault(int(samples[i].session_id), []).append(i)
            for _, bucket in buckets.items():
                donors = np.asarray(bucket, dtype=int).copy()
                rng.shuffle(donors)
                for tgt, src in zip(bucket, donors):
                    override[int(tgt)] = samples[int(src)].cog_features
        return override
    raise ValueError(f"Unknown null_shuffle_scope {scope!r}")


# ---------- main entry ----------


def run_data_informativeness(args: DataInformativenessArgs) -> Dict[str, dict]:
    cfg_path = Path(args.run_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json in {args.run_dir}")
    run_config = RunConfig.load(cfg_path)

    adapter: DatasetAdapter = create_adapter(
        run_config.data_config.adapter_name,
        data_root=run_config.data_root,
        cog_base_cols=run_config.data_config.cog_base_cols,
        **(run_config.data_config.adapter_kwargs or {}),
    )
    cache_dir = cache_root() / run_config.data_config.adapter_name
    samples = adapter.materialize(
        cog_base_cols=run_config.data_config.cog_base_cols,
        stim_encoding_mode=run_config.data_config.stim_encoding,
        contrast_pairs=run_config.data_config.contrast_pairs,
        subject_session_filter=None,
        cache_dir=cache_dir,
    )
    if run_config.data_config.single_contrast is not None:
        samples = filter_samples_by_contrast(
            samples, run_config.data_config.single_contrast, adapter,
        )

    _, cog_std = cog_stats_from_unique_pairs(samples)

    corr_configs = args.correlation_configs
    if corr_configs is None and run_config.eval_correlation_configs:
        corr_configs = run_config.eval_correlation_configs
    if corr_configs is None:
        corr_configs = [CorrelationConfig(name="voxel", space=0)]

    masker = adapter.get_masker()
    proj_by_n: Dict[int, AtlasProjector] = {}
    for c in corr_configs:
        if c.space in {1, 2} and c.atlas_n_rois not in proj_by_n:
            proj_by_n[c.atlas_n_rois] = AtlasProjector(masker=masker, n_rois=c.atlas_n_rois)
    correlate = _make_score_fn(corr_configs, proj_by_n)

    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        key = s.contrast_name if s.is_contrast else adapter.global_labels[s.stim_id]
        groups.setdefault(key, []).append(i)

    if args.output_dir is None:
        out_dir = Path(args.run_dir) / f"data_informativeness_{datetime.now():%Y%m%d_%H%M%S}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- observed pass ----
    observed_by_mode, rows, _ = _evaluate_one_pass(
        samples=samples,
        groups=groups,
        cog_std=cog_std,
        corr_configs=corr_configs,
        score=correlate,
        pairing_modes=args.pairing_modes,
        max_super_pairs_per_contrast=args.max_super_pairs_per_contrast,
        unique_samples=args.unique_samples,
        seed=args.seed,
        cog_override=None,
        collect_rows=True,
        save_dir=out_dir if args.save_maps else None,
        show_progress=True,
    )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "data_informativeness_detailed.csv", index=False)

    # ---- null model ----
    null_top: Optional[dict] = None
    if args.run_null and args.n_null_permutations > 0:
        print(f"\n[info-null] running {args.n_null_permutations} permutations "
              f"(scope={args.null_shuffle_scope})")
        # null accuracy distributions per (mode, metric)
        null_overall = {
            mode: {m: [] for m in EVAL_METRICS} for mode in args.pairing_modes
        }
        null_by_cfg = {
            mode: {m: {c.name: [] for c in corr_configs} for m in EVAL_METRICS}
            for mode in args.pairing_modes
        }
        null_per_contrast = {
            mode: {m: {cname: [] for cname in groups.keys()} for m in EVAL_METRICS}
            for mode in args.pairing_modes
        }

        for perm_idx in range(args.n_null_permutations):
            rng_np = np.random.default_rng(args.null_seed + perm_idx)
            cog_override = _make_cog_override_for_info(
                samples, groups, rng_np, args.null_shuffle_scope,
            )
            perm_by_mode, _, _ = _evaluate_one_pass(
                samples=samples, groups=groups, cog_std=cog_std,
                corr_configs=corr_configs, score=correlate,
                pairing_modes=args.pairing_modes,
                max_super_pairs_per_contrast=args.max_super_pairs_per_contrast,
                unique_samples=args.unique_samples,
                seed=args.seed + perm_idx + 1,
                cog_override=cog_override,
                collect_rows=False, save_dir=None, show_progress=False,
            )
            for mode in args.pairing_modes:
                for m in EVAL_METRICS:
                    block = perm_by_mode[mode]["by_metric"][m]
                    null_overall[mode][m].append(block["accuracy"])
                    for cfg in corr_configs:
                        null_by_cfg[mode][m][cfg.name].append(
                            block["by_corr_config"][cfg.name]["accuracy"]
                        )
                    for cname in null_per_contrast[mode][m]:
                        if cname in block["per_contrast"]:
                            null_per_contrast[mode][m][cname].append(
                                block["per_contrast"][cname]["accuracy"]
                            )
            print(f"[info-null] perm {perm_idx + 1}/{args.n_null_permutations}: "
                  + ", ".join(
                      f"{mode}={perm_by_mode[mode]['accuracy']:.3f}"
                      for mode in args.pairing_modes
                  ))

        def _null_block_for(mode: str, m: str) -> dict:
            obs = observed_by_mode[mode]["by_metric"][m]
            block = {
                "overall": _summarise_null(obs["accuracy"], null_overall[mode][m]),
                "by_corr_config": {
                    cfg.name: _summarise_null(
                        obs["by_corr_config"][cfg.name]["accuracy"],
                        null_by_cfg[mode][m][cfg.name],
                    )
                    for cfg in corr_configs
                },
                "per_contrast": {
                    cname: _summarise_null(
                        obs["per_contrast"].get(cname, {}).get("accuracy", 0.0),
                        null_per_contrast[mode][m][cname],
                    )
                    for cname in null_per_contrast[mode][m]
                },
            }
            if args.store_null_distribution:
                block["distributions"] = {
                    "overall": null_overall[mode][m],
                    "by_corr_config": null_by_cfg[mode][m],
                    "per_contrast": null_per_contrast[mode][m],
                }
            return block

        # Attach per-mode null blocks: legacy top-level (correlation) + per metric.
        for mode in args.pairing_modes:
            observed_by_mode[mode]["null_model"] = _null_block_for(mode, "correlation")
            for m in EVAL_METRICS:
                observed_by_mode[mode]["by_metric"][m]["null_model"] = _null_block_for(mode, m)

        null_top = {
            "n_permutations": int(args.n_null_permutations),
            "shuffle_scope": args.null_shuffle_scope,
            "seed": int(args.null_seed),
            "store_distribution": bool(args.store_null_distribution),
        }

    summary = {
        "config": {
            "run_dir": str(args.run_dir),
            "max_super_pairs_per_contrast": args.max_super_pairs_per_contrast,
            "pairing_modes": args.pairing_modes,
            "min_cog_distance": args.min_cog_distance,
            "unique_samples": args.unique_samples,
            "single_contrast": run_config.data_config.single_contrast,
            "cog_base_cols": run_config.data_config.cog_base_cols,
            "map_source": run_config.data_config.adapter_name,
        },
        "chance_level": 0.25,
        "eval_metrics": list(EVAL_METRICS),
        "correlation_configs": [c.to_dict() for c in corr_configs],
        "results_by_mode": observed_by_mode,
        "null_model": null_top,
    }
    save_json(summary, out_dir / "data_informativeness_summary.json")
    print(f"\n[data_informativeness] saved -> {out_dir}")
    return summary


__all__ = ["DataInformativenessArgs", "run_data_informativeness"]
