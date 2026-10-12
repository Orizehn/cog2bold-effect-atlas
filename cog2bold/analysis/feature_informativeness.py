"""Per-feature data-informativeness scan.

The data-informativeness oracle test (cog2bold/analysis/data_informativeness.py)
tells us *whether* a set of cognitive features is informative about a task's
activation maps. This module answers *which individual feature* drives the
signal: it runs the same oracle test SEPARATELY for each single cognitive
feature (a one-element cog vector), each with its own empirical null model, and
emits a feature-ranking table (CSV + JSON).

Why it is cheap
---------------
The map-similarity score `score(m1, m2, cfg)` (data_informativeness._make_score_fn)
depends ONLY on the two maps + the CorrelationConfig — never on cog features. So
the pairwise score r(map_i, map_j, cfg) is identical across every feature run AND
every null permutation. We therefore:
  1. materialize the maps ONCE (with an empty cog list, so no subject is dropped
     for cog reasons), and
  2. memoize pairwise scores keyed by stable map identity + cfg,
reusing the cache across all features × all permutations. A 100-permutation null
over many features collapses from days to minutes.

Reuse
-----
The observed/null passes reuse `_evaluate_one_pass`, `_make_score_fn`, and the
scope branching of `_make_cog_override_for_info` from data_informativeness.py
verbatim (no edits there), plus `_summarise_null` from pipelines/lto.py.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cog2bold.analysis.data_informativeness import (
    _evaluate_one_pass,
    _make_score_fn,
)
from cog2bold.config import CorrelationConfig, RunConfig
from cog2bold.datasets import (
    ActivationMapSample,
    DatasetAdapter,
    create_adapter,
    filter_samples_by_contrast,
)
from cog2bold.io import save_json
from cog2bold.pipelines._eval_base import EVAL_METRICS
from cog2bold.pipelines.lto import _summarise_null
from cog2bold.utils.atlas_projector import AtlasProjector
from cog2bold.workspace import cache_root


MapId = Tuple[int, int, int, bool, Optional[str]]


@dataclass
class FeatureScanArgs:
    """Inputs for a per-feature data-informativeness scan.

    The scan needs no trained run_dir (the oracle has no model). `run_dir` is an
    optional convenience: when given, the dataset/feature/correlation settings
    default to that run's config.json.
    """

    adapter_name: Optional[str] = None
    data_root: Optional[Path] = None
    cog_base_cols_to_scan: Optional[List[str]] = None  # None -> resolved (see below)
    scan_all_numeric: bool = False
    single_contrast: Optional[object] = None
    stim_encoding: str = "contrast"
    contrast_pairs: Optional[List[Tuple[str, str]]] = None

    pairing_modes: List[str] = field(default_factory=lambda: ["nearest", "furthest", "random"])
    max_super_pairs_per_contrast: int = 100
    unique_samples: bool = False
    correlation_configs: Optional[List[CorrelationConfig]] = None  # None -> [voxel]

    n_null_permutations: int = 100
    null_seed: int = 1
    # "within_contrast" | "global" | "within_contrast_session"
    null_shuffle_scope: str = "within_contrast"
    store_null_distribution: bool = False

    seed: int = 0
    output_dir: Optional[Path] = None
    adapter_kwargs: Optional[dict] = None
    run_dir: Optional[Path] = None

    # Persistent (on-disk) correlation cache: reuse computed map-pair scores
    # across separate process invocations (different features/parameters).
    persist_score_cache: bool = True
    force_recompute_scores: bool = False


# ---------- map identity + score cache ----------


def _map_id(s: ActivationMapSample) -> MapId:
    """Stable identity of the map a sample carries (independent of cog)."""
    return (s.subject_id, s.session_id, s.stim_id, s.is_contrast, s.contrast_name)


def _cfg_sig(cfg: CorrelationConfig) -> str:
    """Signature capturing everything that determines a cfg's score value.

    Uses the computation-determining fields (space, atlas size, ROI subset), NOT
    the human-chosen `name`, so the cache stays correct if a run reuses a name
    with different settings — and reusable across runs that request different
    cfg subsets.
    """
    return f"s{cfg.space}_n{cfg.atlas_n_rois}_r{'|'.join(cfg.roi_name_patterns or [])}"


def make_cached_score(
    base_score: Callable[[np.ndarray, np.ndarray, CorrelationConfig], Dict[str, float]],
    id_by_obj: Dict[int, MapId],
    cache: Dict[Tuple[Tuple[MapId, MapId], str], Dict[str, float]],
) -> Callable[[np.ndarray, np.ndarray, CorrelationConfig], Dict[str, float]]:
    """Wrap `base_score` so identical map-pairs are scored once and reused.

    The two maps are recovered to their stable ids via `id(map_array)` — safe
    because the base sample list is materialized once and its `.map` ndarrays are
    never copied (only cog vectors are overridden per feature/permutation). The
    key is (sorted map-id pair, cfg-signature), so it is stable across runs and
    can be persisted to / loaded from disk.
    """

    def score(m1: np.ndarray, m2: np.ndarray, cfg: CorrelationConfig) -> Dict[str, float]:
        ka = id_by_obj[id(m1)]
        kb = id_by_obj[id(m2)]
        pair = (ka, kb) if ka <= kb else (kb, ka)
        key = (pair, _cfg_sig(cfg))
        hit = cache.get(key)
        if hit is None:
            hit = base_score(m1, m2, cfg)
            cache[key] = hit
        return hit

    return score


# ---------- persistent score cache ----------


def _score_cache_fingerprint(
    adapter: DatasetAdapter,
    base_samples: List[ActivationMapSample],
    stim_encoding: str,
    single_contrast: object,
) -> str:
    """Fingerprint tying a cache file to the maps it was computed from.

    Includes the mask file identity (path + mtime), the encoding/contrast subset,
    and a hash of all map identities. If any of these change (e.g. the mask or the
    materialised sample set), the fingerprint changes and the stale cache is
    ignored. NOTE: regenerating z-maps in place (same identities, new content) is
    not detected here — use force_recompute_scores in that case (same caveat as
    the materialize .pkl cache).
    """
    try:
        mp = Path(adapter.mask_path())
        mask_id = f"{mp}:{mp.stat().st_mtime_ns}"
    except Exception:
        mask_id = "no-mask"
    ids = sorted(_map_id(s) for s in base_samples)
    ids_hash = hashlib.md5(json.dumps(ids, default=str).encode()).hexdigest()
    payload = {
        "adapter": adapter.name,
        "mask": mask_id,
        "stim_encoding": stim_encoding,
        "single_contrast": str(single_contrast),
        "n_samples": len(base_samples),
        "ids_hash": ids_hash,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _score_cache_path(adapter_name: str) -> Path:
    d = cache_root() / "feature_informativeness" / "score_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{adapter_name}.pkl"


def _load_score_cache(path: Path, fingerprint: str) -> dict:
    """Load a persisted score cache iff its fingerprint matches; else empty."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("fingerprint") == fingerprint:
            return blob.get("scores", {})
        print(f"[feature-scan] score cache {path.name} stale (fingerprint changed) — ignoring")
    except Exception as e:
        print(f"[feature-scan] could not read score cache {path.name}: {e}")
    return {}


def _save_score_cache(path: Path, fingerprint: str, scores: dict) -> None:
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"fingerprint": fingerprint, "scores": scores}, f)
    tmp.replace(path)


# ---------- per-feature cog injection ----------


def _build_feature_override(
    adapter: DatasetAdapter,
    base_samples: List[ActivationMapSample],
    feat: str,
) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """Return (cog_override idx->[value], kept_indices) for one feature.

    Temporarily points the adapter at the single column and reuses the adapter's
    own missing/NaN/non-numeric validation; samples whose value is unusable are
    dropped from `kept_indices` for this feature only.
    """
    override: Dict[int, np.ndarray] = {}
    kept: List[int] = []
    saved = adapter.cog_base_cols
    try:
        adapter.cog_base_cols = [feat]
        for i, s in enumerate(base_samples):
            try:
                v = adapter.get_cog_vector(s.subject_id, s.session_id)
            except (KeyError, ValueError):
                continue
            # We set exactly one base column, so v[0] is that feature's value.
            # (Some adapters, e.g. ERA812, append a group one-hot after the base
            # cols — ignore it; the scan is about the single raw feature.)
            if v.shape[0] < 1 or not np.isfinite(v[0]):
                continue
            override[i] = np.asarray([v[0]], dtype=np.float32)
            kept.append(i)
    finally:
        adapter.cog_base_cols = saved
    return override, kept


def _feature_cog_std(
    override: Dict[int, np.ndarray],
    base_samples: List[ActivationMapSample],
) -> np.ndarray:
    """Std of the single feature over unique (subject, session); floored at 1e-8.

    Mirrors cog_features.cog_stats_from_unique_pairs but reads the override values
    rather than each sample's (empty) cog_features.
    """
    seen: Dict[Tuple[int, int], float] = {}
    for i, vec in override.items():
        s = base_samples[i]
        seen.setdefault((s.subject_id, s.session_id), float(vec[0]))
    vals = np.asarray(list(seen.values()), dtype=np.float64)
    return np.maximum(np.asarray([vals.std()]), 1e-8)


def _shuffle_feature_override(
    feature_override: Dict[int, np.ndarray],
    feat_groups: Dict[str, List[int]],
    rng: np.random.Generator,
    scope: str,
    base_samples: List[ActivationMapSample],
) -> Dict[int, np.ndarray]:
    """Null shuffle of the cog<->map correspondence, pulling values from
    `feature_override`. Scope branching mirrors
    data_informativeness._make_cog_override_for_info.
    """
    out: Dict[int, np.ndarray] = {}
    if scope == "global":
        all_idx = sorted({i for ids in feat_groups.values() for i in ids})
        donors = np.asarray(all_idx, dtype=int).copy()
        rng.shuffle(donors)
        for tgt, src in zip(all_idx, donors):
            out[int(tgt)] = feature_override[int(src)]
        return out
    if scope == "within_contrast":
        for _, ids in feat_groups.items():
            donors = np.asarray(ids, dtype=int).copy()
            rng.shuffle(donors)
            for tgt, src in zip(ids, donors):
                out[int(tgt)] = feature_override[int(src)]
        return out
    if scope == "within_contrast_session":
        for _, ids in feat_groups.items():
            buckets: Dict[int, List[int]] = {}
            for i in ids:
                buckets.setdefault(int(base_samples[i].session_id), []).append(i)
            for _, bucket in buckets.items():
                donors = np.asarray(bucket, dtype=int).copy()
                rng.shuffle(donors)
                for tgt, src in zip(bucket, donors):
                    out[int(tgt)] = feature_override[int(src)]
        return out
    raise ValueError(f"Unknown null_shuffle_scope {scope!r}")


# ---------- feature-list resolution ----------


def _enumerate_numeric_cog_cols(adapter: DatasetAdapter) -> List[str]:
    """Best-effort enumeration of numeric cog columns from the adapter's table.

    Works for adapters that expose a pandas frame as `_df` (AOMIC/ERA812) or
    `_df_behav` (DFG); otherwise falls back to default_cog_base_cols().

    A column counts as numeric if it either has a numeric dtype OR coerces
    (`pd.to_numeric`) to a majority of finite values — so a numeric battery
    column that pandas typed as ``object`` only because of a few stray blank
    cells (e.g. ERA812's reading scores) is still picked up. Genuinely
    categorical text columns coerce entirely to NaN and are dropped.
    """
    df = getattr(adapter, "_df", None)
    if df is None:
        df = getattr(adapter, "_df_behav", None)
    if df is None:
        return list(adapter.default_cog_base_cols())
    drop = {"Subject", "subject", "subject_id", "participant_id", "session",
            "session_id", "Group_New", "group", "ses"}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        finite = coerced.notna().sum()
        if finite >= 2 and finite >= 0.5 * len(coerced):
            cols.append(c)
    return cols


def _resolve_features(adapter: DatasetAdapter, args: FeatureScanArgs) -> List[str]:
    if args.cog_base_cols_to_scan:
        return list(args.cog_base_cols_to_scan)
    if args.scan_all_numeric:
        return _enumerate_numeric_cog_cols(adapter)
    feats = list(adapter.default_cog_base_cols())
    if feats:
        return feats
    raise ValueError(
        "No features to scan: pass cog_base_cols_to_scan, set scan_all_numeric=True, "
        "or use an adapter with default_cog_base_cols()."
    )


# ---------- null aggregation per feature ----------


def _null_blocks(
    observed_by_mode: Dict[str, dict],
    null_overall: Dict[str, Dict[str, List[float]]],
    null_by_cfg: Dict[str, Dict[str, Dict[str, List[float]]]],
    null_per_contrast: Dict[str, Dict[str, Dict[str, List[float]]]],
    corr_configs: List[CorrelationConfig],
    pairing_modes: List[str],
    store_distribution: bool,
) -> Dict[str, dict]:
    """Build per-mode null_model blocks (one per metric) from accumulated dists."""
    out: Dict[str, dict] = {}
    for mode in pairing_modes:
        out[mode] = {"by_metric": {}}
        for m in EVAL_METRICS:
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
                        dist,
                    )
                    for cname, dist in null_per_contrast[mode][m].items()
                },
            }
            if store_distribution:
                block["distributions"] = {
                    "overall": null_overall[mode][m],
                    "by_corr_config": null_by_cfg[mode][m],
                    "per_contrast": null_per_contrast[mode][m],
                }
            out[mode]["by_metric"][m] = block
    return out


# ---------- main entry ----------


def run_feature_informativeness_scan(args: FeatureScanArgs) -> Dict[str, dict]:
    # Borrow defaults from a trained run's config.json, if provided.
    adapter_name = args.adapter_name
    data_root = args.data_root
    cog_cols_scan = args.cog_base_cols_to_scan
    corr_configs = args.correlation_configs
    stim_encoding = args.stim_encoding
    contrast_pairs = args.contrast_pairs
    single_contrast = args.single_contrast
    adapter_kwargs = dict(args.adapter_kwargs or {})

    if args.run_dir is not None:
        rc = RunConfig.load(Path(args.run_dir) / "config.json")
        adapter_name = adapter_name or rc.data_config.adapter_name
        data_root = data_root or rc.data_root
        stim_encoding = stim_encoding or rc.data_config.stim_encoding
        contrast_pairs = contrast_pairs or rc.data_config.contrast_pairs
        if single_contrast is None:
            single_contrast = rc.data_config.single_contrast
        if corr_configs is None and rc.eval_correlation_configs:
            corr_configs = rc.eval_correlation_configs
        if not adapter_kwargs and rc.data_config.adapter_kwargs:
            adapter_kwargs = dict(rc.data_config.adapter_kwargs)
        if cog_cols_scan is None and not args.scan_all_numeric:
            cog_cols_scan = rc.data_config.cog_base_cols

    if adapter_name is None:
        raise ValueError("adapter_name is required (or pass run_dir).")
    if data_root is None:
        # Fall back to the adapter class's default data root (per-dataset location).
        from cog2bold.datasets.registry import _REGISTRY
        cls = _REGISTRY.get(adapter_name)
        data_root = cls.default_data_root() if cls is not None else None
        if data_root is None:
            raise ValueError(
                f"data_root not given and adapter {adapter_name!r} has no default_data_root().")
    if corr_configs is None:
        corr_configs = [CorrelationConfig(name="voxel", space=0)]

    # --- adapter + materialize ONCE (empty cog list keeps all maps) ---
    adapter: DatasetAdapter = create_adapter(
        adapter_name, data_root=Path(data_root), cog_base_cols=[], **adapter_kwargs,
    )
    cache_dir = cache_root() / adapter_name
    base_samples = adapter.materialize(
        cog_base_cols=[],
        stim_encoding_mode=stim_encoding,
        contrast_pairs=contrast_pairs,
        subject_session_filter=None,
        cache_dir=cache_dir,
    )
    if single_contrast is not None:
        base_samples = filter_samples_by_contrast(base_samples, single_contrast, adapter)
    print(f"[feature-scan] {adapter_name}: {len(base_samples)} base maps materialized")

    # --- groups (same key as the oracle) ---
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(base_samples):
        key = s.contrast_name if s.is_contrast else adapter.global_labels[s.stim_id]
        groups.setdefault(key, []).append(i)

    # --- projectors + cached scorer (built once, shared across features) ---
    proj_by_n: Dict[int, AtlasProjector] = {}
    if any(c.space in {1, 2} for c in corr_configs):
        masker = adapter.get_masker()
        for c in corr_configs:
            if c.space in {1, 2} and c.atlas_n_rois not in proj_by_n:
                proj_by_n[c.atlas_n_rois] = AtlasProjector(masker=masker, n_rois=c.atlas_n_rois)
    base_score = _make_score_fn(corr_configs, proj_by_n)
    id_by_obj = {id(s.map): _map_id(s) for s in base_samples}

    # Correlation cache: reused across features + null permutations, and — when
    # persist_score_cache is set — loaded from / saved to disk so re-runs with
    # different features/parameters skip already-computed map-pair scores.
    cache_fp = _score_cache_fingerprint(adapter, base_samples, stim_encoding, single_contrast)
    cache_path = _score_cache_path(adapter_name)
    if args.persist_score_cache and not args.force_recompute_scores:
        score_cache = _load_score_cache(cache_path, cache_fp)
        if score_cache:
            print(f"[feature-scan] loaded {len(score_cache)} cached map-pair scores "
                  f"from {cache_path.name}")
    else:
        score_cache = {}
    n_cached_at_start = len(score_cache)
    cached_score = make_cached_score(base_score, id_by_obj, score_cache)

    features = _resolve_features(adapter, args)
    print(f"[feature-scan] scanning {len(features)} features: {features}")

    # ranking axis: furthest mode, correlation metric, first voxel cfg (else first)
    rank_cfg = next((c for c in corr_configs if c.space == 0), corr_configs[0])
    rank_mode = "furthest" if "furthest" in args.pairing_modes else args.pairing_modes[0]

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        from cog2bold.workspace import split_adapter
        _ds, _task = split_adapter(adapter_name)
        out_dir = (cache_root() / "feature_informativeness" / _ds / _task
                   / f"{adapter_name}_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    features_summary: Dict[str, dict] = {}
    detailed_rows: List[dict] = []
    ranking_rows: List[dict] = []

    for fi, feat in enumerate(features, start=1):
        override, kept = _build_feature_override(adapter, base_samples, feat)
        kept_set = set(kept)
        feat_groups = {k: [i for i in idxs if i in kept_set] for k, idxs in groups.items()}
        feat_groups = {k: v for k, v in feat_groups.items() if len(v) >= 4}
        kept_n = len(kept)
        if not feat_groups:
            print(f"[feature-scan] {feat}: kept_n={kept_n} — no group with >=4 samples, skipping")
            continue
        cog_std = _feature_cog_std(override, base_samples)
        print(f"[feature-scan] [{fi}/{len(features)}] {feat}: kept_n={kept_n}, "
              f"groups={list(feat_groups)}")

        # observed pass (times the cold-cache map-pair scoring; nulls reuse the cache)
        print(f"[feature-scan]   observed pass (cache: {len(score_cache)} scores) ...",
              flush=True)
        t0 = time.perf_counter()
        observed_by_mode, _, _ = _evaluate_one_pass(
            samples=base_samples, groups=feat_groups, cog_std=cog_std,
            corr_configs=corr_configs, score=cached_score,
            pairing_modes=args.pairing_modes,
            max_super_pairs_per_contrast=args.max_super_pairs_per_contrast,
            unique_samples=args.unique_samples, seed=args.seed,
            cog_override=override, collect_rows=False, save_dir=None,
            show_progress=True,
        )
        print(f"[feature-scan]   observed pass done in {time.perf_counter() - t0:.1f}s "
              f"(cache: {len(score_cache)} scores); running {args.n_null_permutations} "
              f"null permutations ...", flush=True)

        # null distributions
        null_overall = {mode: {m: [] for m in EVAL_METRICS} for mode in args.pairing_modes}
        null_by_cfg = {
            mode: {m: {c.name: [] for c in corr_configs} for m in EVAL_METRICS}
            for mode in args.pairing_modes
        }
        null_per_contrast = {
            mode: {m: {cname: [] for cname in feat_groups} for m in EVAL_METRICS}
            for mode in args.pairing_modes
        }
        for perm in range(args.n_null_permutations):
            if perm and perm % 25 == 0:
                print(f"[feature-scan]   null perm {perm}/{args.n_null_permutations}",
                      flush=True)
            rng = np.random.default_rng(args.null_seed + perm)
            shuffled = _shuffle_feature_override(
                override, feat_groups, rng, args.null_shuffle_scope, base_samples,
            )
            perm_by_mode, _, _ = _evaluate_one_pass(
                samples=base_samples, groups=feat_groups, cog_std=cog_std,
                corr_configs=corr_configs, score=cached_score,
                pairing_modes=args.pairing_modes,
                max_super_pairs_per_contrast=args.max_super_pairs_per_contrast,
                unique_samples=args.unique_samples, seed=args.seed + perm + 1,
                cog_override=shuffled, collect_rows=False, save_dir=None,
                show_progress=False,
            )
            for mode in args.pairing_modes:
                for m in EVAL_METRICS:
                    blk = perm_by_mode[mode]["by_metric"][m]
                    null_overall[mode][m].append(blk["accuracy"])
                    for cfg in corr_configs:
                        null_by_cfg[mode][m][cfg.name].append(
                            blk["by_corr_config"][cfg.name]["accuracy"])
                    for cname in null_per_contrast[mode][m]:
                        if cname in blk["per_contrast"]:
                            null_per_contrast[mode][m][cname].append(
                                blk["per_contrast"][cname]["accuracy"])

        null_by_mode = _null_blocks(
            observed_by_mode, null_overall, null_by_cfg, null_per_contrast,
            corr_configs, args.pairing_modes, args.store_null_distribution,
        )
        features_summary[feat] = {
            "kept_n": kept_n,
            "results_by_mode": observed_by_mode,
            "null_model_by_mode": null_by_mode,
        }

        # detailed rows (every mode x metric x cfg x level) + ranking row
        for mode in args.pairing_modes:
            for m in EVAL_METRICS:
                obs = observed_by_mode[mode]["by_metric"][m]
                nm = null_by_mode[mode]["by_metric"][m]
                for cfg in corr_configs:
                    o_acc = obs["by_corr_config"][cfg.name]["accuracy"]
                    nblk = nm["by_corr_config"][cfg.name]
                    detailed_rows.append(_row(
                        feat, kept_n, mode, m, cfg.name, "overall", "",
                        o_acc, nblk, obs["total"]))
                    for cname, stats in obs["per_contrast"].items():
                        c_acc = stats["by_corr_config"][cfg.name]["accuracy"]
                        c_nblk = nm["per_contrast"][cname]
                        detailed_rows.append(_row(
                            feat, kept_n, mode, m, cfg.name, "contrast", cname,
                            c_acc, c_nblk, stats["n_super_pairs"]))

        ro = observed_by_mode[rank_mode]["by_metric"]["correlation"]
        r_acc = ro["by_corr_config"][rank_cfg.name]["accuracy"]
        r_null = null_by_mode[rank_mode]["by_metric"]["correlation"]["by_corr_config"][rank_cfg.name]
        ranking_rows.append({
            "feature": feat, "kept_n": kept_n,
            "observed_accuracy": r_acc,
            "null_mean": r_null["mean"], "null_std": r_null["std"],
            "ci95_low": r_null["ci95_low"], "ci95_high": r_null["ci95_high"],
            "empirical_p_one_sided": r_null["empirical_p_one_sided"],
            "delta_over_null_mean": r_null["delta_over_null_mean"],
            "z": r_null["z_observed_vs_null"],
        })

    # --- write outputs ---
    if args.persist_score_cache:
        _save_score_cache(cache_path, cache_fp, score_cache)
        print(f"[feature-scan] score cache: {n_cached_at_start} reused + "
              f"{len(score_cache) - n_cached_at_start} new = {len(score_cache)} total "
              f"-> {cache_path.name}")
    ranking_rows.sort(
        key=lambda r: (r["delta_over_null_mean"] if r["delta_over_null_mean"] is not None else -1e9),
        reverse=True,
    )
    pd.DataFrame(ranking_rows).to_csv(out_dir / "feature_informativeness_ranking.csv", index=False)
    pd.DataFrame(detailed_rows).to_csv(out_dir / "feature_informativeness_detailed.csv", index=False)

    summary = {
        "config": {
            "adapter_name": adapter_name,
            "data_root": str(data_root),
            "adapter_kwargs": adapter_kwargs,
            "single_contrast": single_contrast,
            "pairing_modes": args.pairing_modes,
            "max_super_pairs_per_contrast": args.max_super_pairs_per_contrast,
            "correlation_configs": [c.to_dict() for c in corr_configs],
            "n_null_permutations": args.n_null_permutations,
            "null_shuffle_scope": args.null_shuffle_scope,
            "null_seed": args.null_seed,
            "seed": args.seed,
            "features_scanned": features,
            "ranking_axis": {"mode": rank_mode, "metric": "correlation",
                             "corr_config": rank_cfg.name, "level": "overall"},
        },
        "chance_level": 0.25,
        "eval_metrics": list(EVAL_METRICS),
        "features": features_summary,
        "ranking": ranking_rows,
    }
    save_json(summary, out_dir / "feature_informativeness_summary.json")
    print(f"\n[feature-scan] saved -> {out_dir}")
    return summary


def _row(feat, kept_n, mode, metric, cfg_name, level, contrast,
         observed_accuracy, null_block, n_super_pairs) -> dict:
    return {
        "feature": feat, "kept_n": kept_n, "mode": mode, "metric": metric,
        "corr_config": cfg_name, "level": level, "contrast": contrast,
        "observed_accuracy": observed_accuracy,
        "null_mean": null_block["mean"], "null_std": null_block["std"],
        "ci95_low": null_block["ci95_low"], "ci95_high": null_block["ci95_high"],
        "empirical_p_one_sided": null_block["empirical_p_one_sided"],
        "delta_over_null_mean": null_block["delta_over_null_mean"],
        "z": null_block["z_observed_vs_null"],
        "n_permutations": null_block["n_permutations"],
        "n_super_pairs": n_super_pairs,
    }


__all__ = ["FeatureScanArgs", "run_feature_informativeness_scan"]
