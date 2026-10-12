"""Orchestration: model → combinations → Δ maps → statistic → blobs → labelled peak table.

``run_scan`` is the single entry point used by the CLI, the report and the Streamlit app.
``threshold_effect`` re-thresholds one stored Δ without recomputing anything, which is what
the UI's live threshold controls call.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from cog2bold.analysis.effect_atlas import atlas_labels as al
from cog2bold.analysis.effect_atlas.clusters import ClusterParams, find_clusters
from cog2bold.analysis.effect_atlas.combinations import Combination, build_combinations, stack
from cog2bold.analysis.effect_atlas.model_io import Condition, LoadedRun
from cog2bold.analysis.effect_atlas.thresholding import (
    BootstrapStats,
    bootstrap,
    fdr_z_threshold,
    pooled_percentile_threshold,
    z_from_bootstrap,
)


@dataclass
class ScanParams:
    combos: Tuple[str, ...] = ("single", "pairwise", "archetype")
    features: Optional[Tuple[str, ...]] = None      # restrict to these features (None = all)
    conditions: Optional[Tuple[str, ...]] = None    # restrict to these conditions (None = all)
    k_sd: float = 1.0
    archetypes_k: Optional[int] = None
    mode: str = "effect"                            # effect | bootstrap
    # effect mode
    percentile: float = 99.0
    abs_threshold: Optional[float] = None           # overrides percentile
    # bootstrap mode
    n_boot: int = 200
    z_thr: float = 3.1
    fdr_q: Optional[float] = None                   # overrides z_thr
    # clusters
    min_cluster_voxels: int = 10
    connectivity: int = 26
    min_peak_distance_mm: float = 8.0
    max_subpeaks: int = 2
    seed: Optional[int] = None                      # None = the run's random_seed

    def __post_init__(self):
        if self.mode not in ("effect", "bootstrap"):
            raise ValueError(f"mode must be 'effect' or 'bootstrap' (got {self.mode!r})")

    @property
    def cluster_params(self) -> ClusterParams:
        return ClusterParams(self.connectivity, self.min_cluster_voxels,
                             self.min_peak_distance_mm, self.max_subpeaks)


@dataclass
class EffectMap:
    condition: str
    combo: Combination
    delta: np.ndarray                     # (n_voxels,) Δ in map units
    stat: np.ndarray                      # Δ (effect mode) or z (bootstrap mode)
    threshold: float
    peaks: pd.DataFrame
    labels: np.ndarray                    # signed cluster id per voxel
    sd: Optional[np.ndarray] = None
    consistency: Optional[np.ndarray] = None

    @property
    def key(self) -> Tuple[str, str]:
        return (self.condition, self.combo.name)


@dataclass
class ScanResult:
    run: LoadedRun
    params: ScanParams
    combos: List[Combination]
    conditions: List[Condition]
    effects: Dict[Tuple[str, str], EffectMap]
    regions: al.VoxelRegions
    notes: List[str] = field(default_factory=list)
    boot: Optional[BootstrapStats] = None

    # ---- tidy tables ----

    @property
    def stat_label(self) -> str:
        return "z (bootstrap)" if self.params.mode == "bootstrap" else "Δ"

    def peaks(self) -> pd.DataFrame:
        frames = [e.peaks.assign(condition=e.condition, combo=e.combo.name, kind=e.combo.kind)
                  for e in self.effects.values() if len(e.peaks)]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        front = ["condition", "kind", "combo", "cluster_id", "sign", "peak_type",
                 "ho_region", "ho_prob", "ho_flag", "schaefer_parcel", "network",
                 "x", "y", "z", "peak_stat", "peak_delta", "n_voxels", "volume_mm3"]
        return df[[c for c in front if c in df] + [c for c in df if c not in front]]

    def summary(self) -> pd.DataFrame:
        """One row per (condition, combination), ranked most-interesting first."""
        rows = []
        for e in self.effects.values():
            pk = e.peaks[e.peaks["peak_type"] == "peak"] if len(e.peaks) else e.peaks
            top = pk.sort_values("volume_mm3", ascending=False).iloc[0] if len(pk) else None
            rows.append(dict(
                condition=e.condition, kind=e.combo.kind, combo=e.combo.name,
                description=e.combo.description,
                n_activation=int((pk["sign"] == "activation").sum()) if len(pk) else 0,
                n_deactivation=int((pk["sign"] == "deactivation").sum()) if len(pk) else 0,
                suprathreshold_mm3=float(pk["volume_mm3"].sum()) if len(pk) else 0.0,
                peak_abs_stat=float(np.abs(e.stat).max()),
                delta_l2=float(np.linalg.norm(e.delta)),
                delta_max_abs=float(np.abs(e.delta).max()),
                largest_blob=(f"{top['ho_region']} ({top['sign']})" if top is not None else ""),
                threshold=e.threshold,
            ))
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(["suprathreshold_mm3", "delta_l2"], ascending=False, ignore_index=True)
            df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df

    def roi_effects(self, atlas: str = "yeo7") -> pd.DataFrame:
        """Mean Δ per region for every effect (+ bootstrap 95 % CI when available)."""
        codes, names = ((self.regions.yeo7, self.regions.yeo7_names) if atlas == "yeo7"
                        else (self.regions.ho, self.regions.ho_names))
        rows = []
        for cond in self.conditions:
            keys = [k for k in self.effects if k[0] == cond.name]
            if not keys:
                continue
            D = np.stack([self.effects[k].delta for k in keys])
            means = al.region_means(D, codes, len(names))
            boot = (self.boot.roi_boot.get(cond.name, {}).get(atlas)
                    if self.boot is not None else None)
            combo_idx = {c.name: i for i, c in enumerate(self.combos)}
            for i, k in enumerate(keys):
                for r, name in enumerate(names):
                    row = dict(condition=cond.name, combo=k[1], kind=self.effects[k].combo.kind,
                               region=name, mean_delta=means[i, r])
                    if boot is not None:
                        b = boot[:, combo_idx[k[1]], r]
                        row.update(ci_low=np.nanpercentile(b, 2.5), ci_high=np.nanpercentile(b, 97.5))
                    rows.append(row)
        return pd.DataFrame(rows)

    def params_dict(self) -> dict:
        d = asdict(self.params)
        d.update(run_dir=str(self.run.run_dir), n_boot_done=self.boot.n_boot if self.boot else 0,
                 conditions=[c.name for c in self.conditions], notes=self.notes)
        return json.loads(json.dumps(d, default=str))


# ---------------------------------------------------------------- core steps


def compute_deltas(run: LoadedRun, combos: Sequence[Combination],
                   conditions: Sequence[Condition]) -> Dict[str, np.ndarray]:
    """cond name -> (n_combos, n_voxels) effect maps from the trained predictor."""
    W, P = stack(combos)
    return {c.name: run.combine(W, P, c) for c in conditions}


def label_peaks(peaks: pd.DataFrame, labels: np.ndarray, regions: al.VoxelRegions) -> pd.DataFrame:
    if not len(peaks):
        return peaks
    lab = pd.DataFrame([al.label_coordinate((x, y, z))
                        for x, y, z in peaks[["x", "y", "z"]].to_numpy()])
    comp = {cid: al.cluster_composition(regions.ho[labels == cid], regions.ho_names)
            for cid in peaks["cluster_id"].unique()}
    out = pd.concat([peaks.reset_index(drop=True), lab], axis=1)
    out["regions_covered"] = out["cluster_id"].map(comp)
    return out


def threshold_effect(
    run: LoadedRun, regions: al.VoxelRegions, condition: str, combo: Combination,
    delta: np.ndarray, params: ScanParams, threshold: Optional[float] = None,
    sd: Optional[np.ndarray] = None, consistency: Optional[np.ndarray] = None,
) -> EffectMap:
    """Statistic + threshold + labelled blobs for one Δ. ``threshold`` overrides the params."""
    if params.mode == "bootstrap":
        if sd is None:
            raise ValueError("bootstrap mode needs the bootstrap SD map")
        stat = z_from_bootstrap(delta, sd)
        if threshold is None:
            threshold = fdr_z_threshold(stat, params.fdr_q) if params.fdr_q else params.z_thr
    else:
        stat = np.asarray(delta, dtype=np.float32)
        if threshold is None:
            threshold = (params.abs_threshold if params.abs_threshold is not None
                         else pooled_percentile_threshold([delta], params.percentile))
    peaks, labels = find_clusters(stat, threshold, run.mask_img, delta=delta,
                                  consistency=consistency, params=params.cluster_params)
    return EffectMap(
        condition=condition, combo=combo, delta=np.asarray(delta, dtype=np.float32), stat=stat,
        threshold=float(threshold), peaks=label_peaks(peaks, labels, regions), labels=labels,
        sd=sd, consistency=consistency,
    )


def run_scan(
    run: Union[LoadedRun, str, Path],
    params: ScanParams = ScanParams(),
    extra_combos: Sequence[Combination] = (),
    progress: Optional[Callable[[str, float], None]] = None,
) -> ScanResult:
    """Scan a trained run. ``progress(stage, fraction)`` is called as work proceeds."""
    say = progress or (lambda stage, frac: None)
    if not isinstance(run, LoadedRun):
        say("loading run", 0.0)
        run = LoadedRun.from_dir(run)
    seed = params.seed if params.seed is not None else int(run.config.random_seed)

    combos, notes = build_combinations(run, params.combos, params.k_sd, params.archetypes_k,
                                       params.features, seed=seed)
    combos += list(extra_combos)
    if not combos:
        raise ValueError("no combinations to scan")
    conditions = run.conditions()
    if params.conditions:
        conditions = [c for c in conditions if c.name in params.conditions]
        if not conditions:
            raise ValueError(f"none of {params.conditions} among the run's conditions")

    say("labelling model voxels", 0.05)
    regions = al.voxel_regions(run.mask_img)
    say("computing effect maps", 0.1)
    deltas = compute_deltas(run, combos, conditions)

    boot = None
    thr_global = None
    if params.mode == "bootstrap":
        boot = bootstrap(
            run, combos, conditions, deltas, n_boot=params.n_boot, seed=seed, regions=regions,
            progress=lambda b, n: say(f"bootstrap {b}/{n}", 0.1 + 0.8 * b / n),
        )
    elif params.abs_threshold is None:
        thr_global = pooled_percentile_threshold(list(deltas.values()), params.percentile)
        notes.append(f"Effect-mode threshold |Δ| > {thr_global:.4g} = the {params.percentile:g}th "
                     f"percentile of |Δ| pooled over all {len(combos) * len(conditions)} maps.")

    effects: Dict[Tuple[str, str], EffectMap] = {}
    for c in conditions:
        for i, combo in enumerate(combos):
            effects[(c.name, combo.name)] = threshold_effect(
                run, regions, c.name, combo, deltas[c.name][i], params,
                threshold=thr_global,
                sd=boot.sd[c.name][i] if boot else None,
                consistency=boot.consistency[c.name][i] if boot else None,
            )
    say("done", 1.0)
    return ScanResult(run=run, params=params, combos=combos, conditions=conditions,
                      effects=effects, regions=regions, notes=notes, boot=boot)
