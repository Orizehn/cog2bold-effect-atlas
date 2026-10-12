"""Leave-One-Out evaluation: is the model better than the population-mean map?

For each held-out unit (a (subject, session) sample-key by default, or a whole
subject):
  1. train a fresh basis + predictor on all OTHER samples
  2. for each stimulus/contrast the held-out unit has:
       - predict the held-out map from its (cog, stim) features
       - build the population-mean map for that stimulus over the training set
       - r_pred = corr(prediction, GT) ; r_mean = corr(mean map, GT)
       - "win" iff r_pred > r_mean ; delta = r_pred - r_mean

Reported per correlation space, per contrast, and overall:
  - accuracy = #wins / N      (chance = 0.5; one-sided binomial test)
  - mean / sum of delta       (one-sided Wilcoxon signed-rank test, delta > 0)

The population-mean map IS the null here, so no cog-shuffle permutation is run
(unlike LTO / data-informativeness).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon, t as student_t
from tqdm import tqdm

from cog2bold.config import CorrelationConfig, RunConfig
from cog2bold.io import save_json
from cog2bold.pipelines._eval_base import (
    EVAL_METRICS,
    BaseLeaveOutEvaluator,
    metric_delta,
    metric_is_better,
)


@dataclass
class LOOArgs:
    """Parameters for the leave-one-out evaluation (CLI + IDE share this)."""

    run_dir: Path
    output_dir: Optional[Path] = None
    save_maps: bool = False
    leave_out_level: str = "subject_session"  # | "subject"
    max_units: Optional[int] = None
    correlation_configs: Optional[List[CorrelationConfig]] = None
    train_atlas_cfg: Optional[CorrelationConfig] = None
    p_threshold: Optional[float] = None
    p_threshold_mode: str = "zero"
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "LOOArgs":
        d = dict(d)
        if d.get("run_dir") is not None:
            d["run_dir"] = Path(d["run_dir"])
        if d.get("output_dir") is not None:
            d["output_dir"] = Path(d["output_dir"])
        if d.get("correlation_configs") is not None:
            d["correlation_configs"] = [
                CorrelationConfig.from_dict(c) if isinstance(c, dict) else c
                for c in d["correlation_configs"]
            ]
        if isinstance(d.get("train_atlas_cfg"), dict):
            d["train_atlas_cfg"] = CorrelationConfig.from_dict(d["train_atlas_cfg"])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


def _loo_stats(deltas: List[float], wins: List[bool]) -> Dict[str, object]:
    """Accuracy (binomial vs 0.5) + improvement (Wilcoxon on deltas) statistics."""
    n = len(deltas)
    if n == 0:
        return {
            "n_trials": 0, "n_win": 0, "accuracy": 0.0,
            "mean_delta": 0.0, "sum_delta": 0.0,
            "binomial_p": None, "wilcoxon_p": None,
            "delta_ci95_low": None, "delta_ci95_high": None,
        }
    d = np.asarray(deltas, dtype=float)
    n_win = int(np.sum(wins))
    # one-sided binomial: P(#wins this high | p=0.5)
    binom_p = float(binomtest(n_win, n, 0.5, alternative="greater").pvalue)
    # one-sided Wilcoxon signed-rank that median delta > 0 (needs ≥1 nonzero delta)
    wil_p: Optional[float] = None
    if np.any(np.abs(d) > 1e-12):
        try:
            wil_p = float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            wil_p = None
    # t-based 95% CI on the mean delta
    if n > 1 and d.std(ddof=1) > 1e-12:
        se = d.std(ddof=1) / np.sqrt(n)
        tcrit = float(student_t.ppf(0.975, n - 1))
        lo, hi = float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)
    else:
        lo = hi = float(d.mean())
    return {
        "n_trials": n,
        "n_win": n_win,
        "accuracy": n_win / n,
        "mean_delta": float(d.mean()),
        "sum_delta": float(d.sum()),
        "binomial_p": binom_p,
        "wilcoxon_p": wil_p,
        "delta_ci95_low": lo,
        "delta_ci95_high": hi,
    }


class LeaveOneOutEvaluator(BaseLeaveOutEvaluator):
    def __init__(self, run_config: RunConfig, args: LOOArgs):
        super().__init__(run_config)
        self.args = args
        self.correlation_configs: List[CorrelationConfig] = self._resolve_correlation_configs(
            args.correlation_configs
        )
        self.train_atlas_cfg = args.train_atlas_cfg
        self.p_threshold = args.p_threshold
        self.p_threshold_mode = args.p_threshold_mode
        self._enforce_correlation_config_compatibility()

        self.output_dir: Path = self._make_output_dir()
        self.results: List[dict] = []

    # ---- output dir ----

    def _make_output_dir(self) -> Path:
        if self.args.output_dir is not None:
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            return self.args.output_dir
        parts = ["loo_eval", self.args.leave_out_level]
        if self.args.max_units:
            parts.append(f"maxunits{self.args.max_units}")
        if self.args.p_threshold is not None:
            parts.append(
                f"p{self.args.p_threshold:.4f}".replace(".", "p").rstrip("0")
                + f"-{self.args.p_threshold_mode}"
            )
        corr_parts = []
        for c in self.correlation_configs:
            if c.space == 0:
                corr_parts.append("vox")
            elif c.space == 1:
                corr_parts.append(f"atlas{c.atlas_n_rois}")
            else:
                corr_parts.append(f"roi{c.atlas_n_rois}_" + "+".join(c.roi_name_patterns or []))
        parts.append("corr_" + "+".join(corr_parts))
        if self.args.train_atlas_cfg is not None:
            parts.append(f"train_{self.args.train_atlas_cfg.name}")
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        out = self.run_config.run_dir / "_".join(parts)
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ---- unit grouping ----

    def _unit_key(self, idx: int) -> Tuple:
        s = self.samples[idx]
        if self.args.leave_out_level == "subject":
            return (int(s.subject_id),)
        return (int(s.subject_id), int(s.session_id))

    def _group_units(self) -> Dict[Tuple, List[int]]:
        units: Dict[Tuple, List[int]] = {}
        for i in range(len(self.samples)):
            units.setdefault(self._unit_key(i), []).append(i)
        return units

    # ---- main loop ----

    def run(self) -> float:
        if not self.samples:
            self.load_dataset()

        units = self._group_units()
        unit_keys = list(units.keys())
        if self.args.max_units and len(unit_keys) > self.args.max_units:
            rng = np.random.default_rng(self.args.seed)
            sel = rng.choice(len(unit_keys), size=self.args.max_units, replace=False)
            unit_keys = [unit_keys[i] for i in sorted(sel)]

        print(
            f"\n[LOO] {len(unit_keys)} folds (level={self.args.leave_out_level}), "
            f"{len(self.samples)} samples, chance=0.5 (model vs mean map)"
        )

        all_indices = set(range(len(self.samples)))
        n_win = {m: 0 for m in EVAL_METRICS}
        n_trials = 0

        for u in tqdm(unit_keys, desc="[LOO] folds"):
            held_idx = units[u]
            train_idx = sorted(all_indices.difference(held_idx))
            if not train_idx:
                continue

            basis, predictor = self._train_on_subset(train_idx)

            # population-mean map per stimulus/contrast over the training set
            mean_by_contrast: Dict[str, np.ndarray] = {}
            train_by_contrast: Dict[str, List[int]] = {}
            for k in train_idx:
                train_by_contrast.setdefault(self._contrast_name_of(k), []).append(k)
            for cname, ks in train_by_contrast.items():
                mean_by_contrast[cname] = np.mean([self._get_map(k) for k in ks], axis=0)

            for i in held_idx:
                contrast = self._contrast_name_of(i)
                if contrast not in mean_by_contrast:
                    continue  # no baseline for a stimulus only the held-out unit has
                gt = self._apply_threshold(self._get_map(i))
                pred = self._predict(self.samples[i], basis, predictor)
                meanmap = mean_by_contrast[contrast]

                # per_cfg[name][metric] = {v_pred, v_mean, delta, win}
                per_cfg: Dict[str, Dict[str, dict]] = {}
                for cfg in self.correlation_configs:
                    per_cfg[cfg.name] = {}
                    for m in EVAL_METRICS:
                        v_pred = self._metric_value(pred, gt, cfg, metric=m)
                        v_mean = self._metric_value(meanmap, gt, cfg, metric=m)
                        per_cfg[cfg.name][m] = {
                            "v_pred": v_pred,
                            "v_mean": v_mean,
                            # positive delta => model beats the mean map (both metrics)
                            "delta": metric_delta(m, v_pred, v_mean),
                            "win": bool(metric_is_better(m, v_pred, v_mean)),
                        }

                primary = self.correlation_configs[0].name
                n_trials += 1
                for m in EVAL_METRICS:
                    if per_cfg[primary][m]["win"]:
                        n_win[m] += 1

                pc = per_cfg[primary]["correlation"]
                pm = per_cfg[primary]["mse"]
                row = {
                    "trial_id": n_trials - 1,
                    "subject": int(self.samples[i].subject_id),
                    "session": int(self.samples[i].session_id),
                    "unit": "_".join(map(str, u)),
                    "idx": i,
                    "contrast": contrast,
                    "is_contrast_map": bool(self.samples[i].is_contrast),
                    "cog": self.samples[i].cog_features.tolist(),
                    # primary correlation (legacy columns)
                    "r_pred": pc["v_pred"], "r_mean": pc["v_mean"],
                    "delta": pc["delta"], "win": pc["win"],
                    # primary MSE
                    "mse_pred": pm["v_pred"], "mse_mean": pm["v_mean"],
                    "mse_delta": pm["delta"], "win_mse": pm["win"],
                }
                for cfg in self.correlation_configs:
                    cc = per_cfg[cfg.name]["correlation"]
                    mm = per_cfg[cfg.name]["mse"]
                    row[f"r_pred_{cfg.name}"] = cc["v_pred"]
                    row[f"r_mean_{cfg.name}"] = cc["v_mean"]
                    row[f"delta_{cfg.name}"] = cc["delta"]
                    row[f"win_{cfg.name}"] = cc["win"]
                    row[f"mse_pred_{cfg.name}"] = mm["v_pred"]
                    row[f"mse_mean_{cfg.name}"] = mm["v_mean"]
                    row[f"mse_delta_{cfg.name}"] = mm["delta"]
                    row[f"win_mse_{cfg.name}"] = mm["win"]
                self.results.append(row)

                if self.args.save_maps:
                    mdir = self.output_dir / "maps" / ("_".join(map(str, u))) / _safe_name(contrast)
                    mdir.mkdir(parents=True, exist_ok=True)
                    np.save(mdir / "pred.npy", pred)
                    np.save(mdir / "gt.npy", gt)
                    np.save(mdir / "mean.npy", meanmap)

            if n_trials:
                tqdm.write(
                    f"  corr acc {n_win['correlation']}/{n_trials} = "
                    f"{n_win['correlation']/n_trials:.3f} | "
                    f"mse acc {n_win['mse']/n_trials:.3f}  (chance 0.500)"
                )

        self.save_results()
        return n_win["correlation"] / n_trials if n_trials else 0.0

    # ---- serialisation ----

    def save_results(self) -> None:
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "loo_results_detailed.csv", index=False)

        # Column naming per metric: correlation uses the legacy "delta"/"win"
        # (primary) + "delta_{cfg}"/"win_{cfg}"; mse uses "mse_delta"/"win_mse".
        def _cols(metric: str, cfg_name: Optional[str]) -> Tuple[str, str]:
            if metric == "correlation":
                if cfg_name is None:
                    return "delta", "win"
                return f"delta_{cfg_name}", f"win_{cfg_name}"
            if cfg_name is None:
                return "mse_delta", "win_mse"
            return f"mse_delta_{cfg_name}", f"win_mse_{cfg_name}"

        def stats_for(mask_df: pd.DataFrame, metric: str, cfg_name: Optional[str]) -> Dict[str, object]:
            dcol, wcol = _cols(metric, cfg_name)
            if dcol not in mask_df.columns:
                return _loo_stats([], [])
            return _loo_stats(mask_df[dcol].tolist(), mask_df[wcol].tolist())

        def metric_block(metric: str) -> Dict[str, object]:
            overall = stats_for(df, metric, None) if len(df) else _loo_stats([], [])
            by_cfg = {
                cfg.name: (stats_for(df, metric, cfg.name) if len(df) else _loo_stats([], []))
                for cfg in self.correlation_configs
            }
            per_contrast = {}
            if len(df):
                for cname, group in df.groupby("contrast"):
                    per_contrast[cname] = {
                        "primary": stats_for(group, metric, None),
                        "by_corr_config": {
                            cfg.name: stats_for(group, metric, cfg.name)
                            for cfg in self.correlation_configs
                        },
                    }
            return {"overall": overall, "by_corr_config": by_cfg, "per_contrast": per_contrast}

        metrics_blocks = {m: metric_block(m) for m in EVAL_METRICS}
        corr_block = metrics_blocks["correlation"]

        summary = {
            "evaluation_method": "Leave-One-Out (prediction vs population-mean map)",
            "chance_level": 0.5,
            "eval_metrics": list(EVAL_METRICS),
            "timestamp": datetime.now().isoformat(),
            "leave_out_level": self.args.leave_out_level,
            "correlation_configs": [c.to_dict() for c in self.correlation_configs],
            "constraints": {
                "max_units": self.args.max_units,
                "p_threshold": self.args.p_threshold,
                "p_threshold_mode": self.args.p_threshold_mode,
                "train_atlas_cfg": self.args.train_atlas_cfg.to_dict() if self.args.train_atlas_cfg else None,
            },
            # Legacy top-level correlation block (back-compat).
            "overall": corr_block["overall"],
            "by_corr_config": corr_block["by_corr_config"],
            "per_contrast": corr_block["per_contrast"],
            # New: both metrics, toggled in the viewer.
            "metrics": metrics_blocks,
            "model_config": {
                "basis_type": self.run_config.basis_config.basis_type,
                "n_components": int(self.run_config.basis_config.n_components),
                "predictor_type": self.run_config.predictor_config.model_type,
                "loss_mode": self.run_config.predictor_config.loss_mode,
                "adapter_name": self.run_config.data_config.adapter_name,
            },
        }
        save_json(summary, self.output_dir / "loo_results_summary.json")

        ov = corr_block["overall"]
        mse_ov = metrics_blocks["mse"]["overall"]
        print(
            f"\n[LOO] DONE — corr accuracy {ov['n_win']}/{ov['n_trials']} = "
            f"{ov['accuracy']:.3f} | mse accuracy {mse_ov['accuracy']:.3f} (chance 0.500)\n"
            f"      corr mean Δ(r_pred−r_mean) = {ov['mean_delta']:+.4f}, "
            f"sum Δ = {ov['sum_delta']:+.3f}\n"
            f"      corr binomial p = {ov['binomial_p']}, wilcoxon p = {ov['wilcoxon_p']}"
        )


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:60]


def run_loo_from_run_dir(args: LOOArgs) -> float:
    """Load RunConfig from `args.run_dir`, then run LOO."""
    cfg_path = Path(args.run_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json in {args.run_dir}")
    run_config = RunConfig.load(cfg_path)
    evaluator = LeaveOneOutEvaluator(run_config, args)
    evaluator.load_dataset()
    return evaluator.run()
