"""Leave-Two-Out classification (Walters et al. 2022 style).

For every valid pair (i, j) within the same stimulus contrast:
  1. train a fresh basis + predictor on N-2 samples
  2. predict the two held-out maps
  3. count as correct iff r(p1,t1) > r(p1,t2) AND r(p2,t2) > r(p2,t1)

Nominal chance is 25%, but the TRUE chance level is `1/4 - Var(F(W)) <= 1/4`, where W is
the hubness gap between the two held-out ground-truth maps — the two sub-decisions share
those maps with opposite roles and are therefore negatively correlated. See
`docs/data_informativeness_null_below_chance.tex`. Judge every accuracy against the
empirical null (`run_null`), never against the flat 0.25 reported in the summary JSON.

`score_space="deviation"` subtracts the per-contrast mean of the N-2 training maps from
both sides of every comparison (see LTOArgs.score_space). That removes the shared component
which otherwise makes both predictions rank the two true maps identically — the concrete
driver of the below-chance accuracies this evaluator reports in map space.

With `store_null_verdicts`, a `null_verdicts.npz` is written holding the per-pair verdict
of every pass. Since a pair's verdict does not depend on `min_cog_distance` (training is
always the other N-2 samples), accuracy-vs-distance curves, their per-point null bands and
the pair-level trend test are all post-hoc functionals of that one file — see
`cog2bold/analysis/lto_curve_calibration.py`.

Bug fixes vs the legacy `evaluation/leave_two_out_classification.py`:
  - single `LTOArgs` dataclass for both CLI and IDE; no Args.__init__ silently
    dropping IDE keys (#2)
  - atlas masker setup wrapped in try/finally so a failed `_setup_atlas_correlation`
    can't leave correlation_configs mutated (#12)
  - cog stats train-only (#3)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from cog2bold.config import CorrelationConfig, RunConfig
from cog2bold.io import save_json, save_pickle
from cog2bold.pipelines._eval_base import (
    EVAL_METRICS,
    BaseLeaveOutEvaluator,
    metric_is_better,
    safe_corr,
)


# Backwards-compatible alias (other modules import `_safe_corr` from here)
_safe_corr = safe_corr


@dataclass
class LTOArgs:
    """Single source of truth for LTO parameters.

    Constructed from CLI args or directly from a dict — any new field added here
    is automatically picked up by both paths (no more silent IDE/CLI drift).
    """

    run_dir: Path
    save_maps: bool = False
    output_dir: Optional[Path] = None
    max_pairs_per_contrast: Optional[int] = None
    different_subjects: bool = False
    different_sessions: bool = False
    min_cog_distance: Optional[float] = None
    unique_samples: bool = False
    same_diagnosis: bool = False  # only pair two samples from the SAME diagnosis group
    p_threshold: Optional[float] = None
    p_threshold_mode: str = "zero"  # "zero" (hard) | "attenuate" (soft, factor=0.1)
    session_group_a: Optional[List[int]] = None
    session_group_b: Optional[List[int]] = None
    correlation_configs: Optional[List[CorrelationConfig]] = None
    train_atlas_cfg: Optional[CorrelationConfig] = None

    # Space the metrics are computed in:
    #   "map"       — absolute maps, as saved by the GLM (the historical behaviour)
    #   "deviation" — subtract the per-contrast mean of the N-2 TRAINING maps from both the
    #                 prediction and the ground truth before scoring.
    # Absolute maps of one contrast share a large common component μ, so r(pred_k, true_j)
    # is dominated by cos(μ, true_j) — a term that depends on j but barely on k. Both
    # predictions then rank the two true maps the same way and the pair fails regardless of
    # cognition, which is what pushes LTO accuracy *below* the nominal 25%. Scoring in
    # deviation space removes that common term so the comparison sees only the
    # sample-specific structure the model is actually asked to predict.
    # Independent of RunConfig.mean_map_offset (what the *model* fits): any run can be
    # scored either way, and the μ used here is always refit on the pair's training subset,
    # so the two held-out maps never leak into it.
    score_space: str = "map"

    # Empirical null model: shuffle map<->cog correspondence and re-run LTO.
    # When `run_null=True`, n_null_permutations extra passes are executed; for
    # each pass, training cog vectors are reassigned (see null_shuffle_scope).
    # The summary JSON gains a `null_model` block with empirical p-values, null
    # mean/CI, and (optionally) the per-permutation distribution.
    run_null: bool = False
    n_null_permutations: int = 0
    null_seed: int = 1
    null_shuffle_scope: str = "within_contrast"
    # "within_contrast" | "global" | "within_contrast_diagnosis"
    # Use "within_contrast_diagnosis" whenever the cohort spans diagnosis groups AND pairs
    # are formed across them: a cross-group shuffle also breaks the cognition<->diagnosis
    # association, so the observed-minus-null gap absorbs the group confound rather than
    # isolating cognition. See _make_cog_override.
    null_shuffle_session_labels: bool = False
    store_null_distribution: bool = True

    # Write `null_verdicts.npz` — the per-pair verdict of the observed pass plus a
    # (n_permutations, n_pairs) boolean matrix for the null passes. Because a pair's
    # verdict does NOT depend on min_cog_distance (training is always N-2 samples), any
    # accuracy-vs-distance curve, its empirical null band, and the pair-level trend test
    # are post-hoc functionals of this one artifact — see
    # cog2bold/analysis/lto_curve_calibration.py. Requires run_null for the null matrix.
    store_null_verdicts: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "LTOArgs":
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


# ---- correlation helpers ----


def two_way_correct(
    pred1: np.ndarray, pred2: np.ndarray, true1: np.ndarray, true2: np.ndarray,
    corr_fn: Callable[[np.ndarray, np.ndarray], float] = _safe_corr,
) -> Tuple[bool, Dict[str, float]]:
    r11 = corr_fn(pred1, true1)
    r12 = corr_fn(pred1, true2)
    r21 = corr_fn(pred2, true1)
    r22 = corr_fn(pred2, true2)
    return (r11 > r12) and (r22 > r21), {"r11": r11, "r12": r12, "r21": r21, "r22": r22}


# ---- null-model statistics ----


def _summarise_null(observed: float, null_values: List[float]) -> Dict[str, object]:
    """Compute null-model statistics for a single accuracy axis.

    Returned keys mirror what the streamlit app expects (`mean`, `ci95_low/high`,
    `empirical_p_one_sided`, `delta_over_null_mean`, `z_observed_vs_null`).
    """
    if not null_values:
        return {
            "observed": float(observed),
            "n_permutations": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "empirical_p_one_sided": None,
            "delta_over_null_mean": None,
            "z_observed_vs_null": None,
        }
    arr = np.asarray(null_values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    lo, hi = (float(x) for x in np.percentile(arr, [2.5, 97.5]))
    # one-sided empirical p with +1 correction (Phipson & Smyth 2010)
    p = float((1 + int((arr >= observed).sum())) / (1 + arr.size))
    z = float((observed - mean) / std) if std > 1e-12 else None
    return {
        "observed": float(observed),
        "n_permutations": int(arr.size),
        "mean": mean,
        "std": std,
        "ci95_low": lo,
        "ci95_high": hi,
        "empirical_p_one_sided": p,
        "delta_over_null_mean": float(observed - mean),
        "z_observed_vs_null": z,
    }


# ---- main classifier ----


class LeaveTwoOutClassifier(BaseLeaveOutEvaluator):
    def __init__(self, run_config: RunConfig, args: LTOArgs):
        super().__init__(run_config)
        self.args = args

        # attributes the base reads
        self.correlation_configs: List[CorrelationConfig] = self._resolve_correlation_configs(
            args.correlation_configs
        )
        self.train_atlas_cfg = args.train_atlas_cfg
        self.p_threshold = args.p_threshold
        self.p_threshold_mode = args.p_threshold_mode
        if args.score_space not in ("map", "deviation"):
            raise ValueError(
                f"score_space must be 'map' or 'deviation', got {args.score_space!r}"
            )
        self.score_space = args.score_space
        self._enforce_correlation_config_compatibility()  # may filter or raise

        self.output_dir: Path = self._make_output_dir()
        self.results: List[dict] = []

        # Populated by run() — observed first, then null distributions
        self._observed_summary: Dict[str, object] = {}
        self.null_metrics: Dict[str, object] = {}
        self._used_samples: set = set()  # tracker for unique_samples constraint

        # Per-pair verdict log (see LTOArgs.store_null_verdicts)
        self._verdict_pairs: Optional[np.ndarray] = None     # (n_pairs, 2) of idx1/idx2
        self._observed_verdicts: Dict[str, np.ndarray] = {}  # "metric__cfg" -> (n_pairs,)
        self._null_verdicts: Dict[str, np.ndarray] = {}      # "metric__cfg" -> (B, n_pairs)

    # ---- setup ----

    def _make_output_dir(self) -> Path:
        if self.args.output_dir is not None:
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            return self.args.output_dir
        parts = ["lto_eval"]
        if self.args.max_pairs_per_contrast:
            parts.append(f"maxpairs{self.args.max_pairs_per_contrast}")
        if self.args.different_subjects:
            parts.append("diffsubj")
        if self.args.different_sessions:
            parts.append("diffsess")
        if self.args.min_cog_distance is not None:
            parts.append(f"mindist{self.args.min_cog_distance:.1f}".replace(".", "p"))
        if self.args.unique_samples:
            parts.append("unique")
        if self.args.same_diagnosis:
            parts.append("samedx")
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
        if self.score_space == "deviation":
            parts.append("devspace")
        if self.args.train_atlas_cfg is not None:
            parts.append(f"train_{self.args.train_atlas_cfg.name}")
        if self.args.session_group_a and self.args.session_group_b:
            a = "-".join(map(str, self.args.session_group_a))
            b = "-".join(map(str, self.args.session_group_b))
            parts.append(f"sesA{a}_sesB{b}")
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        out = self.run_config.run_dir / "_".join(parts)
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ---- pair generation ----

    def _passes_constraints(
        self, i1: int, i2: int, session_override: Optional[Dict[int, int]] = None,
    ) -> bool:
        s1, s2 = self.samples[i1], self.samples[i2]
        if self.args.different_subjects and s1.subject_id == s2.subject_id:
            return False
        if self.args.different_sessions:
            ses1 = self._session_id(i1, session_override)
            ses2 = self._session_id(i2, session_override)
            if ses1 == ses2:
                return False
        if self.args.min_cog_distance is not None:
            d = float(np.linalg.norm((s1.cog_features - s2.cog_features) / self.cog_std))
            if d < self.args.min_cog_distance:
                return False
        if self.args.same_diagnosis:
            g1, g2 = self._diagnosis(s1.subject_id), self._diagnosis(s2.subject_id)
            if g1 is None or g2 is None or g1 != g2:
                return False
        return True

    def _diagnosis(self, subject_id: int):
        """Best-effort subject->diagnosis via the adapter's _diag_by_sid map (ds000030);
        None if the dataset exposes no diagnosis info."""
        return getattr(self.adapter, "_diag_by_sid", {}).get(int(subject_id))

    def _session_id(self, idx: int, session_override: Optional[Dict[int, int]] = None) -> int:
        if session_override is not None and idx in session_override:
            return int(session_override[idx])
        return int(self.samples[idx].session_id)

    def _make_pairs(
        self,
        indices: List[int],
        session_override: Optional[Dict[int, int]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[int, int]]:
        rng = rng or random.Random(self.run_config.random_seed)
        if self.args.session_group_a and self.args.session_group_b:
            return self._make_session_split_pairs(
                indices, session_override=session_override, rng=rng,
            )
        rnd = list(indices)
        rng.shuffle(rnd)
        out = []
        for i, j in combinations(rnd, 2):
            if self.args.unique_samples and (i in self._used_samples or j in self._used_samples):
                continue
            if not self._passes_constraints(i, j, session_override=session_override):
                continue
            out.append((i, j))
            if self.args.unique_samples:
                self._used_samples.update({i, j})
        return out

    def _make_session_split_pairs(
        self,
        indices: List[int],
        session_override: Optional[Dict[int, int]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[int, int]]:
        rng = rng or random.Random(self.run_config.random_seed)
        a_set = set(self.args.session_group_a or [])
        b_set = set(self.args.session_group_b or [])
        a_idx = [i for i in indices if self._session_id(i, session_override) in a_set]
        b_idx = [i for i in indices if self._session_id(i, session_override) in b_set]
        rng.shuffle(a_idx)
        rng.shuffle(b_idx)
        out = []
        for ia in a_idx:
            for ib in b_idx:
                if self.args.unique_samples and (ia in self._used_samples or ib in self._used_samples):
                    continue
                if not self._passes_constraints(ia, ib, session_override=session_override):
                    continue
                out.append((ia, ib))
                if self.args.unique_samples:
                    self._used_samples.update({ia, ib})
        return out

    # ---- core loop ----

    def run(self) -> float:
        if not self.samples:
            self.load_dataset()

        # ---- observed pass ----
        observed = self._run_one_pass(
            label="observed", cog_override=None, session_override=None,
            collect_results=True, show_progress=True,
        )
        self.results = observed["rows"]
        self._observed_summary = observed
        self._verdict_pairs = observed["verdict_pairs"]
        self._observed_verdicts = observed["verdicts"]
        corr_block = observed["by_metric"]["correlation"]
        mse_block = observed["by_metric"]["mse"]
        print(
            f"\n[LTO] OBSERVED — total {observed['total_pairs']}  "
            f"| corr acc {corr_block['correct_pairs']}/{observed['total_pairs']} "
            f"= {corr_block['accuracy']:.3f}  "
            f"| mse acc {mse_block['accuracy']:.3f}  "
            f"| scored in {self.score_space} space  "
            f"(nominal 0.250 — judge against the empirical null, not this)"
        )

        # ---- null permutations ----
        # Collect accuracy distributions for BOTH metrics in one sweep (verdicts
        # for both are already computed per pass, so this is free).
        self.null_metrics = {}
        if self.args.run_null and self.args.n_null_permutations > 0:
            all_indices = list(range(len(self.samples)))
            groups = self._group_by_contrast()
            null_overall = {m: [] for m in EVAL_METRICS}
            null_per_cfg = {
                m: {c.name: [] for c in self.correlation_configs} for m in EVAL_METRICS
            }
            null_per_contrast = {
                m: {name: [] for name in groups.keys()} for m in EVAL_METRICS
            }
            null_verdict_lists: Dict[str, List[np.ndarray]] = {
                k: [] for k in self._observed_verdicts
            }

            for perm_idx in range(self.args.n_null_permutations):
                rng_np = np.random.default_rng(self.args.null_seed + perm_idx)
                cog_override = self._make_cog_override(all_indices, groups, rng_np)
                session_override = (
                    self._make_session_override(all_indices, rng_np)
                    if self.args.null_shuffle_session_labels else None
                )
                pass_summary = self._run_one_pass(
                    label=f"null-{perm_idx + 1}",
                    cog_override=cog_override,
                    session_override=session_override,
                    collect_results=False,
                    show_progress=False,
                )
                self._assert_pair_alignment(pass_summary)
                for k, arr in pass_summary["verdicts"].items():
                    null_verdict_lists[k].append(arr)
                for m in EVAL_METRICS:
                    block = pass_summary["by_metric"][m]
                    null_overall[m].append(block["accuracy"])
                    for cfg in self.correlation_configs:
                        null_per_cfg[m][cfg.name].append(
                            block["by_corr_config"][cfg.name]["accuracy"]
                        )
                    for cname in null_per_contrast[m]:
                        if cname in block["per_contrast"]:
                            null_per_contrast[m][cname].append(
                                block["per_contrast"][cname]["accuracy"]
                            )
                print(
                    f"[LTO] null {perm_idx + 1}/{self.args.n_null_permutations}: "
                    f"corr acc={pass_summary['by_metric']['correlation']['accuracy']:.3f}  "
                    f"mse acc={pass_summary['by_metric']['mse']['accuracy']:.3f}"
                )

            self.null_metrics = {
                m: self._build_null_summary(
                    observed["by_metric"][m],
                    null_overall[m], null_per_cfg[m], null_per_contrast[m],
                )
                for m in EVAL_METRICS
            }
            self._null_verdicts = {
                k: np.vstack(v) for k, v in null_verdict_lists.items() if v
            }

        self.save_results()
        return float(observed["by_metric"]["correlation"]["accuracy"])

    def _assert_pair_alignment(self, pass_summary: Dict[str, object]) -> None:
        """Every pass must evaluate the SAME pairs in the SAME order.

        This holds by construction — `_run_one_pass` re-seeds `pair_rng` from
        `random_seed` and resets `_used_samples` on entry, and `_passes_constraints`
        reads the TRUE `cog_features` rather than `cog_override`, so shuffling the
        cognition never moves the pair set. That last point is deliberate: it makes the
        permutation a null *conditional on the pair set*, which is what allows an
        observed accuracy and its null to be compared at a fixed distance threshold.
        (Contrast `analysis/data_informativeness.py`, which rebuilds its inner pairs per
        permutation because its pairing is itself the object of study.)

        We still check, because a silent misalignment would corrupt every downstream
        curve without changing any aggregate number.
        """
        if self._verdict_pairs is None:
            return
        got = pass_summary["verdict_pairs"]
        if got.shape == self._verdict_pairs.shape and np.array_equal(got, self._verdict_pairs):
            return
        raise RuntimeError(
            "LTO null permutation evaluated a different pair set than the observed pass "
            f"(observed {self._verdict_pairs.shape[0]} pairs, permutation {got.shape[0]}). "
            "The per-pair null matrix would be misaligned. The usual cause is "
            "null_shuffle_session_labels=True combined with different_sessions=True, "
            "which lets the shuffled session labels change which pairs pass the filter."
        )

    # ---------- pass machinery ----------

    def _run_one_pass(
        self,
        label: str,
        cog_override: Optional[Dict[int, np.ndarray]],
        session_override: Optional[Dict[int, int]],
        collect_results: bool,
        show_progress: bool,
    ) -> Dict[str, object]:
        """Run a single LTO sweep over all valid pairs.

        Reused by both the observed pass and each null permutation. With
        `collect_results=False` we skip per-pair row construction and map saving
        and just track the running counts — that's what makes 100 null permutations
        affordable.
        """
        all_indices = list(range(len(self.samples)))
        groups = self._group_by_contrast()
        pair_rng = random.Random(self.run_config.random_seed)
        self._used_samples = set()
        cfgs = self.correlation_configs
        primary = cfgs[0].name

        rows: List[dict] = []
        total = 0
        correct = {m: 0 for m in EVAL_METRICS}
        by_cfg = {(m, c.name): {"total": 0, "correct": 0} for m in EVAL_METRICS for c in cfgs}
        by_contrast: Dict[str, dict] = {}
        skipped_contrasts: List[str] = []

        # Per-pair verdict log, collected on EVERY pass (observed and null alike). Appending
        # a couple of ints and bools is free next to the per-pair basis+predictor refit that
        # dominates this loop, and it is what lets the whole distance curve — plus its null
        # band and the trend test — be recomputed post-hoc without re-running anything.
        v_idx1: List[int] = []
        v_idx2: List[int] = []
        v_correct: Dict[str, List[bool]] = {
            f"{m}__{c.name}": [] for m in EVAL_METRICS for c in cfgs
        }

        def _scores(ma, mb, cfg):
            """correlation + mse between two maps in cfg's space (project once)."""
            va, vb = self._project_pair(ma, mb, cfg)
            a = np.asarray(va).ravel().astype(np.float64)
            b = np.asarray(vb).ravel().astype(np.float64)
            return {"correlation": safe_corr(a, b), "mse": float(np.mean((a - b) ** 2))}

        for cname, idx_list in groups.items():
            if len(idx_list) < 2:
                continue
            # Deviation scoring needs a μ built from the N-2 training maps of this contrast;
            # with only two samples the training subset holds none of them, so the contrast
            # cannot be scored at all (rather than silently scored against a wrong μ).
            if self.score_space == "deviation" and len(idx_list) < 3:
                if show_progress:
                    print(f"\n[LTO-{label}] {cname}: skipped — deviation scoring needs >= 3 "
                          f"samples per contrast, found {len(idx_list)}")
                skipped_contrasts.append(cname)
                continue
            pairs = self._make_pairs(idx_list, session_override=session_override, rng=pair_rng)
            if self.args.max_pairs_per_contrast and len(pairs) > self.args.max_pairs_per_contrast:
                pairs = pair_rng.sample(pairs, self.args.max_pairs_per_contrast)
            if show_progress:
                print(f"\n[LTO-{label}] {cname}: {len(pairs)} pairs")
            by_contrast[cname] = {
                "n_pairs": 0,
                "by_metric": {
                    m: {"n_correct": 0,
                        "by_corr_config": {c.name: {"n_correct": 0} for c in cfgs}}
                    for m in EVAL_METRICS
                },
            }

            iterator = tqdm(pairs, desc=f"  {cname}") if show_progress else pairs
            for (i1, i2) in iterator:
                train_indices = [k for k in all_indices if k not in {i1, i2}]
                basis, predictor = self._train_on_subset(train_indices, cog_override=cog_override)
                p1 = self._predict(self.samples[i1], basis, predictor)
                p2 = self._predict(self.samples[i2], basis, predictor)
                t1 = self._apply_threshold(self._get_map(i1))
                t2 = self._apply_threshold(self._get_map(i2))

                # Scoring space. The absolute maps are what get saved (--save-maps) and what
                # the viewer shows; only the maps handed to the metrics are converted, and
                # both sides go through the same per-contrast training μ.
                if not self._score_mu_available(cname):
                    raise RuntimeError(
                        f"deviation scoring has no training mean map for contrast '{cname}' "
                        f"(pair {i1}/{i2}) — this should have been filtered out earlier"
                    )
                # Both samples of a pair come from the same contrast group, so one μ applies
                # to all four maps — which is exactly why it cancels out of the comparison.
                q1, q2 = self._to_score_space(p1, cname), self._to_score_space(p2, cname)
                u1, u2 = self._to_score_space(t1, cname), self._to_score_space(t2, cname)
                # per_cfg[name][metric] = {v11,v12,v21,v22, correct}
                per_cfg: Dict[str, Dict[str, dict]] = {}
                for cfg in cfgs:
                    s11 = _scores(q1, u1, cfg); s12 = _scores(q1, u2, cfg)
                    s21 = _scores(q2, u1, cfg); s22 = _scores(q2, u2, cfg)
                    per_cfg[cfg.name] = {}
                    for m in EVAL_METRICS:
                        v11, v12, v21, v22 = s11[m], s12[m], s21[m], s22[m]
                        ok = metric_is_better(m, v11, v12) and metric_is_better(m, v22, v21)
                        per_cfg[cfg.name][m] = {
                            "v11": v11, "v12": v12, "v21": v21, "v22": v22, "correct": ok,
                        }

                total += 1
                by_contrast[cname]["n_pairs"] += 1
                v_idx1.append(int(i1))
                v_idx2.append(int(i2))
                for m in EVAL_METRICS:
                    for cfg in cfgs:
                        v_correct[f"{m}__{cfg.name}"].append(
                            bool(per_cfg[cfg.name][m]["correct"])
                        )
                for m in EVAL_METRICS:
                    if per_cfg[primary][m]["correct"]:
                        correct[m] += 1
                        by_contrast[cname]["by_metric"][m]["n_correct"] += 1
                    for cfg in cfgs:
                        by_cfg[(m, cfg.name)]["total"] += 1
                        if per_cfg[cfg.name][m]["correct"]:
                            by_cfg[(m, cfg.name)]["correct"] += 1
                            by_contrast[cname]["by_metric"][m]["by_corr_config"][cfg.name]["n_correct"] += 1

                if collect_results:
                    cog1 = self.samples[i1].cog_features
                    cog2 = self.samples[i2].cog_features
                    if self.cog_std is not None:
                        cog_dist = float(np.linalg.norm((cog1 - cog2) / self.cog_std))
                    else:
                        cog_dist = float(np.linalg.norm(cog1 - cog2))
                    pc_corr = per_cfg[primary]["correlation"]
                    pc_mse = per_cfg[primary]["mse"]
                    row = {
                        "pair_id": total - 1,
                        "contrast": cname,
                        "is_contrast_map": self.samples[i1].is_contrast,
                        "idx1": i1,
                        "subj1": self.samples[i1].subject_id,
                        "ses1": self._session_id(i1, session_override),
                        "cog1": cog1.tolist(),
                        "idx2": i2,
                        "subj2": self.samples[i2].subject_id,
                        "ses2": self._session_id(i2, session_override),
                        "cog2": cog2.tolist(),
                        "cog_distance": cog_dist,
                        # primary correlation (legacy columns)
                        "correct": pc_corr["correct"],
                        "r11": pc_corr["v11"], "r12": pc_corr["v12"],
                        "r21": pc_corr["v21"], "r22": pc_corr["v22"],
                        # primary MSE
                        "correct_mse": pc_mse["correct"],
                        "mse11": pc_mse["v11"], "mse12": pc_mse["v12"],
                        "mse21": pc_mse["v21"], "mse22": pc_mse["v22"],
                    }
                    for cfg in cfgs:
                        cc = per_cfg[cfg.name]["correlation"]
                        mm = per_cfg[cfg.name]["mse"]
                        row[f"correct_{cfg.name}"] = cc["correct"]
                        row[f"r11_{cfg.name}"] = cc["v11"]; row[f"r12_{cfg.name}"] = cc["v12"]
                        row[f"r21_{cfg.name}"] = cc["v21"]; row[f"r22_{cfg.name}"] = cc["v22"]
                        row[f"correct_mse_{cfg.name}"] = mm["correct"]
                        row[f"mse11_{cfg.name}"] = mm["v11"]; row[f"mse12_{cfg.name}"] = mm["v12"]
                        row[f"mse21_{cfg.name}"] = mm["v21"]; row[f"mse22_{cfg.name}"] = mm["v22"]
                    rows.append(row)

                    if self.args.save_maps:
                        pdir = self.output_dir / "maps" / f"pair_{total - 1:05d}"
                        pdir.mkdir(parents=True, exist_ok=True)
                        np.save(pdir / "pred_map_1.npy", p1)
                        np.save(pdir / "pred_map_2.npy", p2)
                        np.save(pdir / "true_map_1.npy", t1)
                        np.save(pdir / "true_map_2.npy", t2)

                if show_progress:
                    tqdm.write(
                        f"  corr acc {correct['correlation']}/{total} = "
                        f"{correct['correlation']/max(1,total):.3f} | "
                        f"mse acc {correct['mse']/max(1,total):.3f}  (nominal 0.250)"
                    )

        def _by_metric(m: str) -> dict:
            out = {
                "correct_pairs": int(correct[m]),
                "total_pairs": int(total),
                "accuracy": float(correct[m] / total) if total else 0.0,
                "by_corr_config": {
                    cfg.name: {
                        "correct_pairs": int(by_cfg[(m, cfg.name)]["correct"]),
                        "total_pairs": int(by_cfg[(m, cfg.name)]["total"]),
                        "accuracy": float(by_cfg[(m, cfg.name)]["correct"] / by_cfg[(m, cfg.name)]["total"])
                        if by_cfg[(m, cfg.name)]["total"] else 0.0,
                    }
                    for cfg in cfgs
                },
                "per_contrast": {},
            }
            for cname, stats in by_contrast.items():
                n = stats["n_pairs"]
                bm = stats["by_metric"][m]
                out["per_contrast"][cname] = {
                    "n_pairs": int(n),
                    "n_correct": int(bm["n_correct"]),
                    "accuracy": float(bm["n_correct"] / n) if n else 0.0,
                    "by_corr_config": {
                        cfg.name: {
                            "n_correct": int(bm["by_corr_config"][cfg.name]["n_correct"]),
                            "accuracy": float(bm["by_corr_config"][cfg.name]["n_correct"] / n) if n else 0.0,
                        }
                        for cfg in cfgs
                    },
                }
            return out

        by_metric_out = {m: _by_metric(m) for m in EVAL_METRICS}
        return {
            "label": label,
            "total_pairs": int(total),
            "by_metric": by_metric_out,
            # legacy correlation aliases (used by the null loop's prints)
            "accuracy": by_metric_out["correlation"]["accuracy"],
            "correct_pairs": by_metric_out["correlation"]["correct_pairs"],
            "rows": rows,
            "skipped_contrasts": skipped_contrasts,
            # per-pair verdict log (see the accumulator comment above)
            "verdict_pairs": np.asarray(
                list(zip(v_idx1, v_idx2)), dtype=np.int64,
            ).reshape(-1, 2),
            "verdicts": {k: np.asarray(v, dtype=bool) for k, v in v_correct.items()},
        }

    # ---------- null permutation builders ----------

    def _make_cog_override(
        self,
        all_indices: List[int],
        contrast_groups: Dict[str, List[int]],
        rng: np.random.Generator,
    ) -> Dict[int, np.ndarray]:
        """Build idx → cog_features mapping by shuffling correspondence.

        With scope='global', every sample receives a randomly drawn cog vector.
        With scope='within_contrast', the shuffle is done independently inside
        each contrast group so the marginal distribution of cog vectors per
        contrast is preserved (any signal must come from cog↔map *correspondence*,
        not from contrast-level cog distributions).

        With scope='within_contrast_diagnosis', the shuffle is further restricted to
        samples sharing a diagnosis. This matters whenever the cohort spans diagnosis
        groups and pairs are formed ACROSS them: cognition partly encodes diagnosis, so a
        cross-group shuffle also destroys the cognition↔diagnosis association. The observed
        model can then exploit group membership while the permuted model cannot, and the
        resulting observed-minus-null gap absorbs the group confound instead of isolating
        cognition. Shuffling within diagnosis leaves group-level cognitive structure intact
        and puts only the within-group cog↔map correspondence at risk, so a surviving effect
        is cognitive rather than diagnostic. Falls back to `within_contrast` (with a warning)
        for datasets that expose no diagnosis map.
        """
        override: Dict[int, np.ndarray] = {}
        if self.args.null_shuffle_scope == "global":
            donors = np.asarray(all_indices, dtype=int).copy()
            rng.shuffle(donors)
            for tgt, src in zip(all_indices, donors):
                override[int(tgt)] = self.samples[int(src)].cog_features
            return override
        if self.args.null_shuffle_scope == "within_contrast_diagnosis":
            if not getattr(self.adapter, "_diag_by_sid", None):
                if not getattr(self, "_warned_no_diag", False):
                    print("[LTO] null_shuffle_scope='within_contrast_diagnosis' but the "
                          "adapter exposes no diagnosis map — falling back to "
                          "'within_contrast'")
                    self._warned_no_diag = True
            else:
                for _, group in contrast_groups.items():
                    buckets: Dict[object, List[int]] = {}
                    for i in group:
                        buckets.setdefault(
                            self._diagnosis(self.samples[i].subject_id), []
                        ).append(i)
                    for _, bucket in buckets.items():
                        donors = np.asarray(bucket, dtype=int).copy()
                        rng.shuffle(donors)
                        for tgt, src in zip(bucket, donors):
                            override[int(tgt)] = self.samples[int(src)].cog_features
                return override
        for _, group in contrast_groups.items():
            donors = np.asarray(group, dtype=int).copy()
            rng.shuffle(donors)
            for tgt, src in zip(group, donors):
                override[int(tgt)] = self.samples[int(src)].cog_features
        return override

    def _make_session_override(
        self, all_indices: List[int], rng: np.random.Generator,
    ) -> Dict[int, int]:
        sessions = np.asarray([self.samples[i].session_id for i in all_indices], dtype=int)
        rng.shuffle(sessions)
        return {int(idx): int(s) for idx, s in zip(all_indices, sessions)}

    def _build_null_summary(
        self,
        observed: Dict[str, object],
        null_overall: List[float],
        null_per_cfg: Dict[str, List[float]],
        null_per_contrast: Dict[str, List[float]],
    ) -> Dict[str, object]:
        out = {
            "settings": {
                "run_null": True,
                "n_null_permutations": int(self.args.n_null_permutations),
                "null_seed": int(self.args.null_seed),
                "null_shuffle_scope": self.args.null_shuffle_scope,
                "null_shuffle_session_labels": bool(self.args.null_shuffle_session_labels),
                "store_null_distribution": bool(self.args.store_null_distribution),
            },
            "overall": _summarise_null(observed["accuracy"], null_overall),
            "by_corr_config": {
                cfg_name: _summarise_null(observed["by_corr_config"][cfg_name]["accuracy"], values)
                for cfg_name, values in null_per_cfg.items()
            },
            "per_contrast": {
                cname: _summarise_null(
                    observed["per_contrast"].get(cname, {}).get("accuracy", 0.0), values,
                )
                for cname, values in null_per_contrast.items()
            },
        }
        if self.args.store_null_distribution:
            out["distributions"] = {
                "overall": [float(v) for v in null_overall],
                "by_corr_config": {k: [float(v) for v in vs] for k, vs in null_per_cfg.items()},
                "per_contrast": {k: [float(v) for v in vs] for k, vs in null_per_contrast.items()},
            }
        return out

    def save_results(self) -> None:
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "lto_results_detailed.csv", index=False)

        total = len(self.results)
        correct = int(sum(r["correct"] for r in self.results))
        accuracy = correct / total if total else 0.0
        per_contrast: Dict[str, dict] = {}
        if total:
            for cname, group in df.groupby("contrast"):
                n = len(group)
                k = int(group["correct"].sum())
                cfg_stats = {}
                for cfg in self.correlation_configs:
                    col = f"correct_{cfg.name}"
                    if col in group.columns:
                        kc = int(group[col].sum())
                        cfg_stats[cfg.name] = {"n_correct": kc, "accuracy": kc / n}
                per_contrast[cname] = {
                    "n_pairs": int(n),
                    "n_correct": k,
                    "accuracy": k / n,
                    "by_corr_config": cfg_stats,
                }

        # Per-metric blocks (correlation + mse) built from the observed pass +
        # the per-metric null distributions. Each block mirrors the legacy shape.
        metrics_blocks: Dict[str, dict] = {}
        obs_by_metric = self._observed_summary.get("by_metric", {}) if self._observed_summary else {}
        for m in EVAL_METRICS:
            block = obs_by_metric.get(m)
            if block is None:
                continue
            metrics_blocks[m] = {
                "overall": {
                    "total_pairs": block["total_pairs"],
                    "correct_pairs": block["correct_pairs"],
                    "accuracy": block["accuracy"],
                },
                "by_corr_config": block["by_corr_config"],
                "per_contrast": block["per_contrast"],
                "null_model": (self.null_metrics.get(m) if self.null_metrics else None),
            }

        summary = {
            "evaluation_method": "Leave-Two-Out Classification (correlation + MSE)",
            "reference": "Walters et al. 2022",
            "chance_level": 0.25,
            "eval_metrics": list(EVAL_METRICS),
            "timestamp": datetime.now().isoformat(),
            "correlation_configs": [c.to_dict() for c in self.correlation_configs],
            # Space the metrics above were computed in — "map" or "deviation" (per-contrast
            # training mean subtracted from prediction and ground truth alike). Any saved
            # per-pair maps stay absolute regardless.
            "score_space": self.score_space,
            "skipped_contrasts": self._observed_summary.get("skipped_contrasts", []),
            "constraints": {
                "max_pairs_per_contrast": self.args.max_pairs_per_contrast,
                "different_subjects": self.args.different_subjects,
                "different_sessions": self.args.different_sessions,
                "min_cog_distance": self.args.min_cog_distance,
                "unique_samples": self.args.unique_samples,
                "same_diagnosis": self.args.same_diagnosis,
                "p_threshold": self.args.p_threshold,
                "p_threshold_mode": self.args.p_threshold_mode,
                "session_group_a": self.args.session_group_a,
                "session_group_b": self.args.session_group_b,
                "train_atlas_cfg": self.args.train_atlas_cfg.to_dict() if self.args.train_atlas_cfg else None,
            },
            # Legacy top-level correlation block (back-compat for old viewers).
            "overall": {
                "total_pairs": total,
                "correct_pairs": correct,
                "accuracy": accuracy,
                "by_corr_config": {
                    cfg.name: {
                        "correct_pairs": int(df[f"correct_{cfg.name}"].sum())
                        if (f"correct_{cfg.name}" in df.columns) else correct,
                        "accuracy": float(df[f"correct_{cfg.name}"].sum() / total)
                        if (f"correct_{cfg.name}" in df.columns and total) else accuracy,
                    }
                    for cfg in self.correlation_configs
                },
            },
            "per_contrast": per_contrast,
            # New: both metrics, toggled in the viewer.
            "metrics": metrics_blocks,
            "model_config": {
                "basis_type": self.run_config.basis_config.basis_type,
                "n_components": int(self.run_config.basis_config.n_components),
                "predictor_type": self.run_config.predictor_config.model_type,
                "loss_mode": self.run_config.predictor_config.loss_mode,
                "adapter_name": self.run_config.data_config.adapter_name,
            },
            # Legacy: correlation null block at top level (back-compat).
            "null_model": (self.null_metrics.get("correlation") if self.null_metrics else None),
        }
        save_json(summary, self.output_dir / "lto_results_summary.json")
        self._save_score_mean_maps()

        if self.args.store_null_verdicts:
            self._save_null_verdicts()

        print(f"\n[LTO] DONE — corr {correct}/{total} = {accuracy:.3f}")
        print(f"[LTO]        scored in {self.score_space} space")
        if "mse" in metrics_blocks:
            print(f"[LTO]        mse acc = {metrics_blocks['mse']['overall']['accuracy']:.3f}")
        if self.null_metrics and self.null_metrics.get("correlation"):
            ov = self.null_metrics["correlation"]["overall"]
            print(
                f"[LTO] corr null mean={ov['mean']:.3f}  "
                f"95%CI=[{ov['ci95_low']:.3f}, {ov['ci95_high']:.3f}]  "
                f"empirical p={ov['empirical_p_one_sided']:.4g}"
            )

    def _save_score_mean_maps(self) -> None:
        """Write `score_mean_maps.pkl` for a deviation-scored eval — μ for the viewer.

        Scoring itself uses each pair's own N-2 training mean (refit in `_train_on_subset`,
        never the held-out maps). Persisting 1000 nearly identical μ would be wasteful, so
        what lands here is the per-contrast mean over ALL samples — it differs from any pair's
        μ by O(1/N) and exists so the app can *display* the same deviation space the metrics
        were computed in. Same layout as MeanMapOffset.state(), so the app's loader reads
        this and the training-offset file with one code path.
        """
        if self.score_space != "deviation":
            return
        groups = self._group_by_contrast()
        mean_by_contrast = {
            cname: np.mean([self._get_map(i) for i in idx_list], axis=0)
            for cname, idx_list in groups.items() if idx_list
        }
        save_pickle(
            {"enabled": True, "mean_by_contrast": mean_by_contrast, "source": "all_samples"},
            self.output_dir / "score_mean_maps.pkl",
        )

    def _save_null_verdicts(self) -> None:
        """Write `null_verdicts.npz`: one row per evaluated pair + the (B, n_pairs) null matrix.

        This is the artifact `cog2bold.analysis.lto_curve_calibration` consumes. It stores
        `cog1`/`cog2` and not just the precomputed `cog_distance`, so the distance axis can be
        re-standardised afterwards — e.g. onto a cohort-wide `cog_std` so that per-group
        regimes become comparable — without re-running the evaluation.
        """
        if self._verdict_pairs is None or not len(self._verdict_pairs):
            print("[LTO] store_null_verdicts: no pairs evaluated, nothing written")
            return
        i1 = self._verdict_pairs[:, 0]
        i2 = self._verdict_pairs[:, 1]

        # NOTE: do NOT coerce the dtype here. cog_features are float32, and the detailed CSV
        # computes `norm((c1 - c2) / cog_std)` with that same float32 subtraction before
        # promoting. Integer-valued test scores produce many EXACT distance ties, so casting
        # to float64 first changes low bits and can flip which pairs clear a threshold.
        # Same hazard documented in analysis/data_informativeness._build_inner_pairs.
        c1 = np.asarray([self.samples[int(i)].cog_features for i in i1])
        c2 = np.asarray([self.samples[int(i)].cog_features for i in i2])
        cog_distance = np.linalg.norm((c1 - c2) / self.cog_std, axis=-1)

        n_perm = 0
        if self._null_verdicts:
            n_perm = int(next(iter(self._null_verdicts.values())).shape[0])

        payload: Dict[str, np.ndarray] = {
            "idx1": i1.astype(np.int64),
            "idx2": i2.astype(np.int64),
            "subj1": np.asarray([self.samples[int(i)].subject_id for i in i1], dtype=np.int64),
            "subj2": np.asarray([self.samples[int(i)].subject_id for i in i2], dtype=np.int64),
            "ses1": np.asarray([self.samples[int(i)].session_id for i in i1], dtype=np.int64),
            "ses2": np.asarray([self.samples[int(i)].session_id for i in i2], dtype=np.int64),
            "cog1": c1,
            "cog2": c2,
            "cog_distance": np.asarray(cog_distance, dtype=np.float64),
            "cog_std": np.asarray(self.cog_std, dtype=np.float64),
            "contrast": np.asarray([self._contrast_name_of(int(i)) for i in i1]),
        }
        for key, arr in self._observed_verdicts.items():
            payload[f"obs__{key}"] = arr
        for key, arr in self._null_verdicts.items():
            payload[f"null__{key}"] = arr

        primary = self.correlation_configs[0].name
        payload["meta"] = np.asarray(json.dumps({
            "n_pairs": int(len(i1)),
            "n_permutations": n_perm,
            "null_seed": int(self.args.null_seed),
            "null_shuffle_scope": self.args.null_shuffle_scope,
            "null_conditional_on_pair_set": True,
            "p_floor": 1.0 / (1.0 + n_perm) if n_perm else None,
            "primary_corr_config": primary,
            "corr_configs": [c.name for c in self.correlation_configs],
            "eval_metrics": list(EVAL_METRICS),
            "max_pairs_per_contrast": self.args.max_pairs_per_contrast,
            "min_cog_distance": self.args.min_cog_distance,
            "unique_samples": bool(self.args.unique_samples),
            "same_diagnosis": bool(self.args.same_diagnosis),
            "adapter_name": self.run_config.data_config.adapter_name,
            "adapter_kwargs": self.run_config.data_config.adapter_kwargs or {},
            "cog_base_cols": list(self.run_config.data_config.cog_base_cols),
            "cog_feature_names": [
                str(n) for n in getattr(self.adapter, "cog_feature_names",
                                        self.run_config.data_config.cog_base_cols)
            ],
        }))

        out = self.output_dir / "null_verdicts.npz"
        np.savez_compressed(out, **payload)
        print(f"[LTO] wrote {out}  ({len(i1)} pairs x {n_perm} permutations)")


def run_lto_from_run_dir(args: LTOArgs) -> float:
    """Convenience: load RunConfig from `args.run_dir`, then run LTO."""
    cfg_path = Path(args.run_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json in {args.run_dir}")
    run_config = RunConfig.load(cfg_path)
    classifier = LeaveTwoOutClassifier(run_config, args)
    classifier.load_dataset()
    return classifier.run()
