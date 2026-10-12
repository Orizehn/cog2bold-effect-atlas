"""Cog2BOLD command-line interface.

Subcommands:
    train     run the full training pipeline
    lto       run leave-two-out classification on a trained run
    info      run data-informativeness analysis on a trained run
    effects   effect atlas: which features move which brain regions (blobs + peak tables)
    export-run  write a self-contained, shareable bundle of a run (for the public app)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from cog2bold import (
    BasisConfig,
    CorrelationConfig,
    DataConfig,
    PredictorConfig,
    RunConfig,
)
from cog2bold.datasets.registry import create_adapter, list_adapters
from cog2bold.workspace import runs_root, split_adapter


# ---------- helpers ----------


def _parse_correlation_configs(spec: Optional[str]) -> Optional[List[CorrelationConfig]]:
    """Parse a JSON string or path -> list of CorrelationConfig."""
    if not spec:
        return None
    if Path(spec).exists():
        data = json.loads(Path(spec).read_text())
    else:
        data = json.loads(spec)
    return [CorrelationConfig.from_dict(d) for d in data]


def _parse_train_atlas_cfg(spec: Optional[str]) -> Optional[CorrelationConfig]:
    if not spec:
        return None
    data = json.loads(spec)
    return CorrelationConfig.from_dict(data)


def _hidden_dims(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _make_run_dir(args) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    # <dataset>_<predictor>_<basis>[_<sup>_<unsup>]_<n_comp>, nested under <dataset>/<task>/
    parts = [args.dataset, args.predictor]
    if args.basis == "hybrid":
        parts += ["hybrid", args.hybrid_supervised, args.hybrid_unsupervised]
    else:
        parts.append(args.basis)
    parts.append(str(args.n_components))
    dataset, task = split_adapter(args.dataset)
    return runs_root() / dataset / task / "_".join(parts)


# ---------- commands ----------


def _parse_single_contrast(val):
    """--single-contrast: comma-separated names → list; one name → str; None → None.

    Vector-string form ("[1 0 0 -1]") is single-only and left untouched.
    """
    if not val:
        return None
    if "," in val and not val.strip().startswith("["):
        return [s.strip() for s in val.split(",") if s.strip()]
    return val


def cmd_train(args: argparse.Namespace) -> int:
    from cog2bold.pipelines.training import TrainingPipeline

    data_root = Path(args.data_root).expanduser().resolve()
    run_dir = _make_run_dir(args)

    cog_cols = args.cog_features.split(",") if args.cog_features else []
    if not cog_cols:
        raise SystemExit("--cog-features is required (comma-separated column names)")

    adapter_kwargs: dict = {}
    if getattr(args, "adapter_kwargs", None):
        import json as _json
        adapter_kwargs = _json.loads(args.adapter_kwargs)
        if not isinstance(adapter_kwargs, dict):
            raise SystemExit("--adapter-kwargs must be a JSON object")

    adapter = create_adapter(
        args.dataset, data_root=data_root, cog_base_cols=cog_cols, **adapter_kwargs)

    extra_basis: dict = {}
    if args.basis == "atlas":
        extra_basis["atlas_name"] = args.atlas_name
    if args.basis == "ridgecol":
        extra_basis["include_quadratic"] = args.ridgecol_quadratic
        extra_basis["include_interactions"] = args.ridgecol_interactions
    if args.basis == "hybrid":
        extra_basis["supervised_method"] = args.hybrid_supervised
        extra_basis["unsupervised_method"] = args.hybrid_unsupervised
        extra_basis["normalize_basis_vectors"] = args.hybrid_normalize

    basis_config = BasisConfig(
        basis_type=args.basis,
        n_components=args.n_components,
        random_seed=args.seed,
        extra_params=extra_basis,
    )
    predictor_config = PredictorConfig(
        model_type=args.predictor,
        output_dim=args.n_components,
        loss_mode=args.loss_mode,
        map_loss_type=args.map_loss_type,
        hidden_dims=_hidden_dims(args.hidden) if args.hidden else None,
        activation=args.activation,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        random_seed=args.seed,
        cog_only=args.cog_only,
    )
    data_config = DataConfig(
        adapter_name=args.dataset,
        cog_base_cols=cog_cols,
        adapter_kwargs=adapter_kwargs,
        stim_encoding=args.stim_encoding,
        contrast_pairs=[tuple(c.split(",")) for c in (args.contrast_pairs_train or [])] or None,
        subject_session_filter=Path(args.filter) if args.filter else None,
        single_contrast=_parse_single_contrast(args.single_contrast),
        force_recompute=args.force_recompute,
    )
    run_config = RunConfig(
        data_root=data_root,
        run_dir=run_dir,
        basis_config=basis_config,
        predictor_config=predictor_config,
        data_config=data_config,
        train_fraction=args.train_frac,
        random_seed=args.seed,
        contrast_pairs=[tuple(c.split(",")) for c in (args.contrasts or [])] or None,
        eval_correlation_configs=_parse_correlation_configs(args.eval_correlation_configs),
        train_atlas_cfg=_parse_train_atlas_cfg(args.train_atlas_cfg),
    )
    print("=" * 70)
    print(f"  dataset:      {args.dataset}")
    print(f"  basis:        {args.basis} (k={args.n_components})")
    print(f"  predictor:    {args.predictor} (loss={args.loss_mode}/{args.map_loss_type})")
    print(f"  run_dir:      {run_dir}")
    print("=" * 70)

    TrainingPipeline(run_config, adapter).run()
    return 0


def cmd_lto(args: argparse.Namespace) -> int:
    from cog2bold.pipelines.lto import LTOArgs, run_lto_from_run_dir

    lto_args = LTOArgs.from_dict(
        {
            "run_dir": Path(args.run_dir),
            "save_maps": args.save_maps,
            "output_dir": Path(args.output_dir) if args.output_dir else None,
            "max_pairs_per_contrast": args.max_pairs_per_contrast,
            "different_subjects": args.different_subjects,
            "different_sessions": args.different_sessions,
            "min_cog_distance": args.min_cog_distance,
            "unique_samples": args.unique_samples,
            "same_diagnosis": args.same_diagnosis,
            "p_threshold": args.p_threshold,
            "p_threshold_mode": args.p_threshold_mode,
            "session_group_a": args.session_group_a,
            "session_group_b": args.session_group_b,
            "correlation_configs": _parse_correlation_configs(args.correlation_configs),
            "train_atlas_cfg": _parse_train_atlas_cfg(args.train_atlas_cfg),
            "score_space": args.score_space,
            "run_null": args.run_null,
            "n_null_permutations": args.n_null_permutations,
            "null_seed": args.null_seed,
            "null_shuffle_scope": args.null_shuffle_scope,
            "null_shuffle_session_labels": args.null_shuffle_session_labels,
            "store_null_distribution": args.store_null_distribution,
            "store_null_verdicts": args.store_null_verdicts,
        }
    )
    run_lto_from_run_dir(lto_args)
    return 0


def cmd_loo(args: argparse.Namespace) -> int:
    from cog2bold.pipelines.loo import LOOArgs, run_loo_from_run_dir

    loo_args = LOOArgs.from_dict(
        {
            "run_dir": Path(args.run_dir),
            "output_dir": Path(args.output_dir) if args.output_dir else None,
            "save_maps": args.save_maps,
            "leave_out_level": args.leave_out_level,
            "max_units": args.max_units,
            "correlation_configs": _parse_correlation_configs(args.correlation_configs),
            "train_atlas_cfg": _parse_train_atlas_cfg(args.train_atlas_cfg),
            "p_threshold": args.p_threshold,
            "p_threshold_mode": args.p_threshold_mode,
            "seed": args.seed,
        }
    )
    run_loo_from_run_dir(loo_args)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from cog2bold.analysis import DataInformativenessArgs, run_data_informativeness

    info_args = DataInformativenessArgs(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        max_super_pairs_per_contrast=args.max_super_pairs_per_contrast,
        pairing_modes=args.pairing_modes.split(","),
        min_cog_distance=args.min_cog_distance,
        unique_samples=args.unique_samples,
        save_maps=args.save_maps,
        correlation_configs=_parse_correlation_configs(args.correlation_configs),
        seed=args.seed,
        run_null=args.run_null,
        n_null_permutations=args.n_null_permutations,
        null_seed=args.null_seed,
        null_shuffle_scope=args.null_shuffle_scope,
        store_null_distribution=args.store_null_distribution,
    )
    run_data_informativeness(info_args)
    return 0


def cmd_effects(args: argparse.Namespace) -> int:
    from cog2bold.analysis.effect_atlas import ScanParams, run_scan, write_report

    def _list(v):
        return tuple(x.strip() for x in v.split(",") if x.strip()) if v else None

    params = ScanParams(
        combos=_list(args.combos),
        features=_list(args.features),
        conditions=_list(args.conditions),
        k_sd=args.k_sd,
        archetypes_k=args.archetypes_k,
        mode=args.mode,
        percentile=args.percentile,
        abs_threshold=args.abs_threshold,
        n_boot=args.n_boot,
        z_thr=args.z_thr,
        fdr_q=args.fdr_q,
        min_cluster_voxels=args.min_cluster_voxels,
        connectivity=args.connectivity,
        min_peak_distance_mm=args.min_peak_distance_mm,
        max_subpeaks=args.max_subpeaks,
        seed=args.seed,
    )
    step = max(1, args.n_boot // 10)

    def progress(stage: str, frac: float) -> None:
        if stage.startswith("bootstrap"):
            b = int(stage.split()[1].split("/")[0])
            if b % step:
                return
        print(f"  [{100 * frac:5.1f}%] {stage}")

    res = run_scan(Path(args.run_dir), params, progress=progress)
    out = write_report(res, out_dir=Path(args.out_dir) if args.out_dir else None,
                       max_detailed=args.max_detailed, save_maps=not args.no_maps)
    print(res.summary().drop(columns=["description", "threshold"]).head(15).to_string(index=False))
    print(f"\nreport: {out / 'report.html'}")
    return 0


def cmd_export_run(args: argparse.Namespace) -> int:
    from cog2bold.analysis.effect_atlas.bundle import export_bundle

    out = export_bundle(Path(args.run_dir), Path(args.out_dir), attribution=args.attribution,
                        include_maps=True if args.include_maps else None,
                        data_root=Path(args.data_root) if args.data_root else None)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle written to {out} ({size / 1e6:.1f} MB)")
    return 0


# ---------- argument plumbing ----------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cog2bold")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- train ----
    p = sub.add_parser("train", help="Train + evaluate")
    p.add_argument("--data-root", default=".")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--dataset", default="dfg", choices=sorted(list_adapters()))
    p.add_argument("--cog-features", required=True, help="comma-separated column names")
    p.add_argument("--filter", default=None, help="path to a subject-session filter JSON")
    p.add_argument(
        "--single-contrast", default=None,
        help='Keep one stimulus/contrast by name (e.g. "Words" or "Words>Checkerboard"), '
             'or a comma-separated subset (e.g. "Words>Checkerboard,Words"). '
             'Pair with --stim-encoding contrast to keep both individuals and contrasts.',
    )
    p.add_argument(
        "--adapter-kwargs", default=None,
        help='JSON dict of extra adapter args, persisted to config.json so eval '
             'rebuilds the adapter identically. ERA812 example: '
             '\'{"include_groups": ["Healthy"], "include_group_onehot": false}\'',
    )
    p.add_argument("--force-recompute", action="store_true")

    p.add_argument("--basis", default="pca",
                   choices=["identity", "pca", "ica", "atlas", "random",
                            "ridgecol", "cca", "pls", "hybrid"])
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--atlas-name", default="harvard-oxford")

    p.add_argument("--ridgecol-quadratic", action="store_true")
    p.add_argument("--ridgecol-interactions", action="store_true")

    p.add_argument("--hybrid-supervised", default="ridgecol",
                   choices=["ridgecol", "cca", "pls"])
    p.add_argument("--hybrid-unsupervised", default="pca",
                   choices=["pca", "ica", "atlas", "random"])
    p.add_argument("--hybrid-normalize", action="store_true")

    p.add_argument("--predictor", default="mlp", choices=["linear", "mlp"])
    p.add_argument("--loss-mode", default="map", choices=["coefficient", "map"])
    p.add_argument("--map-loss-type", default="l2", choices=["l2", "l1", "correlation"])
    p.add_argument("--hidden", default="64,32")
    p.add_argument("--activation", default="relu", choices=["relu", "gelu", "tanh"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)

    p.add_argument("--cog-only", action="store_true")
    p.add_argument("--stim-encoding", default="onehot", choices=["onehot", "contrast", "contrast_only"])
    p.add_argument("--contrast-pairs-train", nargs="+", default=None)
    p.add_argument("--contrasts", nargs="+", default=None,
                   help="evaluation contrast pairs (e.g. 'audio,visual')")

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--eval-correlation-configs", default=None,
                   help="JSON literal or file path: list of CorrelationConfig dicts")
    p.add_argument("--train-atlas-cfg", default=None,
                   help="JSON literal: a single CorrelationConfig dict")
    p.set_defaults(func=cmd_train)

    # ---- lto ----
    p = sub.add_parser("lto", help="Leave-two-out classification")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save-maps", action="store_true")
    p.add_argument("--max-pairs-per-contrast", type=int, default=None)
    p.add_argument("--different-subjects", action="store_true")
    p.add_argument("--different-sessions", action="store_true")
    p.add_argument("--min-cog-distance", type=float, default=None)
    p.add_argument("--unique-samples", action="store_true")
    p.add_argument("--same-diagnosis", action="store_true",
                   help="only pair two samples from the SAME diagnosis group")
    p.add_argument("--p-threshold", type=float, default=None)
    p.add_argument("--p-threshold-mode", default="zero", choices=["zero", "attenuate"])
    p.add_argument("--session-group-a", type=int, nargs="+", default=None)
    p.add_argument("--session-group-b", type=int, nargs="+", default=None)
    p.add_argument("--correlation-configs", default=None)
    p.add_argument("--train-atlas-cfg", default=None)
    p.add_argument("--score-space", default="map", choices=["map", "deviation"],
                   help="space the metrics are computed in: 'map' (absolute maps, default) or "
                        "'deviation' (subtract the per-contrast mean of the N-2 training maps "
                        "from prediction and ground truth alike, so the shared component that "
                        "dominates absolute maps cancels out). Independent of the model's own "
                        "mean_map_offset; any run can be scored either way")
    # ----- empirical null model -----
    p.add_argument("--run-null", action="store_true",
                   help="run n_null_permutations passes with shuffled cog<->map "
                        "correspondence and report empirical p-value")
    p.add_argument("--n-null-permutations", type=int, default=0)
    p.add_argument("--null-seed", type=int, default=1)
    p.add_argument("--null-shuffle-scope", default="within_contrast",
                   choices=["within_contrast", "global", "within_contrast_diagnosis"],
                   help="within_contrast_diagnosis restricts the cog<->map shuffle to samples "
                        "sharing a diagnosis; use it when the cohort spans groups and pairs "
                        "are formed across them, so the null does not also destroy the "
                        "cognition<->diagnosis association")
    p.add_argument("--null-shuffle-session-labels", action="store_true")
    p.add_argument("--no-store-null-distribution", dest="store_null_distribution",
                   action="store_false", default=True)
    p.add_argument("--store-null-verdicts", action="store_true",
                   help="write null_verdicts.npz (per-pair verdicts for the observed pass "
                        "+ a (B, n_pairs) null matrix) so accuracy-vs-cognitive-distance "
                        "curves, their per-point null bands and the trend test can be "
                        "computed post-hoc — see cog2bold/analysis/lto_curve_calibration.py")
    p.set_defaults(func=cmd_lto)

    # ---- loo ----
    p = sub.add_parser("loo", help="Leave-one-out: prediction vs population-mean map")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save-maps", action="store_true")
    p.add_argument("--leave-out-level", default="subject_session",
                   choices=["subject_session", "subject"],
                   help="hold out one (subject,session) sample-key (default) or a whole subject")
    p.add_argument("--max-units", type=int, default=None,
                   help="cap number of held-out folds (None = all)")
    p.add_argument("--correlation-configs", default=None)
    p.add_argument("--train-atlas-cfg", default=None)
    p.add_argument("--p-threshold", type=float, default=None)
    p.add_argument("--p-threshold-mode", default="zero", choices=["zero", "attenuate"])
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_loo)

    # ---- info ----
    p = sub.add_parser("info", help="Data-informativeness analysis")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-super-pairs-per-contrast", type=int, default=100)
    p.add_argument("--pairing-modes", default="nearest,furthest,random")
    p.add_argument("--min-cog-distance", type=float, default=None)
    p.add_argument("--unique-samples", action="store_true")
    p.add_argument("--save-maps", action="store_true")
    p.add_argument("--correlation-configs", default=None)
    p.add_argument("--seed", type=int, default=0)
    # ----- empirical null model -----
    p.add_argument("--run-null", action="store_true",
                   help="re-run analysis with shuffled cog<->map correspondence "
                        "and report empirical p-values per pairing mode")
    p.add_argument("--n-null-permutations", type=int, default=0)
    p.add_argument("--null-seed", type=int, default=1)
    p.add_argument("--null-shuffle-scope", default="within_contrast",
                   choices=["within_contrast", "global", "within_contrast_session"])
    p.add_argument("--no-store-null-distribution",
                   dest="store_null_distribution", action="store_false", default=True)
    p.set_defaults(func=cmd_info)

    # ---- effects ----
    p = sub.add_parser("effects", help="Effect atlas: feature -> brain blobs, peak tables, report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", default=None, help="default: <run_dir>/effect_atlas_<mode>_<ts>")
    p.add_argument("--combos", default="single,pairwise,archetype",
                   help="comma-separated subset of single,pairwise,archetype")
    p.add_argument("--features", default=None, help="restrict to these features (comma-separated)")
    p.add_argument("--conditions", default=None, help="restrict to these conditions/contrasts")
    p.add_argument("--k-sd", type=float, default=1.0, help="high/low = mean ± k·SD")
    p.add_argument("--archetypes-k", type=int, default=None, help="default: chosen by silhouette")
    p.add_argument("--mode", choices=["effect", "bootstrap"], default="effect")
    p.add_argument("--percentile", type=float, default=99.0,
                   help="effect mode: pooled |Δ| percentile threshold")
    p.add_argument("--abs-threshold", type=float, default=None,
                   help="effect mode: absolute |Δ| threshold (overrides --percentile)")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--z-thr", type=float, default=3.1, help="bootstrap mode: voxel |z| threshold")
    p.add_argument("--fdr-q", type=float, default=None, help="bootstrap mode: BH-FDR q (overrides --z-thr)")
    p.add_argument("--min-cluster-voxels", type=int, default=10)
    p.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26)
    p.add_argument("--min-peak-distance-mm", type=float, default=8.0)
    p.add_argument("--max-subpeaks", type=int, default=2)
    p.add_argument("--max-detailed", type=int, default=30,
                   help="combinations with figures in the HTML report")
    p.add_argument("--no-maps", action="store_true", help="don't write NIfTI maps")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_effects)

    # ---- export-run ----
    p = sub.add_parser("export-run", help="Self-contained shareable bundle of a run (no dataset needed)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--attribution", default="", help="data citation / license shown in the app")
    p.add_argument("--include-maps", action="store_true",
                   help="also ship training target maps (default: only when map loss needs them)")
    p.add_argument("--data-root", default=None, help="dataset location, if it moved")
    p.set_defaults(func=cmd_export_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
