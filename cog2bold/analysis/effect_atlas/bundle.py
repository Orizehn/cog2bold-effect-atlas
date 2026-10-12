"""Self-contained run bundles: everything the effect atlas needs, nothing it doesn't.

A trained run normally needs its dataset to load (the adapter reads the maps folder and the
behavioural CSV) and carries a large ``predictions.pkl`` with every subject's maps, scores and
IDs. A *bundle* is the shareable alternative used for public deployment::

    bundle.json              format version, provenance, feature/label names, attribution
    config.json              the run config, with local paths removed
    basis.pkl, predictor.pkl the fitted model (unchanged)
    mask.nii.gz              the model's brain mask (voxel space of every map)
    train.npz                the training slice the app uses, subject IDs anonymised;
                             target maps only when a map-loss bootstrap needs them
    mean_maps.npz            per-contrast mean maps (mean-map-offset runs only)
    metrics.json             held-out metrics
    lto_eval_*/lto_results_summary.json   LTO accuracies (+ null), sanitised

``LoadedRun.from_dir`` loads a bundle transparently — no dataset, no workspace cache.

    cog2bold export-run --run-dir <run_dir> --out-dir bundles/<name>
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from cog2bold.io import load_json, save_json

BUNDLE_VERSION = 1
BUNDLE_FILE = "bundle.json"

# predictions.pkl keys the app needs (see model_io._TRAIN_KEYS); maps are added on demand
_KEEP = ("cog_features_train", "stim_encodings_train", "coeffs_gt_train", "stim_ids_train",
         "is_contrast_train", "contrast_names_train")
# lto_results_summary.json keys read by LoadedRun.validity_summary (the rest may hold paths)
_LTO_KEYS = ("evaluation_method", "chance_level", "timestamp", "constraints", "overall",
             "null_model", "score_space")


def is_bundle(path) -> bool:
    return (Path(path) / BUNDLE_FILE).exists()


def export_bundle(run_dir, out_dir, attribution: str = "", include_maps: Optional[bool] = None,
                  data_root: Optional[Path] = None) -> Path:
    """Write a bundle of `run_dir` to `out_dir`. Needs the dataset once (for the mask)."""
    from cog2bold.analysis.effect_atlas.model_io import LoadedRun

    run_dir, out = Path(run_dir), Path(out_dir)
    run = LoadedRun.from_dir(run_dir, data_root=data_root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run_dir / "config.json").read_text())
    cfg["data_root"] = "."
    cfg["run_dir"] = "."
    cfg["data_config"]["subject_session_filter"] = None
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in ("basis.pkl", "predictor.pkl"):
        shutil.copy2(run_dir / f, out / f)
    run.mask_img.to_filename(out / "mask.nii.gz")

    if include_maps is None:  # only a map-loss refit needs the target maps
        include_maps = run.config.predictor_config.loss_mode == "map"
    t = run.train
    _, anon = np.unique(t["subject_ids_train"], return_inverse=True)
    arrays = {k: np.asarray(t[k]) for k in _KEEP if k in t}
    arrays["contrast_names_train"] = arrays["contrast_names_train"].astype(str)
    arrays["subject_ids_train"] = anon.astype(np.int32)
    if include_maps:
        arrays["maps_gt_train"] = np.asarray(t["maps_gt_train"], dtype=np.float32)
    np.savez_compressed(out / "train.npz", **arrays)

    if run.mean_offset.enabled:
        np.savez_compressed(out / "mean_maps.npz", **{
            k: np.asarray(v, dtype=np.float32) for k, v in run.mean_offset.mean_by_contrast.items()})

    if (run_dir / "metrics.json").exists():
        shutil.copy2(run_dir / "metrics.json", out / "metrics.json")
    for s in run_dir.glob("lto_eval_*/lto_results_summary.json"):
        d = out / s.parent.name
        d.mkdir()
        full = load_json(s)
        save_json({k: full[k] for k in _LTO_KEYS if k in full}, d / s.name)
        st = s.stat()
        os.utime(d / s.name, (st.st_atime, st.st_mtime))  # validity_summary orders by mtime

    dc = run.config.data_config
    save_json({
        "bundle_version": BUNDLE_VERSION,
        "run_name": run_dir.name,
        "title": run.title,
        "adapter_name": dc.adapter_name,
        "feature_names": run.feature_names,
        "global_labels": run.global_labels,
        "n_train_subjects": int(len(np.unique(anon))),
        "includes_target_maps": bool(include_maps),
        "exported": datetime.now().isoformat(timespec="seconds"),
        "attribution": attribution,
    }, out / BUNDLE_FILE)
    return out


def load_bundle(bundle_dir):
    """Rebuild a `LoadedRun` from a bundle directory (no dataset needed)."""
    from nilearn import image
    from nilearn.maskers import NiftiMasker

    from cog2bold.analysis.effect_atlas.model_io import LoadedRun
    from cog2bold.bases.base import BasisTransform
    from cog2bold.config import RunConfig
    from cog2bold.pipelines.mean_offset import MeanMapOffset
    from cog2bold.predictors.base import CoefficientPredictor

    b = Path(bundle_dir)
    meta = load_json(b / BUNDLE_FILE)
    if meta.get("bundle_version", 0) > BUNDLE_VERSION:
        raise ValueError(f"bundle version {meta['bundle_version']} is newer than this code")
    masker = NiftiMasker(mask_img=image.load_img(str(b / "mask.nii.gz"))).fit()
    with np.load(b / "train.npz", allow_pickle=False) as z:
        train = {k: z[k] for k in z.files}
    offset = MeanMapOffset(False)
    if (b / "mean_maps.npz").exists():
        with np.load(b / "mean_maps.npz", allow_pickle=False) as z:
            offset.load_state({"enabled": True, "mean_by_contrast": {k: z[k] for k in z.files}})
    run = LoadedRun(
        run_dir=b,
        config=RunConfig.load(b / "config.json"),
        basis=BasisTransform.load(b / "basis.pkl"),
        predictor=CoefficientPredictor.load(b / "predictor.pkl"),
        masker=masker,
        feature_names=list(meta["feature_names"]),
        global_labels=list(meta["global_labels"]),
        train=train,
        mean_offset=offset,
        metrics=load_json(b / "metrics.json") if (b / "metrics.json").exists() else {},
    )
    run.bundle_meta = meta
    return run
