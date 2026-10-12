"""Shared helpers for the streamlit visualisation apps.

Extracted from `app_basis_compare.py`, `compare_activation_maps.py`, and
`app_data_informativeness.py` in the legacy code (the same helpers were
copy-pasted across all three).
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from nilearn import image
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map
from scipy.stats import binomtest

from cog2bold.bases.base import BasisTransform
from cog2bold.config import RunConfig
from cog2bold.datasets import create_adapter
from cog2bold.io import load_json, load_pickle
from cog2bold.predictors.base import CoefficientPredictor
from cog2bold.utils.cog_support import CogSupport


# ---------- default selection (pre-selected in every app's sidebar) ----------

# Path relative to <workspace>/runs/. Override per machine with $COG2BOLD_DEFAULT_RUN.
DEFAULT_RUN = os.environ.get(
    "COG2BOLD_DEFAULT_RUN",
    "ds000030/pamret/ds000030_pamret_linear_identity_0_ridge_corr_gt_ctrl_meanoff",
)
_parts = Path(DEFAULT_RUN).parts
DEFAULT_DATASET = _parts[0] if _parts else ""
DEFAULT_TASK = _parts[1] if len(_parts) > 1 else ""


def default_index(options, preferred: str, prefix: str = "") -> int:
    """Index of `preferred` in `options`; else the first option starting with `prefix`; else 0."""
    options = [str(o) for o in options]
    if preferred in options:
        return options.index(preferred)
    if prefix:
        for i, o in enumerate(options):
            if o.startswith(prefix):
                return i
    return 0


# ---------- statistical helpers ----------


def significance_info(n_correct: int, n_total: int, p0: float = 0.25):
    """One-sided binomial test against H0: p=p0. Returns (p, stars, color)."""
    if n_total <= 0:
        return 1.0, "ns", "#F5F5F5"
    p = float(binomtest(int(n_correct), int(n_total), p=p0, alternative="greater").pvalue)
    if p < 0.001:
        return p, "***", "#C8E6C9"
    if p < 0.01:
        return p, "**", "#DCEDC8"
    if p < 0.05:
        return p, "*", "#FFF9C4"
    return p, "ns", "#F5F5F5"


def sig_stars(p) -> str:
    """Significance stars from a p-value (same cutoffs as significance_info)."""
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def read_dataset_metadata(data_root_str: str) -> dict:
    """Load a dataset's optional `*_metadata.json` (features/groups/tasks). {} if absent.

    Provides per-feature descriptions and the in-scanner / out-of-scanner `source` label.
    Datasets without a metadata file (AOMIC / dfg / era812) degrade gracefully to {}.
    Shared by the feature-informativeness and test-comparison apps; wrap in st.cache_data
    at the call site.
    """
    if not data_root_str:
        return {}
    root = Path(data_root_str)
    if not root.exists():
        return {}
    for p in sorted(root.glob("*_metadata.json")) + [root / "metadata.json"]:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
    return {}


def p_value_bg(p) -> str:
    """Green-scale CSS background for a p-value cell (blank when not significant/missing)."""
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return ""
    if p < 0.001:
        return "background-color: #C8E6C9"
    if p < 0.01:
        return "background-color: #DCEDC8"
    if p < 0.05:
        return "background-color: #FFF9C4"
    return ""


def sig_styler(df: pd.DataFrame, sig_cols):
    """Background-color cells based on embedded significance stars."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    star_colors = {"***": "#C8E6C9", "**": "#DCEDC8", "*": "#FFF9C4", "ns": "#F5F5F5"}
    for col in sig_cols:
        if col not in df.columns:
            continue
        for idx, val in df[col].items():
            for stars, color in star_colors.items():
                if stars in str(val):
                    styles.at[idx, col] = f"background-color: {color}"
                    break
    return styles


def stat_interpretation_expander() -> None:
    with st.expander("📊 Statistical interpretation guide", expanded=False):
        st.markdown(
            "**Stars** mark one-sided binomial p-values against H₀: accuracy = 25%:\n\n"
            "| stars | p-value |\n|---|---|\n"
            "| `***` | p < 0.001 |\n| `**` | p < 0.01 |\n| `*` | p < 0.05 |\n| `ns` | p ≥ 0.05 |\n"
        )
        st.markdown(
            "Effect size (Cohen's h, baseline=25%): "
            "~34% = small, ~40% = moderate, ~49% = medium, ~64% = large."
        )
        st.caption(
            "p-values assume independent trials. With sample reuse the effective N "
            "is smaller and p-values are anti-conservative."
        )


# ---------- map plotting ----------


def _train_atlas_n_rois(run_config: Optional[dict]) -> Optional[int]:
    """Pull `train_atlas_cfg.atlas_n_rois` out of a saved config.json (if any)."""
    if not run_config:
        return None
    cfg = run_config.get("train_atlas_cfg")
    if isinstance(cfg, dict):
        return cfg.get("atlas_n_rois")
    return None


def vector_to_brain_img(
    img_data: np.ndarray, masker, run_config: Optional[dict] = None,
):
    """Convert a (n_voxels,) or (n_rois,) vector to a 3D plottable NIfTI.

    For runs trained in atlas space the saved "maps" are length-`atlas_n_rois`
    ROI vectors, not voxel vectors. We detect that case from `run_config`'s
    `train_atlas_cfg` and re-inflate them via the matching Schaefer parcellation.
    """
    vec = np.asarray(img_data, dtype=np.float32).reshape(-1)
    n_voxels_in_mask = int(np.count_nonzero(masker.mask_img_.get_fdata()))

    if vec.shape[0] == n_voxels_in_mask:
        return masker.inverse_transform(vec)

    n_rois = _train_atlas_n_rois(run_config)
    if n_rois is not None and vec.shape[0] == int(n_rois):
        from nilearn import datasets, image
        from nilearn.maskers import NiftiLabelsMasker

        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=int(n_rois))
        roi_masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=False)
        roi_masker.fit(masker.mask_img_)
        roi_img = roi_masker.inverse_transform(vec[np.newaxis, :])
        if roi_img.ndim == 4 and roi_img.shape[3] == 1:
            return image.index_img(roi_img, 0)
        return roi_img

    raise ValueError(
        f"Map length {vec.shape[0]} matches neither the mask "
        f"({n_voxels_in_mask} voxels) nor a known atlas size. "
        f"For atlas-trained runs, the run config must include train_atlas_cfg."
    )


def aggregate_time_dimension(map_array: np.ndarray, mode: str = "none", tr_index: int = 0):
    """Cog2BOLD maps are 1D z-maps, so this is mostly a pass-through.

    Kept for API parity with the legacy app — also handles the case where someone
    saves multi-volume maps in the future.
    """
    if map_array.ndim == 1:
        return map_array, "1D z-map"
    if mode == "mean":
        return map_array.mean(axis=1), "mean over volumes"
    if mode == "max":
        return map_array.max(axis=1), "max over volumes"
    if mode == "tr_index":
        return map_array[:, tr_index], f"vol={tr_index}"
    return map_array.mean(axis=1), "mean over volumes (default)"


def plot_brain_map_with_controls(
    img_data: np.ndarray, masker, title: str,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    nslices: int = 7, cmap: str = "cold_hot", symmetric: bool = True,
    run_config: Optional[dict] = None,
):
    """Plot a 1D voxel/ROI vector through the masker.

    Pass ``run_config`` (the loaded ``config.json`` dict) so atlas-trained runs
    can be displayed correctly.
    """
    if symmetric and (vmax is not None or vmin is not None):
        v = abs(float(vmax)) if vmax is not None else abs(float(vmin))
        vmin, vmax = -v, v
    img = vector_to_brain_img(img_data, masker, run_config=run_config)
    cut_coords = None
    if nslices > 0:
        mask_img = masker.mask_img_
        affine = mask_img.affine
        zb = (affine[2, 3], affine[2, 3] + affine[2, 2] * mask_img.shape[2])
        cut_coords = np.linspace(zb[0], zb[1], nslices + 2)[1:-1]
    fig = plt.figure(figsize=(12, 5))
    plot_stat_map(
        img, bg_img=None, display_mode="z", cut_coords=cut_coords,
        cmap=cmap, vmin=vmin, vmax=vmax, title=title, figure=fig,
        symmetric_cbar=symmetric,
    )
    return fig


# ---------- run loading ----------


@st.cache_data(show_spinner=False)
def load_global_labels(adapter_name: str, data_root_str: str, cog_cols: tuple):
    """Return the dataset's ordered stimulus labels, or None if the adapter can't load.

    Cached on (adapter_name, data_root, cog_cols). Used to render stimulus/contrast
    choices by name in the UI.
    """
    try:
        adapter = create_adapter(
            adapter_name, data_root=Path(data_root_str), cog_base_cols=list(cog_cols),
        )
        return list(adapter.global_labels)
    except Exception:  # noqa: BLE001 — caller falls back to generic names
        return None


@st.cache_data(show_spinner=False)
def load_cog_feature_names(adapter_name: str, data_root_str: str, cog_cols: tuple):
    """Return the adapter's cognitive feature names (may differ from cog_cols, e.g.
    after one-hot expansion). None if the adapter can't load."""
    try:
        adapter = create_adapter(
            adapter_name, data_root=Path(data_root_str), cog_base_cols=list(cog_cols),
        )
        return list(adapter.cog_feature_names)
    except Exception:  # noqa: BLE001
        return None


def training_space(config: dict) -> str:
    """Space the run's basis/predictor were fitted in: "deviation" or "map".

    A run with `mean_map_offset` subtracts the per-contrast training mean map μ before
    fitting, so its basis.decode() output is a *deviation* from μ; everything else is
    fitted on the absolute maps.
    """
    return "deviation" if config.get("mean_map_offset") else "map"


@st.cache_data(show_spinner=False)
def load_mean_maps(run_path_str: str,
                   filename: str = "mean_maps_by_contrast.pkl") -> Dict[str, np.ndarray]:
    """Per-contrast population-mean maps μ, in `MeanMapOffset.state()` layout.

    Two producers share the layout: `mean_maps_by_contrast.pkl` next to a trained run (the
    μ a `mean_map_offset` model was fitted against) and `score_mean_maps.pkl` inside a
    deviation-scored LTO eval dir (the μ its metrics were computed against). Empty dict when
    the file is absent, which is also what makes the deviation view unavailable.
    """
    path = Path(run_path_str) / filename
    if not path.exists():
        return {}
    try:
        state = load_pickle(path)
    except Exception:  # noqa: BLE001 — a missing/corrupt μ just disables map space
        return {}
    if not state.get("enabled"):
        return {}
    return {str(k): np.asarray(v) for k, v in (state.get("mean_by_contrast") or {}).items()}


def match_mean_map(mean_maps: Dict[str, np.ndarray], label: str) -> Optional[str]:
    """Best contrast key for a stimulus/contrast label, or None if it can't be resolved.

    The live tab names contrasts "A  >  B" while the offset stores the dataset's own
    names (e.g. "correct_gt_control"), so an exact hit is the exception; with a single
    stored μ (the common single-contrast run) there is no ambiguity to resolve.
    """
    if not mean_maps:
        return None
    if label in mean_maps:
        return label
    if len(mean_maps) == 1:
        return next(iter(mean_maps))
    return None


def format_config_summary(config: dict, basis=None) -> str:
    """One-line human-readable summary of a run's config (for headers/captions).

    If `basis` is a fitted HybridBasis, the supervised/unsupervised sub-bases and
    their component counts are included, e.g.
    "basis **hybrid** (sup: ridgecol(9) + unsup: pca(55))".
    """
    bc = config.get("basis_config", {})
    pc = config.get("predictor_config", {})

    basis_str = f"basis **{bc.get('basis_type', '?')}**"
    if bc.get("basis_type") == "hybrid":
        extra = bc.get("extra_params", {}) or {}
        sup = extra.get("supervised_method", "?")
        unsup = extra.get("unsupervised_method", "?")
        k_sup = getattr(basis, "K_sup", None)
        k_unsup = getattr(basis, "K_unsup", None)
        sup_c = f"({k_sup})" if k_sup is not None else ""
        unsup_c = f"({k_unsup})" if k_unsup is not None else ""
        basis_str += f" (sup: {sup}{sup_c} + unsup: {unsup}{unsup_c})"
    elif bc.get("n_components"):
        basis_str += f" k={bc['n_components']}"

    parts = [basis_str, f"predictor **{pc.get('model_type', '?')}**"]
    loss = pc.get("loss_mode", "?")
    if loss == "map":
        loss = f"map/{pc.get('map_loss_type', 'l2')}"
    parts.append(f"loss {loss}")
    if pc.get("model_type") == "mlp":
        hd = pc.get("hidden_dims")
        if hd:
            parts.append(f"hidden {','.join(map(str, hd))}")
        parts.append(f"{pc.get('n_epochs', '?')} epochs")
        parts.append(f"lr {pc.get('learning_rate', '?')}")
    parts.append(f"trained in **{training_space(config)} space**")
    return " · ".join(parts)


def spatial_metrics(gt_vec: np.ndarray, pred_vec: np.ndarray) -> Dict[str, float]:
    """Per-sample spatial quality metrics between one GT map and one prediction.

    Computed *across voxels* for a single (gt, pred) pair — distinct from
    `metrics.voxelwise_correlation`, which correlates across samples and is
    undefined (returns 0) for a single sample.
    """
    g = np.asarray(gt_vec, dtype=np.float64).reshape(-1)
    p = np.asarray(pred_vec, dtype=np.float64).reshape(-1)
    mask = np.isfinite(g) & np.isfinite(p)
    g, p = g[mask], p[mask]
    eps = 1e-12
    mse = float(np.mean((g - p) ** 2)) if g.size else float("nan")
    mae = float(np.mean(np.abs(g - p))) if g.size else float("nan")
    cosine = float(np.dot(g, p) / (np.linalg.norm(g) * np.linalg.norm(p) + eps)) if g.size else float("nan")
    gc, pc = g - g.mean(), p - p.mean()
    den = np.linalg.norm(gc) * np.linalg.norm(pc)
    pearson = float(np.dot(gc, pc) / den) if den > eps else 0.0
    return {"pearson_r": pearson, "cosine": cosine, "mse": mse, "mae": mae}


@st.cache_data(show_spinner=False)
def load_run_files(run_path_str: str):
    run_path = Path(run_path_str)
    try:
        preds = load_pickle(run_path / "predictions.pkl")
        metrics = load_json(run_path / "metrics.json")
        config = load_json(run_path / "config.json")
        return preds, metrics, config, None
    except Exception as e:  # noqa: BLE001 — surface to UI
        return None, None, None, str(e)


def load_masker_for_run(run_path: Path, config: dict, app_data_root: Optional[Path] = None) -> Optional[NiftiMasker]:
    """Build a fresh masker via the run's adapter — no pickled-masker monkey-patching."""
    try:
        data_root = Path(app_data_root) if app_data_root else Path(config["data_root"])
        cog_cols = config["data_config"]["cog_base_cols"]
        adapter_name = config["data_config"]["adapter_name"]
        adapter = create_adapter(adapter_name, data_root=data_root, cog_base_cols=cog_cols,
                                 **(config["data_config"].get("adapter_kwargs") or {}))
        return adapter.get_masker()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build masker: {e}")
        return None


def load_run(run_path: Path, app_data_root: Optional[Path] = None):
    preds, metrics, config, err = load_run_files(str(run_path))
    if err:
        return None, None, None, None, err
    masker = load_masker_for_run(run_path, config, app_data_root)
    return preds, masker, metrics, config, None


# ---------- live prediction ----------


def predict_activation_map(
    basis: BasisTransform,
    predictor: CoefficientPredictor,
    cog_raw: np.ndarray,
    stim_encoding: np.ndarray,
):
    """Run a single prediction.

    `predict()` accepts a 2D stim_encoding (one-hot or contrast vector). The bug in
    the legacy code that fed integer `stim_id` to the predictor is fixed by
    construction here.
    """
    coeffs = predictor.predict(
        cog_raw.reshape(1, -1), stim_encoding.reshape(1, -1), warn_unseen=False,
    )
    return basis.decode(coeffs[0])


# ---------- cognitive sliders constrained to the training support ----------


@st.cache_resource(show_spinner=False)
def fit_cog_support(run_key: str, _X_subjects: np.ndarray) -> CogSupport:
    """CogSupport for a run (cached per run; ``_X_subjects`` = one raw row per subject)."""
    return CogSupport.fit(_X_subjects)


def subject_rows(cog_train: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """One raw cognitive vector per training subject (first occurrence)."""
    _, first = np.unique(np.asarray(subject_ids), return_index=True)
    return np.asarray(cog_train[np.sort(first)], dtype=np.float64)


def render_constrained_sliders(
    support: CogSupport,
    names,
    keys,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    allow_extrap: bool,
    ncols: int = 2,
) -> Tuple[np.ndarray, dict]:
    """Cognitive sliders whose ranges keep the profile inside the plausible region.

    The caller initialises ``st.session_state[key]`` for every key. With ``allow_extrap``
    off, the current profile is first projected inside the region, then each continuous
    slider's range is its closed-form feasible interval (others held fixed) ∩ the box — so
    no drag can leave the region. Binary/constant features keep the box range.
    Returns (profile, info) with info = {d2, tau, pct, inside}.
    """
    box_lo, box_hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
    x = np.clip([float(st.session_state[k]) for k in keys], box_lo, box_hi)
    if not allow_extrap:
        x = support.project(x)  # stays in the box: radial shrink toward an in-box centre
    grid = st.columns(ncols)
    for j, (name, key) in enumerate(zip(names, keys)):
        lo, hi = float(box_lo[j]), float(box_hi[j])
        if hi <= lo:  # constant feature, no slider
            x[j] = lo
            continue
        step = (hi - lo) / 100.0
        iv = None if allow_extrap else support.feasible_interval(x, j)
        if iv is not None:
            lo, hi = max(lo, iv[0]), min(hi, iv[1])
            step = min(step, max(hi - lo, 0.0) / 50.0) or step
        x[j] = float(np.clip(x[j], lo, hi))
        st.session_state[key] = x[j]
        with grid[j % ncols]:
            if hi - lo < 1e-9:  # pinned by the others (profile sits on the boundary)
                st.slider(name, lo - step, hi + step, step=step, key=key, disabled=True)
            else:
                st.slider(name, lo, hi, step=step, key=key)
    d2 = support.mahalanobis2(x)
    info = dict(d2=d2, tau=support.tau, pct=support.percentile(x), inside=d2 <= support.tau * (1 + 1e-9))
    if not support.is_trivial:
        st.caption(
            f"Typicality: more atypical than **{info['pct']:.0f}%** of training subjects "
            f"(Mahalanobis d² = {d2:.2f}; plausible-region limit τ = {support.tau:.2f}, "
            f"the {support.quantile:.0%} quantile of training subjects)."
        )
    return x, info


def ood_warning(info: dict) -> None:
    """Warn that the current profile lies outside the model's training support."""
    if not info["inside"]:
        st.warning(
            f"⚠️ This profile is outside the training support (d² = {info['d2']:.2f} > "
            f"τ = {info['tau']:.2f}; more atypical than {info['pct']:.0f}% of training "
            "subjects). The model never saw such a combination — treat the map as extrapolation."
        )


# ---------- shared LTO-evaluation renderer (used by basis_compare + lto_curves) ----------

_METRIC_OPTIONS = {"Correlation": "correlation", "MSE": "mse"}


def _lto_block(summary: dict, metric: str):
    metrics = summary.get("metrics")
    if metrics and metric in metrics:
        return metrics[metric]
    if metric == "correlation":
        overall = summary.get("overall", {})
        return {"overall": overall, "by_corr_config": overall.get("by_corr_config", {}),
                "per_contrast": summary.get("per_contrast", {}),
                "null_model": summary.get("null_model")}
    return None


def _lto_cols(metric: str):
    if metric == "mse":
        return "mse", "correct_mse", "mse11 < mse12 AND mse22 < mse21"
    return "r", "correct", "r11 > r12 AND r22 > r21"


@st.cache_data(show_spinner=False)
def _pair_cog_context(adapter_name: str, data_root_str: str, cog_cols: tuple,
                      adapter_kwargs_json: str):
    """Feature names (incl. one-hot expansion) + subject->diagnosis map + is-diagnosis-an-input."""
    kw = json.loads(adapter_kwargs_json) if adapter_kwargs_json else {}
    try:
        ad = create_adapter(adapter_name, data_root=Path(data_root_str),
                            cog_base_cols=list(cog_cols), **kw)
    except Exception:  # noqa: BLE001
        return {"names": list(cog_cols), "diag": {}, "onehot": False}
    names = [str(n) for n in getattr(ad, "cog_feature_names", cog_cols)]
    diag = {int(k): str(v) for k, v in (getattr(ad, "_diag_by_sid", {}) or {}).items()}
    onehot = any(n.startswith("Group:") for n in names) or bool(
        getattr(ad, "include_group_onehot", False))
    return {"names": names, "diag": diag, "onehot": onehot}


@st.cache_resource(show_spinner="Building masker…")
def _masker_for_config(adapter_name: str, data_root_str: str, cog_cols: tuple,
                       adapter_kwargs_json: str):
    kw = json.loads(adapter_kwargs_json) if adapter_kwargs_json else {}
    ad = create_adapter(adapter_name, data_root=Path(data_root_str),
                        cog_base_cols=list(cog_cols), **kw)
    return ad.get_masker()


def render_lto_eval(sel_dir: Path, run_config: dict, key_prefix: str = "lto",
                    app_data_root: Optional[Path] = None) -> None:
    """Render one LTO eval dir (summary + config panels + per-pair browser).

    Shared by basis_compare's LTO tab and the lto_curves drill-down, so both look
    identical. Brain-map images show only if the eval saved a `maps/` dir; otherwise
    the numeric per-pair table + config panels still render. All widget keys are
    namespaced by `key_prefix`.
    """
    sel_dir = Path(sel_dir)
    summary_path = sel_dir / "lto_results_summary.json"
    if not summary_path.exists():
        st.warning(f"No lto_results_summary.json in {sel_dir}")
        return
    summary = json.loads(summary_path.read_text())

    dc = run_config.get("data_config", {})
    data_root = str(app_data_root) if app_data_root else run_config.get("data_root", "")
    akw = dc.get("adapter_kwargs") or {}
    akw_json = json.dumps(akw, sort_keys=True)

    # ---- run config + LTO config panels ----
    bc = run_config.get("basis_config", {}); pc = run_config.get("predictor_config", {})
    ep = bc.get("extra_params") or {}
    basis = bc.get("basis_type", "?")
    if basis == "hybrid":
        basis += f" ({ep.get('supervised_method','?')}+{ep.get('unsupervised_method','?')})"
    con = summary.get("constraints", {})
    pcol, lcol = st.columns(2)
    with pcol:
        st.markdown("**Run config** (model)")
        st.dataframe(pd.DataFrame([
            {"param": "adapter", "value": dc.get("adapter_name", "?")},
            {"param": "basis", "value": f"{basis}  k={bc.get('n_components')}"},
            {"param": "predictor", "value": pc.get("model_type", "?")},
            {"param": "loss", "value": pc.get("loss_mode", "?")
                + (f"/{pc.get('map_loss_type')}" if pc.get("loss_mode") == "map" else "")},
            {"param": "cog_only", "value": pc.get("cog_only", False)},
            {"param": "trained space", "value": training_space(run_config)},
            {"param": "cog features", "value": ", ".join(dc.get("cog_base_cols", []))},
            {"param": "diag one-hot", "value": bool(akw.get("include_group_onehot"))},
            {"param": "group filter", "value": ", ".join(akw.get("groups", [])) or "ALL"},
            {"param": "single_contrast", "value": dc.get("single_contrast") or "(all)"},
        ]).astype({"value": str}), hide_index=True, use_container_width=True)
    with lcol:
        st.markdown("**LTO config** (evaluation)")
        st.dataframe(pd.DataFrame([
            {"param": "min_cog_distance", "value": con.get("min_cog_distance")},
            {"param": "max_pairs/contrast", "value": con.get("max_pairs_per_contrast")},
            {"param": "unique_samples", "value": con.get("unique_samples")},
            {"param": "same_diagnosis", "value": con.get("same_diagnosis", False)},
            {"param": "different_subjects", "value": con.get("different_subjects")},
            {"param": "p_threshold", "value": con.get("p_threshold")},
            {"param": "spaces", "value": ", ".join(
                c.get("name", "?") for c in summary.get("correlation_configs", []))},
            {"param": "scored in", "value": f"{summary.get('score_space', 'map')} space"},
            {"param": "chance", "value": summary.get("chance_level")},
        ]).astype({"value": str}), hide_index=True, use_container_width=True)

    metric = _METRIC_OPTIONS[st.radio("Similarity metric", list(_METRIC_OPTIONS),
                                      horizontal=True, key=f"{key_prefix}_metric")]
    block = _lto_block(summary, metric)
    if block is None:
        st.warning("This LTO result predates MSE support; only correlation is available.")
        return
    _, _, rule_txt = _lto_cols(metric)
    scored_space = str(summary.get("score_space", "map"))
    st.caption(f"Scoring by **{metric}** — a pair is *correct* when `{rule_txt}` (chance = 25%). "
               f"Maps were compared in **{scored_space} space**.")
    if scored_space == "deviation":
        st.caption(
            "Deviation scoring: the per-contrast mean of each pair's N−2 training maps was "
            "subtracted from the prediction *and* the ground truth before scoring, so the "
            "shared component common to all maps of a contrast cancels out."
        )
    skipped = summary.get("skipped_contrasts") or []
    if skipped:
        st.warning(
            "Contrasts excluded from this eval (deviation scoring needs ≥3 samples per "
            f"contrast to build μ from the training subset): {', '.join(skipped)}"
        )

    overall = block["overall"]
    p, stars, _ = significance_info(overall["correct_pairs"], overall["total_pairs"])
    c1, c2, c3 = st.columns(3)
    c1.metric("total pairs", overall["total_pairs"])
    c2.metric("accuracy", f"{overall['accuracy']:.1%} {stars}",
              delta=f"{overall['accuracy']-0.25:+.1%} vs chance")
    c3.metric("p-value (binomial)", f"{p:.3g}")
    stat_interpretation_expander()

    # per-contrast table
    per = block.get("per_contrast", {})
    cfg_names = [c["name"] for c in summary.get("correlation_configs", [])]
    if per:
        st.subheader("By stimulus / contrast")
        rows = []
        for cname, s in per.items():
            row = {"contrast": cname, "pairs": s["n_pairs"]}
            bycfg = s.get("by_corr_config", {})
            if cfg_names and bycfg:
                for cn in cfg_names:
                    cs = bycfg.get(cn)
                    if cs:
                        _, st_, _ = significance_info(cs["n_correct"], s["n_pairs"])
                        row[f"acc ({cn})"] = f"{cs['accuracy']:.1%} {st_}"
                    else:
                        row[f"acc ({cn})"] = "—"
            else:
                _, st_, _ = significance_info(s["n_correct"], s["n_pairs"])
                row["accuracy"] = f"{s['accuracy']:.1%} {st_}"
            rows.append(row)
        cdf = pd.DataFrame(rows)
        acc_cols = [c for c in cdf.columns if c.startswith("acc (") or c == "accuracy"]
        st.dataframe(cdf.style.apply(sig_styler, sig_cols=acc_cols, axis=None),
                     hide_index=True, use_container_width=True)

    # ---- per-pair browser ----
    csv_path = sel_dir / "lto_results_detailed.csv"
    if not csv_path.exists():
        return
    pdf = pd.read_csv(csv_path)
    with st.expander("Per-pair details (table)"):
        st.dataframe(pdf, use_container_width=True)
    if "pair_id" not in pdf.columns or not len(pdf):
        return

    st.markdown("---")
    st.subheader("Per-pair inspector")
    prefix, correct_col, rule_txt = _lto_cols(metric)
    if correct_col not in pdf.columns:
        correct_col, prefix = "correct", "r"
    d11, d22 = f"{prefix}11", f"{prefix}22"
    f1, f2 = st.columns(2)
    with f1:
        f_contrast = st.selectbox("Filter contrast", ["All"] + sorted(pdf["contrast"].unique()),
                                  key=f"{key_prefix}_bc")
    with f2:
        f_correct = st.selectbox("Filter outcome", ["All", "correct", "incorrect"],
                                 key=f"{key_prefix}_bo")
    fdf = pdf
    if f_contrast != "All":
        fdf = fdf[fdf["contrast"] == f_contrast]
    if f_correct == "correct":
        fdf = fdf[fdf[correct_col].astype(bool)]
    elif f_correct == "incorrect":
        fdf = fdf[~fdf[correct_col].astype(bool)]
    if not len(fdf):
        st.info("No pairs match the filter.")
        return

    def _pair_label(i):
        r = pdf.loc[i]
        tick = "✓" if bool(r[correct_col]) else "✗"
        return (f"pair #{int(r['pair_id'])} {tick} | {r['contrast']} | "
                f"subj {r['subj1']}/{r['subj2']} | {d11}={r[d11]:.3f} {d22}={r[d22]:.3f}")

    pidx = st.selectbox("Pair", fdf.index.tolist(), format_func=_pair_label,
                        key=f"{key_prefix}_pair")
    prow = pdf.loc[pidx]
    is_corr = bool(prow[correct_col])
    bg, fg = ("#d4edda", "#155724") if is_corr else ("#f8d7da", "#721c24")
    txt = "✓ CORRECT CLASSIFICATION" if is_corr else "✗ INCORRECT CLASSIFICATION"
    st.markdown(f"<div style='background-color:{bg};padding:12px;border-radius:10px;"
                f"border-left:6px solid {fg};margin-bottom:10px;'><div style='font-size:1.3em;"
                f"font-weight:bold;color:{fg};text-align:center;'>{txt}</div>"
                f"<div style='text-align:center;color:{fg};opacity:0.85;'>correct ⇔ {rule_txt}</div>"
                f"</div>", unsafe_allow_html=True)
    if "cog_distance" in prow.index and pd.notna(prow["cog_distance"]):
        st.metric("cognitive distance (standardized L2)", f"{float(prow['cog_distance']):.3f}")

    # cognitive input values for the pair + diagnosis
    ctx = _pair_cog_context(dc.get("adapter_name", "?"), data_root,
                            tuple(dc.get("cog_base_cols", [])), akw_json)
    try:
        cA = [float(x) for x in ast.literal_eval(str(prow["cog1"]))]
        cB = [float(x) for x in ast.literal_eval(str(prow["cog2"]))]
    except Exception:  # noqa: BLE001
        cA, cB = [], []
    if cA and cB:
        s1, s2 = int(prow["subj1"]), int(prow["subj2"])
        names = ctx["names"] if len(ctx["names"]) == len(cA) else [f"feature {i}" for i in range(len(cA))]
        st.markdown("**Input cognitive values** (what the predictor received)")
        st.dataframe(pd.DataFrame({"feature": names, f"subj {s1}": cA, f"subj {s2}": cB}),
                     hide_index=True, use_container_width=True)
        g1, g2 = ctx["diag"].get(s1, "n/a"), ctx["diag"].get(s2, "n/a")
        note = ("✓ included as one-hot input features above" if ctx["onehot"]
                else "⚠️ NOT an input feature — shown for reference only")
        st.caption(f"Diagnosis group — subj {s1}: **{g1}** · subj {s2}: **{g2}**  ({note})")

    # similarity matrix (primary config)
    better_low = (metric == "mse")
    v11, v12 = float(prow[f"{prefix}11"]), float(prow[f"{prefix}12"])
    v21, v22 = float(prow[f"{prefix}21"]), float(prow[f"{prefix}22"])
    v11_win = (v11 < v12) if better_low else (v11 > v12)
    v22_win = (v22 < v21) if better_low else (v22 > v21)
    st.markdown(f"<div style='text-align:center;font-weight:bold;'>{metric.upper()} matrix "
                f"(primary config)</div>", unsafe_allow_html=True)
    mat = pd.DataFrame({"": ["pred 1", "pred 2"],
                        "true 1": [f"{v11:.3f} {'✓' if v11_win else '✗'}", f"{v21:.3f}"],
                        "true 2": [f"{v12:.3f}", f"{v22:.3f} {'✓' if v22_win else '✗'}"]})

    def _style_mat(df):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        g, r = "background-color:#d4edda", "background-color:#f8d7da"
        s.iloc[0, df.columns.get_loc("true 1")] = g if v11_win else r
        s.iloc[1, df.columns.get_loc("true 2")] = g if v22_win else r
        return s
    st.columns([1, 2, 1])[1].dataframe(mat.style.apply(_style_mat, axis=None),
                                       hide_index=True, use_container_width=True)

    # brain maps (only if the eval saved them)
    maps_root = sel_dir / "maps"
    pdir = maps_root / f"pair_{int(prow['pair_id']):05d}"
    if (pdir / "true_map_1.npy").exists():
        t1 = np.load(pdir / "true_map_1.npy"); t2 = np.load(pdir / "true_map_2.npy")
        p1 = np.load(pdir / "pred_map_1.npy"); p2 = np.load(pdir / "pred_map_2.npy")
        try:
            masker = _masker_for_config(dc.get("adapter_name", "?"), data_root,
                                        tuple(dc.get("cog_base_cols", [])), akw_json)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not build masker for maps: {e}")
            return

        def _sym98(*arrs):
            v = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs])
            m = float(np.percentile(np.abs(v), 98)) if v.size else 1.0
            return m or 1.0

        # The pipeline restores μ before scoring, so the saved maps are always absolute.
        # Offer the deviation view whenever a μ is available — from the eval itself when it
        # scored in deviation space, else from the run's training offset.
        space_suffix = ""
        eval_mu = load_mean_maps(str(sel_dir), "score_mean_maps.pkl")
        mean_maps = eval_mu or load_mean_maps(str(sel_dir.parent))
        mu_from_eval = bool(eval_mu)
        if mean_maps or scored_space == "deviation" or training_space(run_config) == "deviation":
            mu_key = match_mean_map(mean_maps, str(prow["contrast"]))
            if mu_key is None:
                st.info(
                    "No per-contrast μ is available for these maps "
                    f"(looked for `{sel_dir.name}/score_mean_maps.pkl` and "
                    f"`{sel_dir.parent.name}/mean_maps_by_contrast.pkl`), so they can only be "
                    "shown in map space."
                )
            else:
                options = ["Map space (absolute)", "Deviation space (− μ)"]
                view = st.radio(
                    "Visualisation space", options,
                    index=1 if scored_space == "deviation" else 0,
                    horizontal=True,
                    # key varies with the eval so switching evals re-defaults the view to
                    # that eval's own scoring space instead of keeping a stale pick
                    key=f"{key_prefix}_view_space__{sel_dir.name}",
                    help=f"This eval scored in **{scored_space} space** — the matrix above and "
                         "the accuracy always reflect that and never change here. The toggle "
                         "only changes the pictures: deviation space subtracts the same μ from "
                         "all four maps, so the shared component stops dominating the view.",
                )
                mu = mean_maps[mu_key]
                if view.startswith("Deviation"):
                    if mu.shape != t1.shape:
                        st.warning(
                            f"μ['{mu_key}'] has shape {mu.shape} but the maps are {t1.shape} "
                            "— showing map space."
                        )
                    else:
                        t1, t2 = t1 - mu, t2 - mu
                        p1, p2 = p1 - mu, p2 - mu
                        space_suffix = f" · deviation space (− μ['{mu_key}'])"
                        note = (
                            "the same space the metrics above were computed in"
                            if scored_space == "deviation" else
                            "note the metrics above were computed in **map** space"
                        )
                        src = ("this eval's scoring μ (per-contrast mean over all samples; "
                               "each pair scored against its own N−2 training mean, which "
                               "differs by O(1/N))" if mu_from_eval else
                               "the run's training-offset μ")
                        st.caption(
                            f"**Deviation space** — μ['{mu_key}'] subtracted from all four "
                            f"maps, taken from {src}. This is {note}."
                        )

        separate = st.checkbox(
            "Separate color scale for GT and predicted", value=False,
            help="Off: ground-truth and predicted maps share one symmetric scale (98th pct "
                 "over all four) so magnitudes are directly comparable. On: the GT pair and "
                 "the predicted pair each use their own 98th-pct scale.",
            key=f"{key_prefix}_sep_scale",
        )
        if separate:
            gt_vmax = _sym98(t1, t2)
            pred_vmax = _sym98(p1, p2)
        else:
            gt_vmax = pred_vmax = _sym98(t1, t2, p1, p2)

        st.markdown(f"**Ground truth** · scale ±{gt_vmax:.3g}{space_suffix}")
        g1c, g2c = st.columns(2)
        g1c.pyplot(plot_brain_map_with_controls(t1, masker, "True 1", -gt_vmax, gt_vmax, 7,
                   run_config=run_config), clear_figure=True)
        g2c.pyplot(plot_brain_map_with_controls(t2, masker, "True 2", -gt_vmax, gt_vmax, 7,
                   run_config=run_config), clear_figure=True)
        st.markdown(f"**Predicted** · scale ±{pred_vmax:.3g}{space_suffix}")
        q1c, q2c = st.columns(2)
        q1c.pyplot(plot_brain_map_with_controls(p1, masker, "Pred 1 (→ True 1)", -pred_vmax, pred_vmax, 7,
                   run_config=run_config), clear_figure=True)
        q2c.pyplot(plot_brain_map_with_controls(p2, masker, "Pred 2 (→ True 2)", -pred_vmax, pred_vmax, 7,
                   run_config=run_config), clear_figure=True)
    else:
        st.caption("No per-pair maps saved for this eval "
                   "(run the LTO with `save_maps` to enable the brain-map browser).")


__all__ = [
    "aggregate_time_dimension",
    "format_config_summary",
    "load_cog_feature_names",
    "load_global_labels",
    "load_masker_for_run",
    "load_mean_maps",
    "load_run",
    "load_run_files",
    "match_mean_map",
    "plot_brain_map_with_controls",
    "predict_activation_map",
    "p_value_bg",
    "read_dataset_metadata",
    "render_lto_eval",
    "sig_stars",
    "sig_styler",
    "significance_info",
    "spatial_metrics",
    "stat_interpretation_expander",
    "training_space",
    "vector_to_brain_img",
]
