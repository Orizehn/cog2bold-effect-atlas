"""Cog2BOLD — Effect atlas.

Which cognitive features move which brain regions, according to a trained run?

Pick a run → scan its feature combinations (single-feature effects, pairwise interactions,
subject archetypes) → explore each effect map's activation/deactivation blobs with MNI peak
coordinates, Harvard-Oxford + Yeo-7 labels and cluster sizes → build custom profiles → ask the
reverse question per ROI → export the full report (HTML, CSV, LaTeX, NIfTI).

    streamlit run apps/effect_atlas.py

Public mode (for deployment, see deploy/hf_space/): set ``COG2BOLD_PUBLIC=1`` and point
``COG2BOLD_BUNDLES`` at a folder of run bundles (``cog2bold export-run``). The app then lists
only those bundles, hides every filesystem path input, caps bootstrap resamples
(``COG2BOLD_MAX_BOOT``, default 500) and offers reports as downloads only.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps._shared import (  # noqa: E402
    DEFAULT_DATASET, DEFAULT_RUN, DEFAULT_TASK, default_index, fit_cog_support, ood_warning,
    render_constrained_sliders,
)
from cog2bold.analysis.effect_atlas import LoadedRun, ScanParams, custom, run_scan, threshold_effect  # noqa: E402
from cog2bold.analysis.effect_atlas.atlas_labels import YEO7_NAMES  # noqa: E402
from cog2bold.analysis.effect_atlas.combinations import feature_levels, stack  # noqa: E402
from cog2bold.analysis.effect_atlas.report import effect_figures, methods_text, region_heatmap, write_report  # noqa: E402
from cog2bold.analysis.effect_atlas.thresholding import bootstrap, pooled_percentile_threshold  # noqa: E402
from cog2bold.analysis.effect_atlas.bundle import BUNDLE_FILE  # noqa: E402
from cog2bold.io import load_json  # noqa: E402
from cog2bold.workspace import get_workspace  # noqa: E402

PUBLIC = os.environ.get("COG2BOLD_PUBLIC", "").lower() in ("1", "true", "yes")
MAX_BOOT = int(os.environ.get("COG2BOLD_MAX_BOOT", "500" if PUBLIC else "2000"))
GUIDE_PDF = Path(os.environ.get(
    "COG2BOLD_GUIDE",
    Path(__file__).parent.parent / "docs/effect_atlas_guide/effect_atlas_user_guide.pdf"))

st.set_page_config(layout="wide", page_title="Cog2BOLD — effect atlas")

POS, NEG = "#e34948", "#2a78d6"  # diverging poles (activation / deactivation)
PEAK_VIEW_COLS = ["cluster_id", "sign", "peak_type", "ho_region", "ho_prob", "ho_flag", "network",
                  "schaefer_parcel", "x", "y", "z", "peak_stat", "peak_delta", "n_voxels",
                  "volume_mm3", "regions_covered", "sign_consistency"]

# ============================================================ sidebar: run ====

if PUBLIC:
    # no path inputs: visitors can only reach the shipped bundles
    runs_dir = Path(os.environ.get("COG2BOLD_BUNDLES", Path(__file__).parent.parent / "bundles"))
    _override = ""
    bundle_dirs = sorted(p.parent for p in runs_dir.rglob(BUNDLE_FILE))
    if not bundle_dirs:
        st.error("No run bundles are installed on this server.")
        st.stop()
    # present each bundle under its original dataset/task/run name
    _by_title = {load_json(d / BUNDLE_FILE)["title"]: str(d.relative_to(runs_dir)) for d in bundle_dirs}
    available = sorted(_by_title)
else:
    st.sidebar.header("Configuration")
    workspace = Path(st.sidebar.text_input("Workspace", value=str(get_workspace()),
                                           help="$COG2BOLD_WORKSPACE — where runs live"))
    runs_dir = workspace / "runs"
    _override = st.sidebar.text_input(
        "Dataset root override (optional)", value="",
        help="Leave blank to use each run's stored data_root.",
    ).strip()
    if not runs_dir.exists():
        st.error(f"runs dir not found: {runs_dir}. Train a model first via `cog2bold train`.")
        st.stop()
    # TRAINED runs (predictor.pkl + predictions.pkl) and exported bundles
    available = sorted(str(p.parent.relative_to(runs_dir)) for p in runs_dir.rglob("predictor.pkl")
                       if (p.parent / "predictions.pkl").exists() or (p.parent / BUNDLE_FILE).exists())
    _by_title = {rp: rp for rp in available}
if not available:
    st.warning(f"No trained runs found under {runs_dir}.")
    st.stop()

tree: dict = {}
for rp in available:
    parts = Path(rp).parts
    ds = parts[0] if parts else "(flat)"
    task = parts[1] if len(parts) >= 3 else "(none)"
    tree.setdefault(ds, {}).setdefault(task, []).append(rp)

st.sidebar.markdown("### Select run")
_ds = st.sidebar.selectbox("Dataset", sorted(tree), key="ea_dataset",
                           index=default_index(sorted(tree), DEFAULT_DATASET))
_task = st.sidebar.selectbox("Task", sorted(tree[_ds]), key="ea_task",
                             index=default_index(sorted(tree[_ds]), DEFAULT_TASK))
run_rel = st.sidebar.selectbox(
    "Run", sorted(tree[_ds][_task]),
    index=default_index(sorted(tree[_ds][_task]), DEFAULT_RUN), format_func=lambda rp: "/".join(Path(rp).parts[2:]) or rp,
    key=f"ea_run__{_ds}__{_task}",
)


@st.cache_resource(show_spinner="Loading run…")
def load_run(run_path: str, data_root: str) -> LoadedRun:
    return LoadedRun.from_dir(run_path, data_root=data_root or None)


run_rel = _by_title[run_rel]  # display name -> folder (identity outside public mode)
try:
    run = load_run(str(runs_dir / run_rel), _override)
except Exception as e:  # noqa: BLE001 — surface any load failure in the UI
    st.error(f"Could not load {run_rel}: {e}")
    st.stop()

# ============================================================ sidebar: scan ===

st.sidebar.markdown("### Scan")
mode = st.sidebar.radio("Threshold mode", ["effect", "bootstrap"], horizontal=True,
                        help="effect: descriptive, on Δ itself. bootstrap: Δ / bootstrap SD over "
                             "subject resamples (predictor refitted each time).")
kinds = st.sidebar.multiselect(
    "Combinations", ["single", "pairwise", "archetype"], default=["single", "pairwise", "archetype"],
    help="pairwise interactions are skipped automatically for linear predictors (identically 0)",
)
k_sd = st.sidebar.number_input("High/low = mean ± k·SD", 0.25, 3.0, 1.0, 0.25)
arch_k = st.sidebar.number_input("Archetypes k (0 = auto by silhouette)", 0, 8, 0)
if mode == "effect":
    percentile = st.sidebar.slider("Pooled |Δ| percentile threshold", 90.0, 99.9, 99.0, 0.1)
    n_boot, z_thr, fdr_q = 0, 3.1, None
else:
    percentile = 99.0
    n_boot = st.sidebar.number_input("Bootstrap resamples", 10, MAX_BOOT,
                                     min(200 if run.is_linear else 50, MAX_BOOT), 10,
                                     help="linear refits take ms; MLP refits take seconds each")
    use_fdr = st.sidebar.checkbox("FDR instead of fixed |z|", value=False)
    fdr_q = st.sidebar.number_input("FDR q", 0.001, 0.2, 0.05, 0.005) if use_fdr else None
    z_thr = st.sidebar.number_input("|z| threshold", 1.0, 8.0, 3.1, 0.1, disabled=use_fdr)
min_vox = st.sidebar.number_input("Min cluster size (voxels)", 1, 500, 10)
conn = st.sidebar.selectbox("Connectivity", [26, 18, 6], index=0)
if GUIDE_PDF.exists():
    st.sidebar.download_button("📘 User guide (PDF)", GUIDE_PDF.read_bytes(),
                               "effect_atlas_user_guide.pdf", mime="application/pdf")

params = ScanParams(
    combos=tuple(kinds), k_sd=float(k_sd), archetypes_k=int(arch_k) or None, mode=mode,
    percentile=float(percentile), n_boot=int(n_boot), z_thr=float(z_thr),
    fdr_q=float(fdr_q) if fdr_q else None, min_cluster_voxels=int(min_vox), connectivity=int(conn),
)
params_key = json.dumps(params.__dict__, sort_keys=True, default=str)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_scan(run_path: str, data_root: str, key: str):
    bar = st.progress(0.0, text="scanning…")
    res = run_scan(load_run(run_path, data_root), ScanParams(**json.loads(key)),
                   progress=lambda stage, f: bar.progress(min(f, 1.0), text=stage))
    bar.empty()
    return res


if not kinds:
    st.warning("Choose at least one combination type.")
    st.stop()
auto = mode == "effect" or st.session_state.get("ea_scanned") == (run_rel, params_key)
if not auto:
    st.info(f"Bootstrap scan with {n_boot} resamples is ready to run.")
    if st.button("▶ Run bootstrap scan", type="primary"):
        st.session_state["ea_scanned"] = (run_rel, params_key)
        auto = True
    else:
        st.stop()
res = cached_scan(str(runs_dir / run_rel), _override, params_key)
summary = res.summary()

st.title("🧠 Effect atlas")
st.caption(f"{run.title} · {run.config.basis_config.basis_type} basis · "
           f"{run.config.predictor_config.model_type} predictor · statistic: {res.stat_label}")
if PUBLIC:
    with st.expander("About this app", expanded=False):
        st.markdown(
            "This app asks a trained **Cog2BOLD** model how a person's cognitive scores change "
            "their predicted fMRI activation map. For each feature or combination of features "
            "it finds the blobs of activation and deactivation and reports them like a "
            "neuroimaging paper: peak MNI coordinates, Harvard-Oxford and Yeo-7 labels, and "
            "cluster size. **The maps describe the model, not the brain:** read them against "
            "the validity banner in the Overview tab.")
        attribution = (run.bundle_meta or {}).get("attribution")
        if attribution:
            st.caption(attribution)

tab_over, tab_explore, tab_custom, tab_roi, tab_report = st.tabs(
    ["🧭 Overview", "🧠 Explore effect", "🎛️ Custom profile", "🔍 ROI lookup", "📄 Report"])


# ------------------------------------------------------------ shared views


def validity_banner() -> None:
    v = run.validity_summary()
    bits = []
    if "voxel_correlation_mean" in v:
        bits.append(f"held-out voxel corr **{v['voxel_correlation_mean']:.3f}**")
    if "explained_variance_r2" in v:
        bits.append(f"held-out R² **{v['explained_variance_r2']:.3f}**")
    if v.get("lto_accuracy") is not None:
        s = f"LTO accuracy **{v['lto_accuracy']:.3f}** ({v.get('lto_pairs')} pairs)"
        s += (" vs null " + ", ".join(f"{k}={x:.3g}" for k, x in v["lto_null"].items())
              if v.get("lto_null") else " — no permutation null stored (don't compare to 0.25)")
        bits.append(s)
    st.warning("**These maps describe the model, not the brain.** Read every blob against the "
               "model's validated signal: " + (" · ".join(bits) if bits else "no held-out metrics found."),
               icon="⚠️")


def show_effect(stat, labels, threshold, title, peaks: pd.DataFrame, key: str, delta=None) -> None:
    """Figures + interactive viewer + selectable peak table for one thresholded effect."""
    sel_cut = None
    if len(peaks):
        cols = [c for c in PEAK_VIEW_COLS if c in peaks]
        event = st.dataframe(
            peaks[cols], hide_index=True, width="stretch", on_select="rerun",
            selection_mode="single-row", key=f"peaks_{key}",
            column_config={"ho_prob": st.column_config.NumberColumn("HO %", format="%.0f"),
                           "peak_stat": st.column_config.NumberColumn(format="%+.2f"),
                           "peak_delta": st.column_config.NumberColumn("peak Δ", format="%+.3f"),
                           "volume_mm3": st.column_config.NumberColumn("mm³", format="%.0f"),
                           "sign_consistency": st.column_config.NumberColumn("sign cons.", format="%.2f")},
        )
        rows = event.selection.rows if event and event.selection else []
        if rows:
            r = peaks.iloc[rows[0]]
            sel_cut = (float(r["x"]), float(r["y"]), float(r["z"]))
            st.caption(f"Slices centred on the selected peak {sel_cut} — {r['ho_region']}")
        else:
            st.caption("Select a row to centre the slices on that peak.")
    else:
        st.info("No suprathreshold clusters at this threshold.")
        return
    glass, ortho = effect_figures(run, stat, labels, threshold, title, cut=sel_cut)
    if glass is not None:
        st.pyplot(glass, clear_figure=True)
        st.pyplot(ortho, clear_figure=True)
    with st.expander("Interactive 3D viewer"):
        from nilearn.plotting import view_img

        shown = np.where(labels != 0, stat, 0.0)
        view = view_img(run.to_img(shown), threshold=threshold if np.isfinite(threshold) else 0,
                        symmetric_cmap=True, cmap="cold_hot", cut_coords=sel_cut,
                        title=title)
        st.iframe(view.html, height=int(view.height) + 30)


def nifti_bytes(vec) -> bytes:
    return run.to_img(np.asarray(vec, dtype=np.float32)).to_bytes()


# ============================================================ Tab 1: overview ==

with tab_over:
    validity_banner()
    for n in res.notes:
        st.caption(f"ℹ️ {n}")
    st.subheader("Ranked combinations")
    st.caption("Ranked by total suprathreshold volume, then ‖Δ‖₂.")
    st.dataframe(summary.drop(columns=["threshold"]), hide_index=True, width="stretch",
                 column_config={"suprathreshold_mm3": st.column_config.ProgressColumn(
                     "suprathreshold mm³", format="%.0f",
                     max_value=float(max(summary["suprathreshold_mm3"].max(), 1.0)))})
    for cond in res.conditions:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.pyplot(region_heatmap(res, cond.name, "yeo7"), clear_figure=True)
        with c2:
            st.pyplot(region_heatmap(res, cond.name, "ho"), clear_figure=True)
    arch = [c for c in res.combos if c.kind == "archetype"]
    if arch:
        st.subheader("Archetype profiles")
        prof = pd.DataFrame({c.name: c.meta["z_profile"] for c in arch}).T
        prof.insert(0, "n subjects", [c.meta["n_subjects"] for c in arch])
        st.caption(f"k-means on z-scored training subjects (k={arch[0].meta['k']}, "
                   f"silhouette {arch[0].meta['silhouette']:.2f}); values are feature z-scores.")
        st.dataframe(prof.style.format("{:+.2f}", subset=prof.columns[1:])
                     .background_gradient(cmap="RdBu_r", vmin=-2, vmax=2, subset=prof.columns[1:]),
                     width="stretch")
    with st.expander("Methods text (paper-ready)"):
        st.write(methods_text(res))


# ============================================================ Tab 2: explore ===

with tab_explore:
    c1, c2 = st.columns([1, 3])
    with c1:
        cond = st.selectbox("Condition", [c.name for c in res.conditions], key="ex_cond")
        ranked = summary[summary["condition"] == cond]
        combo_name = st.selectbox(
            "Combination (ranked)", ranked["combo"].tolist(),
            format_func=lambda n: f"#{int(ranked.set_index('combo').loc[n, 'rank'])} {n}",
            key="ex_combo")
        eff = res.effects[(cond, combo_name)]
        st.caption(eff.combo.description)
        hi = float(np.abs(eff.stat).max())
        default_thr = float(min(eff.threshold, hi)) if np.isfinite(eff.threshold) else hi
        thr = st.slider(f"Threshold on |{res.stat_label}|", 0.0, max(hi, 1e-6), default_thr,
                        max(hi, 1e-6) / 200, format="%.3f", key=f"ex_thr_{cond}_{combo_name}_{params_key}")
        mv = st.number_input("Min cluster (voxels)", 1, 500, int(min_vox), key="ex_minvox")
        st.metric("‖Δ‖₂", f"{np.linalg.norm(eff.delta):.3g}")
        st.metric("max |Δ| (map units)", f"{np.abs(eff.delta).max():.3g}")
    live = eff
    if thr != eff.threshold or mv != min_vox:
        p2 = ScanParams(**{**params.__dict__, "min_cluster_voxels": int(mv)})
        live = threshold_effect(run, res.regions, cond, eff.combo, eff.delta, p2, threshold=thr,
                                sd=eff.sd, consistency=eff.consistency)
    with c2:
        show_effect(live.stat, live.labels, live.threshold, f"{combo_name} · {cond}", live.peaks,
                    key=f"ex_{cond}_{combo_name}")
        d1, d2, d3 = st.columns(3)
        stem = f"{cond}__{combo_name}".replace(" ", "_").replace("×", "x")
        d1.download_button("⬇ Δ map (.nii)", nifti_bytes(live.delta), f"{stem}_delta.nii")
        d2.download_button("⬇ stat map (.nii)", nifti_bytes(live.stat), f"{stem}_stat.nii")
        d3.download_button("⬇ peaks (.csv)", live.peaks.to_csv(index=False).encode(), f"{stem}_peaks.csv")


# ============================================================ Tab 3: custom ====

with tab_custom:
    st.caption("Build any two cognitive profiles A and B; the effect map is pred(A) − pred(B). "
               "Sliders are limited to the joint plausible region of the training subjects "
               "unless extrapolation is allowed.")
    X = run.subject_profiles()
    support = fit_cog_support(str(run.run_dir), X)
    lo_f, hi_f = X.min(axis=0), X.max(axis=0)
    low, high, _ = feature_levels(run, float(k_sd))
    # "f high/low" keep the other features realistic: E[others | f] under the support model,
    # not the population mean (which can be an implausible combination when features correlate)
    presets = {"mean profile": support.project(run.cog_mean)}
    for j, f in enumerate(run.feature_names):
        presets[f"{f} high"] = support.conditional_profile(j, high[j], run.cog_mean)
        presets[f"{f} low"] = support.conditional_profile(j, low[j], run.cog_mean)
    for c in res.combos:
        if c.kind == "archetype":
            presets[c.name] = support.project(c.profiles[0])
    cond_c = st.selectbox("Condition", [c.name for c in res.conditions], key="cu_cond")
    allow_extrap = st.checkbox(
        "Allow extrapolation (leave the training support)", value=False, key="cu_extrap",
        help="Off: moving one feature narrows the range of the features correlated with it, "
             "so A and B stay inside the 95% plausible region of the training subjects.",
    )
    colA, colB = st.columns(2)
    profiles, infos = {}, {}
    for col, tag, default in ((colA, "A", 1), (colB, "B", 0)):
        with col:
            names = list(presets)
            pre = st.selectbox(f"Profile {tag} preset", names,
                               index=min(default, len(names) - 1), key=f"cu_pre_{tag}")
            keys = [f"cu_{tag}_{j}_{pre}" for j in range(len(run.feature_names))]
            for j, k in enumerate(keys):
                st.session_state.setdefault(k, float(np.clip(presets[pre][j], lo_f[j], hi_f[j])))
            profiles[tag], infos[tag] = render_constrained_sliders(
                support, run.feature_names, keys, lo_f, hi_f, allow_extrap, ncols=1)
    for tag in "AB":
        if not infos[tag]["inside"]:
            st.caption(f"Profile {tag}:")
            ood_warning(infos[tag])
    combo = custom("custom", profiles["A"], profiles["B"], "custom profile A − B")
    condition = next(c for c in res.conditions if c.name == cond_c)
    W, P = stack([combo])
    delta = run.combine(W, P, condition)[0]
    za = (profiles["A"] - run.cog_mean) / run.cog_std
    zb = (profiles["B"] - run.cog_mean) / run.cog_std
    st.dataframe(pd.DataFrame({"feature": run.feature_names, "A (z)": za, "B (z)": zb,
                               "A − B (z)": za - zb}).style.format(precision=2),
                 hide_index=True, width="stretch")

    do_boot = st.checkbox("Bootstrap this contrast", value=False,
                          help="refits the predictor on subject resamples for this one contrast")
    sd = cons = None
    cparams = ScanParams(**{**params.__dict__, "mode": "bootstrap" if do_boot else "effect"})
    if do_boot:
        nb = st.number_input("Resamples", 10, 2000, 100 if run.is_linear else 30, 10, key="cu_nb")
        if st.button("▶ Run bootstrap", key="cu_go"):
            with st.spinner("bootstrapping…"):
                b = bootstrap(run, [combo], [condition], {condition.name: delta[None]},
                              n_boot=int(nb), seed=int(run.config.random_seed))
            st.session_state["cu_boot"] = (combo.profiles.tobytes(), cond_c, b.sd[cond_c][0],
                                           b.consistency[cond_c][0])
        cached = st.session_state.get("cu_boot")
        if cached and cached[0] == combo.profiles.tobytes() and cached[1] == cond_c:
            sd, cons = cached[2], cached[3]
        else:
            st.info("Press ▶ to bootstrap the current profiles.")
    if not do_boot or sd is not None:
        if do_boot:
            thr_c = st.number_input("|z| threshold", 1.0, 10.0, float(z_thr), 0.1, key="cu_z")
        else:
            scan_thr = (pooled_percentile_threshold([e.delta for e in res.effects.values()], percentile)
                        if mode == "effect" else float(np.percentile(np.abs(delta), 99)))
            thr_c = st.number_input("|Δ| threshold (default: the scan's pooled threshold)",
                                    0.0, None, float(scan_thr), format="%.4f", key="cu_thr")
        ce = threshold_effect(run, res.regions, cond_c, combo, delta, cparams, threshold=thr_c,
                              sd=sd, consistency=cons)
        st.metric("max |Δ| (map units)", f"{np.abs(delta).max():.3g}")
        show_effect(ce.stat, ce.labels, ce.threshold, "custom A − B", ce.peaks, key="cu")
    with st.expander("Absolute predicted maps for A and B"):
        maps = run.predict_maps(np.stack([profiles["A"], profiles["B"]]), condition)
        from nilearn.plotting import plot_stat_map

        for tag, m in zip("AB", maps):
            fig = plt.figure(figsize=(10, 3))
            plot_stat_map(run.to_img(m), display_mode="z", cut_coords=7, cmap="cold_hot",
                          symmetric_cbar=True, title=f"predicted map, profile {tag}", figure=fig)
            st.pyplot(fig, clear_figure=True)


# ============================================================ Tab 4: ROI ======

with tab_roi:
    st.caption("The reverse question: pick a region, see which features / combinations move it.")
    c1, c2, c3 = st.columns(3)
    atlas = c1.radio("Atlas", ["yeo7", "ho"], horizontal=True,
                     format_func=lambda a: "Yeo-7 networks" if a == "yeo7" else "Harvard-Oxford regions")
    roi = res.roi_effects(atlas)
    names = (list(YEO7_NAMES) if atlas == "yeo7" else res.regions.ho_names)
    region = c2.selectbox("Region", names,
                          format_func=lambda r: YEO7_NAMES.get(r, r) if atlas == "yeo7" else r)
    cond_r = c3.selectbox("Condition", [c.name for c in res.conditions], key="roi_cond")
    sub = roi[(roi["region"] == region) & (roi["condition"] == cond_r)].dropna(subset=["mean_delta"])
    if sub.empty:
        st.info("No model voxels fall in this region.")
    else:
        sub = sub.sort_values("mean_delta")
        err = None
        if "ci_low" in sub:
            err = dict(type="data", symmetric=False, color="#52514e", thickness=1.5,
                       array=(sub["ci_high"] - sub["mean_delta"]).to_numpy(),
                       arrayminus=(sub["mean_delta"] - sub["ci_low"]).to_numpy())
        fig = go.Figure(go.Bar(
            x=sub["mean_delta"], y=sub["combo"], orientation="h", error_x=err,
            marker=dict(color=np.where(sub["mean_delta"] >= 0, POS, NEG), cornerradius=4),
            customdata=sub[["kind"]].to_numpy(),
            hovertemplate="%{y}<br>mean Δ = %{x:+.3f}<br>%{customdata[0]}<extra></extra>",
        ))
        label = YEO7_NAMES.get(region, region) if atlas == "yeo7" else region
        fig.update_layout(
            title=f"Mean Δ in {label} ({cond_r})" + (" · bars: bootstrap 95% CI" if err else ""),
            xaxis_title="mean Δ in region (map units)", yaxis_title=None,
            height=max(260, 34 * len(sub) + 120), margin=dict(l=10, r=10, t=50, b=40),
            plot_bgcolor="#fcfcfb", bargap=0.35,
        )
        fig.update_xaxes(zeroline=True, zerolinecolor="#52514e", gridcolor="#ececea")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(sub.drop(columns=["condition", "region"]), hide_index=True,
                     width="stretch")


# ============================================================ Tab 5: report ===

with tab_report:
    st.caption("Writes the full summary — HTML report, ranked summary, peak tables (CSV, "
               "Markdown, LaTeX/booktabs), ROI effects and NIfTI maps — for the current scan.")
    max_det = st.number_input("Combinations with figures in the HTML", 1, 100 if PUBLIC else 500, 30)
    save_maps = st.checkbox("Include NIfTI maps", value=True)
    # public servers never write into the bundle folder
    where = ("download only" if PUBLIC else
             st.radio("Write to", ["run directory", "download only"], horizontal=True))
    if st.button("📄 Build report", type="primary"):
        import tempfile

        with st.spinner("writing report…"), tempfile.TemporaryDirectory() as tmp:
            out_dir = None if where == "run directory" else Path(tmp) / f"effect_atlas_{mode}"
            out = write_report(res, out_dir=out_dir, max_detailed=int(max_det), save_maps=save_maps)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in out.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(out.parent))
            html_bytes = (out / "report.html").read_bytes()
        st.success("Report ready." if where == "download only" else f"Report written to {out}")
        st.download_button("⬇ Download report (.zip)", buf.getvalue(), f"{out.name}.zip",
                           mime="application/zip")
        st.download_button("⬇ report.html", html_bytes, "report.html", mime="text/html")
