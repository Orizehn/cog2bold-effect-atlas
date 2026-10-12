"""Summary documentation for a `ScanResult` — HTML report, paper-ready tables, NIfTI maps.

Output directory (default ``<run_dir>/effect_atlas_<mode>_<timestamp>/``)::

    report.html                 self-contained (figures embedded as base64 PNG)
    summary.csv                 one ranked row per (condition, combination)
    peaks.csv | peaks.md | peaks.tex   peak tables (LaTeX: booktabs + longtable)
    roi_effects_yeo7.csv | roi_effects_ho.csv   mean Δ per region (+ bootstrap CI)
    figures/*.png
    maps/<cond>__<combo>_{delta,stat,clusters}.nii.gz
    scan_params.json
"""

from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from cog2bold.analysis.effect_atlas.atlas_labels import YEO7_NAMES  # noqa: E402
from cog2bold.analysis.effect_atlas.scan import EffectMap, ScanResult  # noqa: E402

# diverging blue ↔ gray ↔ red (positive = red, matching activation colour convention)
DIVERGING = LinearSegmentedColormap.from_list("c2b_div", ["#2a78d6", "#f0efec", "#e34948"])
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"

_SHORT = {
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "SMA (juxtapositional lobule)",
    "Cingulate Gyrus, anterior division": "Anterior cingulate",
    "Cingulate Gyrus, posterior division": "Posterior cingulate",
}


def short_region(name: str) -> str:
    for long, short in _SHORT.items():
        name = name.replace(long, short)
    return name.replace(", posterior division", ", post.").replace(", anterior division", ", ant.") \
        .replace(", superior division", ", sup.").replace(", inferior division", ", inf.") \
        .replace(", temporooccipital part", ", temporo-occ.")


PEAK_COLUMNS = ["ho_region", "network", "x", "y", "z", "peak_stat", "peak_delta", "n_voxels", "volume_mm3",
                "regions_covered"]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:80]


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------- figures


def effect_figures(run, stat: np.ndarray, labels: np.ndarray, threshold: float, title: str,
                   cut=None):
    """(glass-brain figure, ortho-slice figure) of the surviving clusters of one statistic map.

    Only voxels inside clusters that passed the extent threshold are drawn. ``cut`` defaults to
    the strongest surviving voxel. Returns (None, None) when nothing survived.
    """
    from nilearn.plotting import plot_glass_brain, plot_stat_map

    shown = np.where(labels != 0, stat, 0.0)
    if not np.any(shown):
        return None, None
    img = run.to_img(shown)
    thr = threshold if np.isfinite(threshold) else float(np.abs(stat).max()) + 1
    vmax = max(float(np.abs(shown).max()), thr * 1.01)

    glass = plt.figure(figsize=(10, 3.4))
    plot_glass_brain(img, threshold=thr, plot_abs=False, colorbar=True, cmap="cold_hot",
                     vmax=vmax, vmin=-vmax, symmetric_cbar=True, title=title, figure=glass,
                     display_mode="lyrz")
    if cut is None:
        from cog2bold.analysis.effect_atlas.atlas_labels import mask_voxel_coords

        cut = tuple(mask_voxel_coords(run.mask_img)[int(np.abs(shown).argmax())])
    ortho = plt.figure(figsize=(10, 3.4))
    plot_stat_map(img, threshold=thr, cut_coords=cut, display_mode="ortho", cmap="cold_hot",
                  vmax=vmax, symmetric_cbar=True, figure=ortho,
                  title=f"({cut[0]:.0f}, {cut[1]:.0f}, {cut[2]:.0f})")
    return glass, ortho


def brain_figures(res: ScanResult, eff: EffectMap, fig_dir: Path) -> List[Path]:
    """Save the glass brain + ortho slices (at the strongest peak) for one effect."""
    glass, ortho = effect_figures(res.run, eff.stat, eff.labels, eff.threshold,
                                  f"{eff.combo.name} · {eff.condition}")
    out = []
    base = _slug(f"{eff.condition}__{eff.combo.name}")
    for fig, suffix in ((glass, "glass"), (ortho, "ortho")):
        if fig is not None:
            p = fig_dir / f"{base}_{suffix}.png"
            fig.savefig(p, dpi=110, bbox_inches="tight")
            plt.close(fig)
            out.append(p)
    return out


def region_heatmap(res: ScanResult, condition: str, atlas: str, path: Optional[Path] = None,
                   top_regions: int = 15):
    """Combinations × regions heatmap of mean Δ; bootstrap CIs excluding 0 are marked with •.

    Saves to ``path`` and returns it, or returns the figure when ``path`` is None.
    """
    df = res.roi_effects(atlas)
    df = df[df["condition"] == condition]
    if df.empty:
        return None
    M = df.pivot_table(index="combo", columns="region", values="mean_delta", sort=False)
    M = M.loc[[c.name for c in res.combos if c.name in M.index]]
    if atlas == "yeo7":
        M = M[[r for r in YEO7_NAMES if r in M.columns]]
        M.columns = [YEO7_NAMES[c] for c in M.columns]
    else:
        M = M[M.abs().max().sort_values(ascending=False).index[:top_regions]]
    sig = None
    if "ci_low" in df:
        ex = df.assign(sig=(df["ci_low"] > 0) | (df["ci_high"] < 0))
        sig = ex.pivot_table(index="combo", columns="region", values="sig", sort=False, aggfunc="max")
        sig = sig.loc[M.index]
        sig = sig[[r for r in YEO7_NAMES if r in sig.columns]] if atlas == "yeo7" else sig[M.columns]

    what = "Yeo-7 network" if atlas == "yeo7" else f"top {M.shape[1]} Harvard-Oxford regions"
    if atlas != "yeo7":  # many long names: regions as rows reads better
        M = M.T
        M.index = [short_region(r) for r in M.index]
        sig = sig.T if sig is not None else None
    v = float(np.nanmax(np.abs(M.to_numpy()))) or 1.0
    h = 0.42 * len(M) + 2.2
    w = 1.1 * M.shape[1] + 5.0
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(M.to_numpy(), cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=40, ha="right", fontsize=9, color=TEXT)
    ax.set_yticks(range(len(M)), M.index, fontsize=9, color=TEXT)
    for i in range(len(M)):
        for j in range(M.shape[1]):
            val = M.iat[i, j]
            if np.isnan(val):
                continue
            mark = " •" if sig is not None and bool(sig.iat[i, j]) else ""
            ax.text(j, i, f"{val:+.2f}{mark}", ha="center", va="center", fontsize=7.5, color=TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(M) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mean Δ in region (map units)", color=TEXT_2, fontsize=9)
    cb.outline.set_visible(False)
    note = " · • = bootstrap 95% CI excludes 0" if sig is not None else ""
    ax.set_title(f"{condition}: mean effect per {what}{note}", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    if path is None:
        return fig
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- text tables


def _fmt_row(r) -> dict:
    return {
        "Region": r["ho_region"] + (f" [{r['ho_flag']}]" if r.get("ho_flag") else ""),
        "P(%)": f"{r['ho_prob']:.0f}",
        "Network": YEO7_NAMES.get(r["network"], r["network"] or "–"),
        "x": f"{r['x']:.0f}", "y": f"{r['y']:.0f}", "z": f"{r['z']:.0f}",
        "stat": f"{r['peak_stat']:+.2f}",
        "k": "" if r["peak_type"] == "sub" else str(r["n_voxels"]),
        "mm³": "" if r["peak_type"] == "sub" else f"{r['volume_mm3']:.0f}",
    }


def peaks_markdown(res: ScanResult) -> str:
    out = [f"# Effect-atlas peak tables — {res.run.title}", "",
           f"Statistic: {res.stat_label}. Sub-peaks are indented and carry no size.", ""]
    for e in _ranked_effects(res):
        if not len(e.peaks):
            continue
        out += [f"## {e.combo.description} · {e.condition}", ""]
        for sign in ("activation", "deactivation"):
            sub = e.peaks[e.peaks["sign"] == sign]
            if not len(sub):
                continue
            rows = [_fmt_row(r) for _, r in sub.iterrows()]
            for row, (_, r) in zip(rows, sub.iterrows()):
                if r["peak_type"] == "sub":
                    row["Region"] = "↳ " + row["Region"]
            cols = list(rows[0])
            out += [f"**{sign.capitalize()}s**", "", "| " + " | ".join(cols) + " |",
                    "|" + "|".join("---" for _ in cols) + "|"]
            out += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
            out.append("")
    return "\n".join(out)


def _tex(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("×", r"$\times$"), ("−", "--"), ("↳", r"\quad"),
                 ("³", r"$^3$")):
        s = s.replace(a, b)
    return s


def peaks_latex(res: ScanResult) -> str:
    """booktabs/longtable peak table — needs \\usepackage{booktabs,longtable}."""
    stat = "$z$" if res.params.mode == "bootstrap" else r"$\Delta$"
    out = [r"% Effect-atlas peak table — requires \usepackage{booktabs,longtable}",
           r"\begin{longtable}{llrrrrrr}",
           r"\toprule",
           rf"Region & Network & $x$ & $y$ & $z$ & {stat} & $k$ & mm$^3$ \\",
           r"\midrule", r"\endhead"]
    for e in _ranked_effects(res):
        if not len(e.peaks):
            continue
        out.append(rf"\multicolumn{{8}}{{l}}{{\textbf{{{_tex(e.combo.description)}}} "
                   rf"({_tex(e.condition)})}} \\")
        for sign in ("activation", "deactivation"):
            sub = e.peaks[e.peaks["sign"] == sign]
            if not len(sub):
                continue
            out.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{sign.capitalize()}s}}}} \\")
            for _, r in sub.iterrows():
                f = _fmt_row(r)
                region = ("↳ " if r["peak_type"] == "sub" else "") + f["Region"]
                out.append(" & ".join(_tex(x) for x in (
                    region, f["Network"], f["x"], f["y"], f["z"], f["stat"], f["k"], f["mm³"]
                )) + r" \\")
        out.append(r"\midrule")
    out += [r"\bottomrule",
            rf"\caption{{Peaks of model-predicted effect maps ({_tex(res.stat_label)}). "
            r"Coordinates in MNI space (mm); labels: Harvard--Oxford (probability at peak) "
            r"and Schaefer-200 / Yeo-7.}",
            r"\end{longtable}"]
    return "\n".join(out)


def _ranked_effects(res: ScanResult) -> List[EffectMap]:
    s = res.summary()
    return [res.effects[(r.condition, r.combo)] for r in s.itertuples()]


# ---------------------------------------------------------------- methods text


def methods_text(res: ScanResult) -> str:
    p = res.params
    cfg = res.run.config
    combos = ", ".join(sorted({c.kind for c in res.combos}))
    if p.mode == "bootstrap":
        thr = (f"Benjamini–Hochberg FDR q < {p.fdr_q:g} (two-sided)" if p.fdr_q
               else f"|z| > {p.z_thr:g} (voxelwise, uncorrected)")
        stat = (f"Each effect map Δ was divided by its bootstrap standard deviation over "
                f"{res.boot.n_boot if res.boot else p.n_boot} subject-level resamples of the training set "
                f"(predictor refitted on each resample; spatial basis held fixed), giving a bootstrap z map "
                f"thresholded at {thr}. The bootstrap z measures how stable an effect is across "
                f"refits, not how large it is: a negligible but consistent Δ can reach a high z, so "
                f"read the peak Δ (map units) next to every z.")
    else:
        lvl = (f"|Δ| > {p.abs_threshold:g}" if p.abs_threshold is not None
               else f"the {p.percentile:g}th percentile of |Δ| pooled over all scanned maps")
        stat = (f"Effect maps were thresholded descriptively at {lvl}; no significance is "
                f"implied in this mode.")
    return (
        f"Effect maps were computed from the trained {cfg.predictor_config.model_type} predictor "
        f"({cfg.basis_config.basis_type} basis, {cfg.basis_config.n_components} components) as "
        f"zero-sum contrasts of predicted maps ({combos}); single-feature effects compare the "
        f"feature at +{p.k_sd:g} vs −{p.k_sd:g} training SD (binary features 1 vs 0) with all other "
        f"features at their training mean. {stat} Clusters were formed with "
        f"{p.connectivity}-connectivity and a minimum extent of {p.min_cluster_voxels} voxels; up "
        f"to {p.max_subpeaks} sub-peaks ≥ {p.min_peak_distance_mm:g} mm apart are listed per cluster. "
        f"Peaks were labelled with the Harvard–Oxford probabilistic atlas (cortical + subcortical) "
        f"and the Schaefer-200 parcellation / Yeo-7 networks. Note: both atlases are defined in FSL "
        f"MNI152 (NLin6Asym); the model's maps may be in MNI152NLin2009cAsym, a few-mm difference."
    )


# ---------------------------------------------------------------- HTML


_CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;color:#0b0b0b;max-width:1180px;margin:24px auto;padding:0 16px;line-height:1.45}
h1{font-size:24px;margin-bottom:4px} h2{font-size:19px;margin-top:36px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:15px;margin:22px 0 6px} .muted{color:#52514e} .small{font-size:12px}
table{border-collapse:collapse;font-size:12.5px;margin:8px 0} th,td{padding:4px 8px;border-bottom:1px solid #eee;text-align:left}
th{background:#f6f6f4;font-weight:600} td.num{text-align:right;font-variant-numeric:tabular-nums}
.banner{background:#fff6e5;border-left:4px solid #fab219;padding:10px 14px;margin:14px 0}
.card{display:grid;grid-template-columns:200px 1fr;gap:2px 16px;font-size:13px}
img{max-width:100%} .combo{margin-bottom:28px}
"""


def _table(df: pd.DataFrame, float_fmt="{:.3g}") -> str:
    return df.to_html(index=False, escape=True, border=0, na_rep="",
                      float_format=lambda x: float_fmt.format(x))


def write_report(res: ScanResult, out_dir: Optional[Path] = None, max_detailed: int = 30,
                 save_maps: bool = True) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else res.run.run_dir / f"effect_atlas_{res.params.mode}_{stamp}"
    fig_dir, map_dir = out / "figures", out / "maps"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary, peaks = res.summary(), res.peaks()
    summary.to_csv(out / "summary.csv", index=False)
    peaks.to_csv(out / "peaks.csv", index=False)
    (out / "peaks.md").write_text(peaks_markdown(res))
    (out / "peaks.tex").write_text(peaks_latex(res))
    res.roi_effects("yeo7").to_csv(out / "roi_effects_yeo7.csv", index=False)
    res.roi_effects("ho").to_csv(out / "roi_effects_ho.csv", index=False)
    (out / "scan_params.json").write_text(json.dumps(res.params_dict(), indent=2))

    if save_maps:
        map_dir.mkdir(exist_ok=True)
        for e in res.effects.values():
            stem = _slug(f"{e.condition}__{e.combo.name}")
            res.run.to_img(e.delta).to_filename(map_dir / f"{stem}_delta.nii.gz")
            res.run.to_img(e.stat).to_filename(map_dir / f"{stem}_stat.nii.gz")
            res.run.to_img(e.labels.astype(np.float32)).to_filename(map_dir / f"{stem}_clusters.nii.gz")

    cfg = res.run.config
    val = res.run.validity_summary()
    parts = [f"<html><head><meta charset='utf-8'><title>Effect atlas — {html.escape(res.run.title)}</title>"
             f"<style>{_CSS}</style></head><body>",
             f"<h1>Cog2BOLD effect atlas</h1><div class='muted'>{html.escape(str(res.run.run_dir))}"
             f" · generated {datetime.now():%Y-%m-%d %H:%M}</div>"]

    # model card
    card = {
        "Dataset adapter": cfg.data_config.adapter_name,
        "Condition(s)": ", ".join(c.name for c in res.conditions),
        "Basis": f"{cfg.basis_config.basis_type} ({cfg.basis_config.n_components} comp.)",
        "Predictor": f"{cfg.predictor_config.model_type}, {cfg.predictor_config.loss_mode} loss",
        "Features": ", ".join(res.run.feature_names),
        "Training subjects": str(len(np.unique(res.run.train["subject_ids_train"]))),
        "Threshold mode": res.params.mode,
    }
    parts.append("<h2>Model</h2><div class='card'>" + "".join(
        f"<div class='muted'>{html.escape(k)}</div><div>{html.escape(v)}</div>" for k, v in card.items())
        + "</div>")

    # validity
    vrows = []
    if "voxel_correlation_mean" in val:
        vrows.append(f"held-out voxel correlation (mean): <b>{val['voxel_correlation_mean']:.3f}</b>")
    if "explained_variance_r2" in val:
        vrows.append(f"held-out R²: <b>{val['explained_variance_r2']:.3f}</b>")
    if val.get("lto_accuracy") is not None:
        s = f"LTO accuracy: <b>{val['lto_accuracy']:.3f}</b> over {val.get('lto_pairs')} pairs ({val['lto_eval']})"
        if val.get("lto_null"):
            s += " · null: " + ", ".join(f"{k}={v:.3g}" for k, v in val["lto_null"].items())
        else:
            s += (" · <i>no permutation null stored — do not compare against 0.25 "
                  "(the nominal chance level is an upper bound)</i>")
        vrows.append(s)
    parts.append(
        "<div class='banner'><b>These maps describe the model, not the brain.</b> An effect map is "
        "what the trained predictor would change if a feature changed; it is only as meaningful as "
        "the model's validated out-of-sample signal. Read every blob against the numbers below.<br>"
        + ("<br>".join(vrows) if vrows else "<i>no held-out metrics found in the run directory</i>")
        + "</div>")
    if res.notes:
        parts.append("<ul class='small muted'>" + "".join(f"<li>{html.escape(n)}</li>" for n in res.notes) + "</ul>")
    parts.append(f"<h2>Methods</h2><p class='small'>{html.escape(methods_text(res))}</p>")

    # overview
    show = summary.drop(columns=["threshold"]) if len(summary) else summary
    parts.append("<h2>Ranked combinations</h2><p class='small muted'>Ranked by total suprathreshold "
                 "volume, then by ‖Δ‖₂. Statistic: " + html.escape(res.stat_label) + ".</p>" + _table(show))
    for cond in res.conditions:
        for atlas in ("yeo7", "ho"):
            p = region_heatmap(res, cond.name, atlas, fig_dir / f"{_slug(f'heatmap_{atlas}_{cond.name}')}.png")
            if p:
                parts.append(f"<img src='data:image/png;base64,{_b64(p)}'>")

    # archetype profiles
    arch = [c for c in res.combos if c.kind == "archetype"]
    if arch:
        prof = pd.DataFrame({c.name: c.meta["z_profile"] for c in arch}).T
        prof.insert(0, "n", [c.meta["n_subjects"] for c in arch])
        parts.append("<h3>Archetype profiles (feature z-scores of k-means centroids)</h3>"
                     + _table(prof.reset_index(names="archetype"), "{:+.2f}"))

    # per-combination detail
    parts.append(f"<h2>Effect maps and peak tables</h2><p class='small muted'>Top {max_detailed} "
                 "combinations in rank order; every peak is in peaks.csv / peaks.tex.</p>")
    for e in _ranked_effects(res)[:max_detailed]:
        parts.append(f"<div class='combo'><h3>{html.escape(e.combo.description)} · "
                     f"<span class='muted'>{html.escape(e.condition)}</span></h3>")
        parts.append(f"<div class='small muted'>threshold {html.escape(res.stat_label)} = "
                     f"{e.threshold:.3g} · ‖Δ‖₂ = {np.linalg.norm(e.delta):.3g} · max |Δ| = "
                     f"{np.abs(e.delta).max():.3g}</div>")
        for p in brain_figures(res, e, fig_dir):
            parts.append(f"<img src='data:image/png;base64,{_b64(p)}'>")
        if len(e.peaks):
            cols = ["cluster_id", "sign", "peak_type"] + PEAK_COLUMNS + (
                ["sign_consistency"] if res.params.mode == "bootstrap" else [])
            parts.append(_table(e.peaks[[c for c in cols if c in e.peaks]]))
        else:
            parts.append("<p class='muted small'>No suprathreshold clusters.</p>")
        parts.append("</div>")
    parts.append("</body></html>")
    (out / "report.html").write_text("\n".join(parts))
    return out
