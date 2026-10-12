"""Effect atlas: which cognitive features move which brain regions, according to a trained run.

Scans feature combinations (single-feature effects, pairwise interactions, subject archetypes,
custom profiles), turns each into an effect map Δ, thresholds it (descriptive effect mode or
bootstrap-calibrated z), finds activation/deactivation blobs and labels their peaks with
Harvard-Oxford + Schaefer/Yeo-7 — a classic neuroimaging peak table per combination.

    from cog2bold.analysis.effect_atlas import ScanParams, run_scan, write_report
    res = run_scan(run_dir, ScanParams(mode="bootstrap", n_boot=200))
    write_report(res)
"""

from cog2bold.analysis.effect_atlas.combinations import Combination, custom
from cog2bold.analysis.effect_atlas.model_io import Condition, LoadedRun
from cog2bold.analysis.effect_atlas.scan import (
    EffectMap,
    ScanParams,
    ScanResult,
    run_scan,
    threshold_effect,
)

__all__ = [
    "Combination", "Condition", "EffectMap", "LoadedRun", "ScanParams", "ScanResult",
    "custom", "run_scan", "threshold_effect", "write_report",
]


def write_report(*args, **kwargs):
    from cog2bold.analysis.effect_atlas.report import write_report as _w  # matplotlib-heavy

    return _w(*args, **kwargs)
