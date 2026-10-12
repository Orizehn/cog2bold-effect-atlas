"""Workspace locations for run outputs and caches.

Keeps Cog2BOLD git-portable: datasets live outside the repo (passed to each
adapter via ``data_root``), while *outputs* (trained runs, eval dirs, sample
caches, computed masks) go to a single workspace dir that is independent of both
the repo and the datasets.

Resolution order for the workspace root:
    1. $COG2BOLD_WORKSPACE
    2. ~/cog2bold_workspace   (default)

Layout:
    <workspace>/
        runs/<dataset>/<task>/<run_name>/   # config.json, basis.pkl, predictor.pkl, eval dirs
        cache/<adapter_name>/               # materialized sample caches, computed masks
"""

from __future__ import annotations

import os
from pathlib import Path


def get_workspace() -> Path:
    env = os.environ.get("COG2BOLD_WORKSPACE")
    root = Path(env).expanduser() if env else (Path.home() / "cog2bold_workspace")
    return root


def runs_root() -> Path:
    p = get_workspace() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_root() -> Path:
    p = get_workspace() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def adapter_cache_dir(adapter_name: str) -> Path:
    p = cache_root() / adapter_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_run_name(adapter_name: str, basis_config, predictor_config) -> str:
    """Auto run name: <dataset>_<predictor>_<basis>[_<sup>_<unsup>]_<n_comp>.

    e.g. dfg + mlp + hybrid(ridgecol, pca) + 64  ->  'dfg_mlp_hybrid_ridgecol_pca_64'
         era812 + linear + pca + 32              ->  'era812_linear_pca_32'
    """
    parts = [adapter_name, predictor_config.model_type]
    if basis_config.basis_type == "hybrid":
        ep = basis_config.extra_params or {}
        parts += ["hybrid", ep.get("supervised_method", "sup"), ep.get("unsupervised_method", "unsup")]
    else:
        parts.append(basis_config.basis_type)
    parts.append(str(basis_config.n_components))
    return "_".join(str(p) for p in parts)


def split_adapter(adapter_name: str) -> "tuple[str, str]":
    """Split an adapter name into (dataset, task) for the runs/<dataset>/<task>/ layout.

    Task comes from the registered adapter's TASK class attr (ds000030_<task>,
    aomic_piop1_<task>); single-task datasets (dfg, era812) get task='main'.
    Falls back to a prefix heuristic for old/unregistered configs.
    """
    task = None
    try:  # lazy import to avoid a circular import at module load
        from cog2bold.datasets.registry import list_adapters
        cls = list_adapters().get(adapter_name)
        task = getattr(cls, "TASK", None) if cls is not None else None
    except Exception:  # noqa: BLE001
        task = None
    if not task:
        for pre in ("ds000030", "aomic_piop1"):
            if adapter_name.startswith(pre + "_"):
                return pre, adapter_name[len(pre) + 1:]
        return adapter_name, "main"
    suffix = f"_{task}"
    dataset = adapter_name[:-len(suffix)] if adapter_name.endswith(suffix) else adapter_name
    return dataset, task


def default_run_dir(adapter_name: str, basis_config, predictor_config) -> Path:
    """`runs_root()/<dataset>/<task>/<default_run_name(...)>`."""
    dataset, task = split_adapter(adapter_name)
    return (runs_root() / dataset / task
            / default_run_name(adapter_name, basis_config, predictor_config))
