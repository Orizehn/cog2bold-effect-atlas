"""Pickle / JSON helpers for run artefacts.

Saved per run:
    config.json          full RunConfig (reproducibility)
    basis.pkl            fitted basis
    predictor.pkl        trained predictor
    predictions.pkl      maps_gt / maps_pred / coeffs / metadata for every train+test sample
    metrics.json         scalar metrics
    contrasts/*.npy      contrast maps (optional)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict


def save_pickle(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
