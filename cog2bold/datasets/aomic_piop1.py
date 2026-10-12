"""AOMIC PIOP1 (ds002785) dataset adapter.

AOMIC PIOP1 has FIVE task-fMRI paradigms. Unlike DFG/ERA812, each task is a
self-contained mini-dataset with its own conditions, contrasts, and the cognitive
features that best couple to it. We therefore treat **each task as its own
dataset**: one adapter class, registered under five names
(`aomic_piop1_workingmemory`, `aomic_piop1_gstroop`, `aomic_piop1_emomatching`,
`aomic_piop1_anticipation`, `aomic_piop1_faces`). The active task is held in the
`task` variable (default `DEFAULT_TASK`); loading several tasks together is a
possible future extension, not supported here.

On-disk layout (rooted at `data_root`, e.g. .../Research/datasets/AOMICPIOP1):

    data_root/
      ds002785_behavioral_features.csv     # cognitive + behavioral scores; 'Subject' = sub-XXXX
      actmaps/
        sub-XXXX/
          sub-XXXX_task-<task>_cond-<cond>_zmap.nii.gz          # per-condition z-maps
          sub-XXXX_task-<task>_contrast-<name>_zmap.nii.gz      # contrast z-maps

There are no sessions (session_id is constant 1). The z-maps are first-level GLM
outputs in MNI152NLin2009cAsym; voxels outside each subject's fMRIPrep brain mask
are exactly 0. No per-subject mask files were saved, so the shared brain mask is
computed as the INTERSECTION across subjects of each subject's in-brain voxels
(OR over that task's condition maps of `data != 0`), cached in the workspace.

Cognitive features come from `ds002785_behavioral_features.csv` (see the
behavioral-features docs). For the two PASSIVE tasks (anticipation, faces) there
are no in-scanner behavioral metrics, so the defaults fall back to the static
NEO/age scores. All defaults are overridable via `cog_base_cols`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cog2bold.datasets.base import DatasetAdapter, Record
from cog2bold.datasets.registry import register_adapter
from cog2bold.workspace import adapter_cache_dir


SUBJECT_COL = "Subject"

# The dedicated variable indicating which task we are working on.
DEFAULT_TASK = "workingmemory"

# Per-task specification: condition order (= global_labels / stim_id order),
# contrast weight maps (over the conditions), and default cognitive features.
# Default features are the 3-7 most task-coupled columns (see the feature docs);
# the sparser *_pes columns are strong too but reduce usable N, so they are left
# out of the defaults — add them via cog_base_cols if you want them.
TASK_SPECS: Dict[str, Dict] = {
    "workingmemory": {
        "conditions": ["active_change", "active_nochange", "passive"],
        "contrasts": {
            "active_gt_passive": {"active_change": 0.5, "active_nochange": 0.5, "passive": -1.0},
            "change_gt_nochange": {"active_change": 1.0, "active_nochange": -1.0},
        },
        "default_cog_cols": [
            "wm_dprime", "wm_active_acc", "wm_active_rt_mean", "wm_active_ies", "raven_score", "age"
        ],
    },
    "gstroop": {
        "conditions": ["congruent", "incongruent"],
        "contrasts": {
            "incongruent_gt_congruent": {"incongruent": 1.0, "congruent": -1.0},
        },
        "default_cog_cols": [
            "gstroop_rt_interference", "gstroop_ies_interference",
            "gstroop_incongruent_rt_mean", "gstroop_overall_acc", "raven_score", "age"
        ],
    },
    "emomatching": {
        "conditions": ["emotion", "control"],
        "contrasts": {
            "emotion_gt_control": {"emotion": 1.0, "control": -1.0},
        },
        "default_cog_cols": [
            "emo_rt_diff", "emo_ies_diff", "emo_emotion_rt_mean", "emo_overall_acc", "NEO_N", "raven_score", "age"
        ],
    },
    "anticipation": {  # passive task — no in-scanner behavior, static defaults
        "conditions": ["cue_negative", "cue_neutral", "img_negative", "img_neutral"],
        "contrasts": {
            "cue_neg_gt_neu": {"cue_negative": 1.0, "cue_neutral": -1.0},
            "img_neg_gt_neu": {"img_negative": 1.0, "img_neutral": -1.0},
        },
        "default_cog_cols": ["NEO_N", "NEO_E", "NEO_C","raven_score", "age"],
    },
    "faces": {  # passive task — no in-scanner behavior, static defaults
        "conditions": ["anger", "contempt", "joy", "neutral", "pride"],
        "contrasts": {
            "emotion_gt_neutral": {
                "anger": 0.25, "contempt": 0.25, "joy": 0.25, "pride": 0.25, "neutral": -1.0
            },
        },
        "default_cog_cols": ["NEO_N", "NEO_E", "NEO_A","raven_score", "age"],
    },
}

TASKS = list(TASK_SPECS.keys())


class AOMICPIOP1Adapter(DatasetAdapter):
    """Adapter for one AOMIC PIOP1 task (selected via `task`)."""

    name = "aomic_piop1"
    TASK: Optional[str] = None  # set on the per-task subclasses created below

    @classmethod
    def default_data_root(cls) -> Path:
        return Path(__file__).resolve().parents[3] / "datasets" / "AOMICPIOP1"

    @classmethod
    def default_cog_base_cols(cls) -> List[str]:
        task = cls.TASK or DEFAULT_TASK
        return list(TASK_SPECS[task]["default_cog_cols"])

    def __init__(
        self,
        data_root: Path,
        cog_base_cols: Optional[List[str]] = None,
        task: Optional[str] = None,
    ):
        self.task = task or self.TASK or DEFAULT_TASK
        if self.task not in TASK_SPECS:
            raise ValueError(f"Unknown AOMIC task {self.task!r}; choose from {TASKS}")
        # Per-task identity so caches and the brain mask never collide across tasks.
        self.name = f"aomic_piop1_{self.task}"

        self.data_root = Path(data_root)
        self.cog_base_cols = (
            list(cog_base_cols) if cog_base_cols
            else list(TASK_SPECS[self.task]["default_cog_cols"])
        )

        self.maps_root = self.data_root / "actmaps"
        csv_path = self.data_root / "ds002785_behavioral_features.csv"
        if not self.maps_root.exists():
            raise FileNotFoundError(f"AOMIC actmaps root not found: {self.maps_root}")
        if not csv_path.exists():
            raise FileNotFoundError(f"AOMIC behavioral CSV not found: {csv_path}")

        self._df = pd.read_csv(csv_path, dtype={SUBJECT_COL: str})
        if SUBJECT_COL not in self._df.columns:
            raise ValueError(f"'{SUBJECT_COL}' column missing from {csv_path}")
        self._row_by_subject: Dict[str, pd.Series] = {
            str(row[SUBJECT_COL]).strip(): row for _, row in self._df.iterrows()
        }

        self._conditions = list(TASK_SPECS[self.task]["conditions"])
        self._contrasts = dict(TASK_SPECS[self.task]["contrasts"])

        self._mask_path = adapter_cache_dir(self.name) / "intersection_mask.nii.gz"
        self._mask_bool: Optional[np.ndarray] = None  # cached boolean array

    # ---- DatasetAdapter API ----

    @property
    def global_labels(self) -> List[str]:
        return list(self._conditions)

    def list_records(self) -> List[Record]:
        records: List[Record] = []
        for subj_dir in sorted(self.maps_root.glob("sub-*")):
            if not subj_dir.is_dir():
                continue
            try:
                subject_id = int(subj_dir.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            sub = subj_dir.name  # "sub-0001"

            # Per-condition z-maps (individual stimuli, one-hot encoded)
            for stim_id, cond in enumerate(self._conditions):
                f = subj_dir / f"{sub}_task-{self.task}_cond-{cond}_zmap.nii.gz"
                if f.exists():
                    records.append(Record(
                        subject_id=subject_id, session_id=1, stim_id=stim_id,
                        is_contrast=False, contrast_name=None, zmap_path=f,
                    ))

            # Contrast z-maps (discovered by the GLM contrast name in the filename)
            for f in sorted(subj_dir.glob(f"{sub}_task-{self.task}_contrast-*_zmap.nii.gz")):
                cname = f.name.split("_contrast-", 1)[1].rsplit("_zmap", 1)[0]
                if cname in self._contrasts:
                    records.append(Record(
                        subject_id=subject_id, session_id=1, stim_id=-1,
                        is_contrast=True, contrast_name=cname, zmap_path=f,
                    ))
        return records

    def load_zmap(self, record: Record) -> np.ndarray:
        from nilearn import image

        mask = self._mask_bool_array()
        data = image.load_img(str(record.zmap_path)).get_fdata()
        return data[mask].astype(np.float32)

    def get_cog_vector(self, subject_id: int, session_id: int) -> np.ndarray:
        key = f"sub-{subject_id:04d}"
        row = self._row_by_subject.get(key)
        if row is None:
            raise ValueError(f"No behavioral row for subject_id={subject_id} (key={key!r})")
        vals: List[float] = []
        for col in self.cog_base_cols:
            if col not in row.index:
                raise KeyError(f"Cog column {col!r} not in AOMIC behavioral CSV")
            v = row[col]
            if pd.isna(v) or str(v).strip() == "":
                raise ValueError(f"Missing cog value subj={subject_id} col={col!r}")
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"Non-numeric cog value subj={subject_id} col={col!r}")
            if not np.isfinite(f):
                raise ValueError(f"Non-finite cog value subj={subject_id} col={col!r}")
            vals.append(f)
        return np.asarray(vals, dtype=np.float64)

    def mask_path(self) -> Path:
        self._mask_bool_array()  # ensure the file is materialised
        return self._mask_path

    def get_masker(self):
        """Build a fresh NiftiMasker from the cached intersection mask."""
        from nilearn import image
        from nilearn.maskers import NiftiMasker

        self._mask_bool_array()  # ensure cached file exists
        masker = NiftiMasker(mask_img=image.load_img(str(self._mask_path)))
        masker.fit()
        return masker

    # ---- contrast encoding ----

    def _parse_contrast_name(self, contrast_name: str) -> np.ndarray:
        """Encode an AOMIC contrast (e.g. 'active_gt_passive') as a weight vector.

        Uses the explicit per-task weight map so composite contrasts (more than
        one condition per side) are supported; falls back to the base 'a>b'
        parser for anything not in TASK_SPECS.
        """
        weights = self._contrasts.get(contrast_name)
        if weights is None:
            return super()._parse_contrast_name(contrast_name)
        enc = np.zeros(len(self._conditions), dtype=np.float32)
        idx = {c: i for i, c in enumerate(self._conditions)}
        for cond, w in weights.items():
            enc[idx[cond]] = float(w)
        return enc

    # ---- mask helpers ----

    def _mask_bool_array(self) -> np.ndarray:
        """Intersection across subjects of in-brain voxels (cached on disk)."""
        if self._mask_bool is not None:
            return self._mask_bool

        from nilearn import image

        if self._mask_path.exists():
            self._mask_bool = image.load_img(str(self._mask_path)).get_fdata().astype(bool)
            return self._mask_bool

        print(f"[{self.name}] computing intersection brain mask -> {self._mask_path}")
        inter = None
        ref_img = None
        for subj_dir in sorted(self.maps_root.glob("sub-*")):
            sub = subj_dir.name
            cond_files = [
                subj_dir / f"{sub}_task-{self.task}_cond-{c}_zmap.nii.gz"
                for c in self._conditions
            ]
            cond_files = [f for f in cond_files if f.exists()]
            if not cond_files:
                continue
            # subject in-brain = OR over this task's condition maps of (data != 0)
            subj_mask = None
            for f in cond_files:
                img = image.load_img(str(f))
                m = img.get_fdata() != 0
                subj_mask = m if subj_mask is None else (subj_mask | m)
                if ref_img is None:
                    ref_img = img
            if inter is None:
                inter = subj_mask
            elif subj_mask.shape == inter.shape:
                inter &= subj_mask
            else:
                print(f"[{self.name}] WARN: {sub} map shape {subj_mask.shape} "
                      f"!= {inter.shape}; skipping from intersection")
        if inter is None:
            raise FileNotFoundError(f"No condition z-maps found under {self.maps_root}")

        mask_img = image.new_img_like(ref_img, inter.astype(np.uint8))
        mask_img.to_filename(str(self._mask_path))
        print(f"[{self.name}] intersection mask: {int(inter.sum())} voxels")
        self._mask_bool = inter
        return self._mask_bool


def register_aomic() -> None:
    """Register one adapter per AOMIC task: aomic_piop1_<task>."""
    for task in TASKS:
        cls = type(
            f"AOMICPIOP1_{task}",
            (AOMICPIOP1Adapter,),
            {"name": f"aomic_piop1_{task}", "TASK": task},
        )
        register_adapter(cls.name, cls)
