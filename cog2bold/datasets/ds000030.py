"""ds000030 (UCLA CNP, LA5c) dataset adapter.

UCLA CNP has several task-fMRI paradigms. Like AOMIC PIOP1, we treat **each task as
its own dataset**: one adapter class registered per task (`ds000030_scap`,
`ds000030_stopsignal`, `ds000030_taskswitch`, `ds000030_bart`, `ds000030_pamret`).
The active task is held in `TASK` (default `DEFAULT_TASK`). A new task = one more
`TASK_SPECS` entry (+ its maps under actmaps/, laid out by build_ds000030_alltasks_maps.py).

On-disk layout (rooted at `data_root`, e.g. .../Research/datasets/UCLA_CNP):

    data_root/
      ds000030_behavioral_features.csv         # curated WM + controls; 'Subject' = sub-XXXXX
      ds000030_behavioral_features_broad.csv    # broad phenotype battery (map_type-independent)
      actmaps/                                   # z-stat maps (primary; AOMIC _zmap convention)
        sub-XXXXX/
          sub-XXXXX_task-scap_cond-<cond>_zmap.nii.gz
          sub-XXXXX_task-scap_contrast-<name>_zmap.nii.gz
      actmaps_cope/                              # cope (effect-size) maps, _cope suffix
        sub-XXXXX/ sub-XXXXX_task-scap_cond-<cond>_cope.nii.gz ...

Subject ids are 5-digit and unique across diagnostic groups (1xxxx control, 5xxxx
schizophrenia, 6xxxx bipolar, 7xxxx ADHD), so `subject_id = int(nnnnn)` with no
zero-padding and no group offset — the one real difference from the AOMIC adapter.

There are no sessions (session_id constant 1). The maps are FEAT first-level outputs
run on fMRIPrep-normalized BOLD in MNI152NLin2009cAsym; out-of-brain voxels are 0.
No per-subject mask files, so the shared mask is the INTERSECTION across subjects of
in-brain voxels (OR over the task's condition maps of `data != 0`), cached in the
workspace. `map_type` ("zstat"|"cope") selects which map folder/suffix to load.
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
DEFAULT_TASK = "scap"
# Diagnosis groups, fixed order for the optional one-hot (stable feature columns).
DIAG_GROUPS = ["ADHD", "BIPOLAR", "CONTROL", "SCHZ"]

# Per-task specification: condition order (= global_labels / stim_id order), contrast
# weight maps (over the conditions), and default cognitive features. Adding another CNP
# task = one more entry here (+ its maps under actmaps/). The `linearup_load` weights
# match FEAT's parametric load contrast; `load7_gt_load1` is the high-low difference.
TASK_SPECS: Dict[str, Dict] = {
    "scap": {
        "conditions": ["load1", "load3", "load5", "load7"],
        "contrasts": {
            "linearup_load": {"load1": -3.0, "load3": -1.0, "load5": 1.0, "load7": 3.0},
            "load7_gt_load1": {"load7": 1.0, "load1": -1.0},
        },
        # Default cog vector: clean independent WM (ssp/ds/lns) + in-task capacity +
        # an IQ control + age. Categorical cols (gender/diagnosis) are intentionally
        # excluded from defaults (non-numeric); add via cog_base_cols if encoded.
        "default_cog_cols": [
            "ssp_totalraw", "ds_totalraw", "lns_totalraw",
            "scap_max_capac", "mr_totalraw", "age",
        ],
    },
    "stopsignal": {  # response inhibition
        "conditions": ["go", "stopsuccess", "stopunsuccess"],
        "contrasts": {
            "stopsuccess_gt_go": {"stopsuccess": 1.0, "go": -1.0},
            "stopunsuccess_gt_go": {"stopunsuccess": 1.0, "go": -1.0},
        },
        "default_cog_cols": ["sst_ses_ssrt", "sst_ses_mean_rt", "mr_totalraw", "age"],
    },
    "taskswitch": {  # cognitive flexibility (congruency x switch x ISI cells)
        "conditions": [
            "cong_switch_short", "cong_switch_long", "cong_noswitch_short", "cong_noswitch_long",
            "incong_switch_short", "incong_switch_long", "incong_noswitch_short", "incong_noswitch_long",
        ],
        "contrasts": {
            "switch_gt_noswitch": {
                "cong_switch_short": 0.25, "cong_switch_long": 0.25,
                "incong_switch_short": 0.25, "incong_switch_long": 0.25,
                "cong_noswitch_short": -0.25, "cong_noswitch_long": -0.25,
                "incong_noswitch_short": -0.25, "incong_noswitch_long": -0.25,
            },
            "incong_gt_cong": {
                "incong_switch_short": 0.25, "incong_switch_long": 0.25,
                "incong_noswitch_short": 0.25, "incong_noswitch_long": 0.25,
                "cong_switch_short": -0.25, "cong_switch_long": -0.25,
                "cong_noswitch_short": -0.25, "cong_noswitch_long": -0.25,
            },
        },
        "default_cog_cols": ["ts_interference", "ts_costlong", "mr_totalraw", "age"],
    },
    "bart": {  # risk taking (balloon analogue risk task)
        "conditions": ["accept", "control", "explode", "reject"],
        "contrasts": {
            "explode_gt_reject": {"explode": 1.0, "reject": -1.0},
        },
        "default_cog_cols": ["bart_meanadjustedpumps", "bart_blueexplosions", "age"],
    },
    "pamret": {  # paired-associate memory retrieval
        "conditions": ["correct", "incorrect", "control"],
        "contrasts": {
            "correct_gt_control": {"correct": 1.0, "control": -1.0},
            "correct_gt_incorrect": {"correct": 1.0, "incorrect": -1.0},
        },
        # No in-scanner pamret phenotype table -> matched out-of-scanner memory (CVLT).
        "default_cog_cols": ["cvlt_totcor", "cvlt_ldf", "ds_totalraw", "age"],
    },
}

TASKS = list(TASK_SPECS.keys())

_MAP_KIND = {  # map_type -> (actmaps subdir, filename suffix)
    "zstat": ("actmaps", "zmap"),
    "cope": ("actmaps_cope", "cope"),
}


class DS000030Adapter(DatasetAdapter):
    """Adapter for one UCLA CNP (ds000030) task (selected via `task`)."""

    name = "ds000030"
    TASK: Optional[str] = None  # set on the per-task subclasses created below

    @classmethod
    def default_data_root(cls) -> Path:
        return Path(__file__).resolve().parents[3] / "datasets" / "UCLA_CNP"

    @classmethod
    def default_cog_base_cols(cls) -> List[str]:
        task = cls.TASK or DEFAULT_TASK
        return list(TASK_SPECS[task]["default_cog_cols"])

    def __init__(
        self,
        data_root: Path,
        cog_base_cols: Optional[List[str]] = None,
        task: Optional[str] = None,
        map_type: str = "zstat",
        broad: bool = False,
        groups: Optional[List[str]] = None,
        include_group_onehot: bool = False,
    ):
        self.task = task or self.TASK or DEFAULT_TASK
        if self.task not in TASK_SPECS:
            raise ValueError(f"Unknown ds000030 task {self.task!r}; choose from {TASKS}")
        if map_type not in _MAP_KIND:
            raise ValueError(f"map_type must be one of {sorted(_MAP_KIND)}, got {map_type!r}")
        self.map_type = map_type
        self._maps_subdir, self._suffix = _MAP_KIND[map_type]
        self.include_group_onehot = bool(include_group_onehot)

        # Per-task + per-map-type identity so caches/masks never collide.
        base_name = f"ds000030_{self.task}" + ("_cope" if map_type == "cope" else "")
        # Optional diagnosis-group filter (CONTROL / SCHZ / BIPOLAR / ADHD). None = all.
        self.groups = [g.strip().upper() for g in groups] if groups else None
        self.name = base_name + (f"_g{'+'.join(sorted(self.groups))}" if self.groups else "") \
            + ("_diagoh" if self.include_group_onehot else "")

        self.data_root = Path(data_root)
        self.cog_base_cols = (
            list(cog_base_cols) if cog_base_cols
            else list(TASK_SPECS[self.task]["default_cog_cols"])
        )

        self.maps_root = self.data_root / self._maps_subdir
        csv_name = ("ds000030_behavioral_features_broad.csv" if broad
                    else "ds000030_behavioral_features.csv")
        csv_path = self.data_root / csv_name
        if not self.maps_root.exists():
            raise FileNotFoundError(f"ds000030 maps root not found: {self.maps_root}")
        if not csv_path.exists():
            raise FileNotFoundError(f"ds000030 behavioral CSV not found: {csv_path}")

        self._df = pd.read_csv(csv_path, dtype={SUBJECT_COL: str})
        if SUBJECT_COL not in self._df.columns:
            raise ValueError(f"'{SUBJECT_COL}' column missing from {csv_path}")
        self._row_by_subject: Dict[str, pd.Series] = {
            str(row[SUBJECT_COL]).strip(): row for _, row in self._df.iterrows()
        }
        # subject_id (int) -> diagnosis (upper), for the group filter in list_records
        self._diag_by_sid: Dict[int, str] = {}
        if "diagnosis" in self._df.columns:
            for _, row in self._df.iterrows():
                try:
                    sid = int(str(row[SUBJECT_COL]).split("-")[1])
                except (IndexError, ValueError):
                    continue
                self._diag_by_sid[sid] = str(row["diagnosis"]).strip().upper()
        if self.groups and not self._diag_by_sid:
            raise ValueError("groups filter requested but the behavioral CSV has no 'diagnosis' column")

        self._conditions = list(TASK_SPECS[self.task]["conditions"])
        self._contrasts = dict(TASK_SPECS[self.task]["contrasts"])

        # Brain mask is group-INDEPENDENT (same voxel space for every subset) so it is
        # computed once per task/map-type and reused across group filters.
        self._mask_path = adapter_cache_dir(base_name) / "intersection_mask.nii.gz"
        self._mask_bool: Optional[np.ndarray] = None

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
                subject_id = int(subj_dir.name.split("-")[1])  # 5-digit CNP id, no padding
            except (IndexError, ValueError):
                continue
            if self.groups is not None and self._diag_by_sid.get(subject_id) not in self.groups:
                continue  # skip subjects outside the requested diagnosis group(s)
            sub = subj_dir.name  # "sub-10159"

            # Per-condition maps (individual stimuli, one-hot encoded)
            for stim_id, cond in enumerate(self._conditions):
                f = subj_dir / f"{sub}_task-{self.task}_cond-{cond}_{self._suffix}.nii.gz"
                if f.exists():
                    records.append(Record(
                        subject_id=subject_id, session_id=1, stim_id=stim_id,
                        is_contrast=False, contrast_name=None, zmap_path=f,
                    ))

            # Contrast maps (discovered by the GLM contrast name in the filename)
            for f in sorted(subj_dir.glob(
                    f"{sub}_task-{self.task}_contrast-*_{self._suffix}.nii.gz")):
                cname = f.name.split("_contrast-", 1)[1].rsplit(f"_{self._suffix}", 1)[0]
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
        key = f"sub-{subject_id}"  # CNP ids are unpadded 5-digit
        row = self._row_by_subject.get(key)
        if row is None:
            raise ValueError(f"No behavioral row for subject_id={subject_id} (key={key!r})")
        vals: List[float] = []
        for col in self.cog_base_cols:
            if col not in row.index:
                raise KeyError(f"Cog column {col!r} not in ds000030 behavioral CSV")
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
        if self.include_group_onehot:
            vals.extend(self._group_onehot(subject_id))
        return np.asarray(vals, dtype=np.float64)

    # diagnosis one-hot (optional extra features); fixed order for stable columns
    def _group_onehot(self, subject_id: int) -> List[float]:
        diag = self._diag_by_sid.get(subject_id)
        if diag not in DIAG_GROUPS:
            raise ValueError(f"subj={subject_id} has unknown diagnosis {diag!r} "
                             f"(need one of {DIAG_GROUPS})")
        return [1.0 if g == diag else 0.0 for g in DIAG_GROUPS]

    @property
    def cog_feature_names(self) -> List[str]:
        names = list(self.cog_base_cols)
        if self.include_group_onehot:
            names += [f"Group:{g}" for g in DIAG_GROUPS]
        return names

    def mask_path(self) -> Path:
        self._mask_bool_array()  # ensure the file is materialised
        return self._mask_path

    def get_masker(self):
        """Build a fresh NiftiMasker from the cached intersection mask."""
        from nilearn import image
        from nilearn.maskers import NiftiMasker

        self._mask_bool_array()
        masker = NiftiMasker(mask_img=image.load_img(str(self._mask_path)))
        masker.fit()
        return masker

    # ---- contrast encoding ----

    def _parse_contrast_name(self, contrast_name: str) -> np.ndarray:
        """Encode a SCAP contrast (e.g. 'linearup_load') as a weight vector over conditions."""
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
                subj_dir / f"{sub}_task-{self.task}_cond-{c}_{self._suffix}.nii.gz"
                for c in self._conditions
            ]
            cond_files = [f for f in cond_files if f.exists()]
            if not cond_files:
                continue
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
            raise FileNotFoundError(f"No condition maps found under {self.maps_root}")

        mask_img = image.new_img_like(ref_img, inter.astype(np.uint8))
        mask_img.to_filename(str(self._mask_path))
        print(f"[{self.name}] intersection mask: {int(inter.sum())} voxels")
        self._mask_bool = inter
        return self._mask_bool


def register_ds000030() -> None:
    """Register one adapter per CNP task: ds000030_<task> (currently just scap)."""
    for task in TASKS:
        cls = type(
            f"DS000030_{task}",
            (DS000030Adapter,),
            {"name": f"ds000030_{task}", "TASK": task},
        )
        register_adapter(cls.name, cls)
