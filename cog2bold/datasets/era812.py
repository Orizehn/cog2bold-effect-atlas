"""ERA812 dataset adapter.

Reading study with two scan cohorts (Epilepsy patients and an ages-8-12 group).
Use this as a second worked example of a `DatasetAdapter`; see also `dfg.py`.

On-disk layout (rooted at `data_root`, e.g. .../Research/datasets/ERA812):

    data_root/
      Epilepsy_dataset_behavioural.csv       # cognitive scores; 'Subject' ID column
      glm_activation_maps/
        Group_Epilepsy_Subject_<N>/          # one folder per scanned subject
          mask.nii                            # per-subject SPM analysis mask
          zmap_01_Checkerboard.nii            # modelled stimuli: Checkerboard,
          zmap_04_Non-words.nii               #   Non-words, Words
          zmap_05_Words.nii
          zmap_02_Cross_Start.nii             # present but NOT modelled (task structure,
          zmap_03_Fixation_End.nii            #   not real stimuli) — omitted
          zmap_contrast_Words_vs_Checkerboard.nii
          zmap_contrast_Words_vs_Non-words.nii
        Group_8_12_Subject_<N>/ ...

Subject join (verified against the data):
    Group_Epilepsy_Subject_<N>  ->  CSV Subject == "<N>"        (integer, no dot)
    Group_8_12_Subject_<N>      ->  CSV Subject == "<N>.1"
The two cohorts are DISTINCT people whose folder numbers overlap, so subject_ids
are made globally unique (Epilepsy: N, 8_12: 100000 + N). There are no sessions,
so session_id is constant (1) — a (subject, session) unit degenerates to a subject.

Cognitive features: the requested ZRead reading columns, plus a one-hot encoding
of `Group_New` (1=Healthy, 2=Epilepsy, 3=RD, 4=ADHD) appended as 4 extra dims.

Shared mask: each subject has its own mask.nii (all share one image grid). We use
the INTERSECTION of all subject masks (voxels analysed in everyone), computed once
and cached in the Cog2BOLD workspace (never written into the dataset).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cog2bold.datasets.base import DatasetAdapter, Record
from cog2bold.workspace import adapter_cache_dir


SUBJECT_COL = "Subject"
GROUP_COL = "Group_New"

# Stimulus labels we model. `Cross_Start` and `Fixation_End` exist as z-maps in the
# data but are task structure, not real stimuli, so they are intentionally omitted
# (and never form contrasts).
STIM_LABELS = ["Checkerboard", "Non-words", "Words"]

# Default reading-score feature columns (the user's requested features).
#DEFAULT_COG_COLS = [
#    "ZRead_Aloud_Punc_text_WPM_standard",
#    "ZRead_Aloud_Punc_text_perc_mistakes_standard",
#    "ZRead_Aloud_NotPunc_text_WPM_standard",
#    "ZRead_Aloud_NotPunc_text_perc_mistakes_standard",
#]

DEFAULT_COG_COLS = [
    "Age_AtTest_Years",
    "Child_WordReading_WPM_standard",
    "Child_nonWordReading_WPM_standard",
    #"Syllable_Omit_SD",
    #"Coding_WISC_SD",
    "Symbol_S_WISC_SD",
    "Memory_For_Digits_WISC_SD",
    "Diamond_Time_SD",
    "Naming_Obj_RAN_time_SD",
    "Naming_LO_RAS_pace_SD",
    #"TEA_CH_VisAtt_CorPerSec_AgeScale_C",
    #"TEA_CH_AtteScore_AgeScale",
    #"DKEFS_ColorWord_Condition4_CorrectedErrors_StandardScore",

]

# Group_New code -> one-hot position / readable name.
GROUP_NEW_LABELS = {1: "Healthy", 2: "Epilepsy", 3: "RD", 4: "ADHD"}

# subject_id namespace offset so Epilepsy_N and 8_12_N don't collide.
_GROUP_OFFSET = {"Epilepsy": 0, "8_12": 1000}


class ERA812Adapter(DatasetAdapter):
    name = "era812"

    def __init__(
        self,
        data_root: Path,
        cog_base_cols: Optional[List[str]] = None,
        include_group_onehot: bool = True,
        include_groups: Optional[List] = None,
    ):
        self.data_root = Path(data_root)
        self.cog_base_cols = list(cog_base_cols) if cog_base_cols else list(DEFAULT_COG_COLS)
        self.include_group_onehot = bool(include_group_onehot)

        self.maps_root = self.data_root / "glm_activation_maps"
        csv_path = self.data_root / "Epilepsy_dataset_behavioural.csv"
        if not self.maps_root.exists():
            raise FileNotFoundError(f"ERA812 maps root not found: {self.maps_root}")
        if not csv_path.exists():
            raise FileNotFoundError(f"ERA812 behavioural CSV not found: {csv_path}")

        self._df = pd.read_csv(csv_path, dtype={SUBJECT_COL: str})
        if SUBJECT_COL not in self._df.columns:
            raise ValueError(f"'{SUBJECT_COL}' column missing from {csv_path}")
        self._row_by_subject: Dict[str, pd.Series] = {
            str(row[SUBJECT_COL]).strip(): row for _, row in self._df.iterrows()
        }

        # Optional clinical-group restriction (by Group_New). Accepts names
        # ("Healthy", "ADHD") or integer codes; None = keep every group.
        self.include_groups: Optional[set] = self._resolve_groups(include_groups)

        self._mask_path = adapter_cache_dir(self.name) / "intersection_mask.nii.gz"
        self._mask_bool: Optional[np.ndarray] = None  # cached boolean array

    @staticmethod
    def _resolve_groups(include_groups: Optional[List]) -> Optional[set]:
        if not include_groups:
            return None
        name2code = {v.lower(): k for k, v in GROUP_NEW_LABELS.items()}
        codes = set()
        for g in include_groups:
            if isinstance(g, str):
                key = g.strip().lower()
                if key not in name2code:
                    raise ValueError(
                        f"Unknown Group_New {g!r}; expected one of {list(GROUP_NEW_LABELS.values())}"
                    )
                codes.add(name2code[key])
            else:
                code = int(g)
                if code not in GROUP_NEW_LABELS:
                    raise ValueError(f"Group_New code {code} not in {sorted(GROUP_NEW_LABELS)}")
                codes.add(code)
        return codes

    def _group_code_of(self, subject_id: int) -> Optional[int]:
        """Group_New code for a subject_id, or None if missing/unparseable."""
        row = self._row_by_subject.get(self._csv_key(subject_id))
        if row is None:
            return None
        try:
            return int(float(row.get(GROUP_COL)))
        except (TypeError, ValueError):
            return None

    # Make the materialise cache distinguish group filter + one-hot toggle, so a
    # full-dataset cache is never reused for a group-restricted run (and vice-versa).
    def _cache_key(self, **kwargs) -> str:
        base = super()._cache_key(**kwargs)
        extra = (
            f"groups={sorted(self.include_groups) if self.include_groups else 'all'}"
            f"_oh={int(self.include_group_onehot)}"
        )
        import hashlib
        return hashlib.md5((base + extra).encode("utf-8")).hexdigest()[:16]

    # ---- DatasetAdapter API ----

    @classmethod
    def default_data_root(cls) -> Path:
        env = os.environ.get("ERA812_DATA_ROOT")
        return Path(env) if env else (Path(__file__).resolve().parents[3] / "datasets" / "ERA812")

    @classmethod
    def default_cog_base_cols(cls) -> List[str]:
        return list(DEFAULT_COG_COLS)

    @property
    def global_labels(self) -> List[str]:
        return list(STIM_LABELS)

    @property
    def cog_feature_names(self) -> List[str]:
        names = list(self.cog_base_cols)
        if self.include_group_onehot:
            names += [f"Group:{GROUP_NEW_LABELS[k]}" for k in sorted(GROUP_NEW_LABELS)]
        return names

    def list_records(self) -> List[Record]:
        records: List[Record] = []
        for folder in sorted(self.maps_root.glob("Group_*_Subject_*")):
            if not folder.is_dir():
                continue
            m = re.match(r"Group_(.+)_Subject_(\d+)$", folder.name)
            if not m:
                continue
            group, n = m.group(1), int(m.group(2))
            if group not in _GROUP_OFFSET:
                continue
            subject_id = _GROUP_OFFSET[group] + n

            # clinical-group restriction (by Group_New from the CSV)
            if self.include_groups is not None:
                code = self._group_code_of(subject_id)
                if code is None or code not in self.include_groups:
                    continue

            # individual stimuli — find by label name (filename has a numeric
            # prefix we don't rely on); exclude contrast files.
            for stim_id, label in enumerate(STIM_LABELS):
                matches = [
                    f for f in folder.glob(f"zmap_*_{label}.nii")
                    if "contrast" not in f.name
                ]
                if matches:
                    records.append(Record(
                        subject_id=subject_id, session_id=1, stim_id=stim_id,
                        is_contrast=False, contrast_name=None, zmap_path=matches[0],
                    ))

            # contrasts: parse zmap_contrast_<A>_vs_<B>.nii -> "A>B"
            for f in sorted(folder.glob("zmap_contrast_*.nii")):
                cname = self._infer_contrast_name(f.name)
                if cname is not None:
                    records.append(Record(
                        subject_id=subject_id, session_id=1, stim_id=-1,
                        is_contrast=True, contrast_name=cname, zmap_path=f,
                    ))
        return records

    def load_zmap(self, record: Record) -> np.ndarray:
        from nilearn import image

        mask = self._mask_bool_array()
        img = image.load_img(str(record.zmap_path))
        data = img.get_fdata()
        return data[mask].astype(np.float32)

    def get_cog_vector(self, subject_id: int, session_id: int) -> np.ndarray:
        key = self._csv_key(subject_id)
        row = self._row_by_subject.get(key)
        if row is None:
            raise ValueError(f"No CSV row for subject_id={subject_id} (key={key!r})")

        vals: List[float] = []
        for col in self.cog_base_cols:
            if col not in row.index:
                raise KeyError(f"Cog column {col!r} not in ERA812 CSV")
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
            vals.extend(self._group_onehot(row))
        return np.asarray(vals, dtype=np.float64)

    def mask_path(self) -> Path:
        self._mask_bool_array()  # ensure the file is materialised
        return self._mask_path

    def get_masker(self):
        """Build a fresh NiftiMasker from the cached intersection mask."""
        from nilearn.maskers import NiftiMasker

        self._mask_bool_array()  # ensure cached file exists
        from nilearn import image

        masker = NiftiMasker(mask_img=image.load_img(str(self._mask_path)))
        masker.fit()
        return masker

    # ---- helpers ----

    def _csv_key(self, subject_id: int) -> str:
        """Recover the CSV `Subject` string from a globally-unique subject_id."""
        if subject_id >= _GROUP_OFFSET["8_12"]:
            return f"{subject_id - _GROUP_OFFSET['8_12']}.1"   # 8_12 cohort
        return str(subject_id)                                  # Epilepsy cohort

    def _group_onehot(self, row: pd.Series) -> List[float]:
        raw = row.get(GROUP_COL)
        try:
            code = int(float(raw))
        except (TypeError, ValueError):
            raise ValueError(f"Bad {GROUP_COL}={raw!r}")
        if code not in GROUP_NEW_LABELS:
            raise ValueError(f"{GROUP_COL}={code} outside known {sorted(GROUP_NEW_LABELS)}")
        return [1.0 if k == code else 0.0 for k in sorted(GROUP_NEW_LABELS)]

    def _infer_contrast_name(self, filename: str) -> Optional[str]:
        """`zmap_contrast_Words_vs_Checkerboard.nii` -> 'Words>Checkerboard'."""
        m = re.match(r"zmap_contrast_(.+)_vs_(.+)\.nii$", filename)
        if not m:
            return None
        a, b = m.group(1), m.group(2)
        if a in STIM_LABELS and b in STIM_LABELS:
            return f"{a}>{b}"
        return None

    def _mask_bool_array(self) -> np.ndarray:
        """Intersection of all subject masks (cached in-memory + on disk)."""
        if self._mask_bool is not None:
            return self._mask_bool

        from nilearn import image

        if self._mask_path.exists():
            self._mask_bool = image.load_img(str(self._mask_path)).get_fdata().astype(bool)
            return self._mask_bool

        print(f"[era812] computing intersection mask over all subjects -> {self._mask_path}")
        inter = None
        ref_img = None
        for folder in sorted(self.maps_root.glob("Group_*_Subject_*")):
            mpath = folder / "mask.nii"
            if not mpath.exists():
                continue
            mimg = image.load_img(str(mpath))
            mdata = mimg.get_fdata() > 0
            if inter is None:
                inter = mdata
                ref_img = mimg
            else:
                if mdata.shape != inter.shape:
                    print(f"[era812] WARN: {folder.name} mask shape {mdata.shape} "
                          f"!= {inter.shape}; skipping it from the intersection")
                    continue
                inter &= mdata
        if inter is None:
            raise FileNotFoundError("No subject mask.nii files found to build a mask")

        mask_img = image.new_img_like(ref_img, inter.astype(np.uint8))
        mask_img.to_filename(str(self._mask_path))
        print(f"[era812] intersection mask: {int(inter.sum())} voxels")
        self._mask_bool = inter
        return self._mask_bool
