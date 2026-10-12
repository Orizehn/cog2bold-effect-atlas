"""DFG dataset adapter.

This is the reference adapter — it ports all DFG-specific logic from the
legacy `fmri-stimXcog-model` codebase into one file. Use it as a template
when writing an adapter for a new dataset.

Expected on-disk layout (rooted at `data_root`, e.g. .../Research/datasets/DFG):

    data_root/
      behavioral/
        DFG_behavioural_results.csv         # cog scores per subject (one row per subject)
      metadata/
        session_*_cog2bold.pkl              # one pkl per session, contains meta['stimulus_labels']
        DFG_mask_MNI152NLin6Asym.nii.gz     # brain mask
      glm_first_level_zmaps_fmriprep/
        subject_<N>/
          session_<S>/
            sub-<N>_ses-<S:02d>_cond-condition <K>-*_zmap.nii.gz       # individual stim
            sub-<N>_ses-<S:02d>_contrast-condition <K>-*_vs_condition <L>-*_zmap.nii.gz   # contrasts

The behavioral CSV uses session-suffixed columns: `<base>#<session>` (e.g.
`NamingMixObj_Aleftaf_RawPerMin_Perc#1`). For each base column the loader prefers
the session-specific variant and falls back to the un-suffixed column.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cog2bold.datasets.base import DatasetAdapter, Record


SUBJECT_COL = "Subject"

# Default cognitive columns for the DFG dataset.
DEFAULT_DFG_COG_COLS = [
    "NamingMixObj_Aleftaf_RawPerMin",
    "NamingLetters_Shatil_Raw",
    "NamingLetters_Shatil_Time",
    "SkySearch_TEACH_Attention_SS",
    "Age_at_Scan",
]


class DFGAdapter(DatasetAdapter):
    name = "dfg"

    @classmethod
    def default_data_root(cls) -> Path:
        env = os.environ.get("FMRI_DATA_ROOT")
        return Path(env) if env else (Path(__file__).resolve().parents[3] / "datasets" / "DFG")

    @classmethod
    def default_cog_base_cols(cls) -> List[str]:
        return list(DEFAULT_DFG_COG_COLS)

    def __init__(self, data_root: Path, cog_base_cols: List[str]):
        self.data_root = Path(data_root)
        self.cog_base_cols = list(cog_base_cols)

        self.metadata_root = self.data_root / "metadata"
        self.fmriprep_root = self.data_root / "glm_first_level_zmaps_fmriprep"
        scores_csv = self.data_root / "behavioral" / "DFG_behavioural_results.csv"

        if not self.fmriprep_root.exists():
            raise FileNotFoundError(f"fMRIPrep z-map root not found: {self.fmriprep_root}")
        if not scores_csv.exists():
            raise FileNotFoundError(f"Behavioral CSV not found: {scores_csv}")

        self._df_behav = pd.read_csv(scores_csv)
        if SUBJECT_COL not in self._df_behav.columns:
            raise ValueError(
                f"Subject column '{SUBJECT_COL}' missing from {scores_csv}; "
                f"available: {list(self._df_behav.columns)}"
            )

        self._global_labels = self._discover_global_labels()
        self._mask_path = self.data_root / "metadata" / "DFG_mask_MNI152NLin6Asym.nii.gz"
        self._mask_data: Optional[np.ndarray] = None  # cached on first load

    # ---- DatasetAdapter API ----

    @property
    def global_labels(self) -> List[str]:
        return self._global_labels

    def mask_path(self) -> Path:
        return self._mask_path

    def list_records(self) -> List[Record]:
        records: List[Record] = []
        for subj_dir in sorted(self.fmriprep_root.glob("subject_*")):
            try:
                subj = int(subj_dir.name.split("_")[1])
            except ValueError:
                continue
            for sess_dir in sorted(subj_dir.glob("session_*")):
                try:
                    sess = int(sess_dir.name.split("_")[1])
                except ValueError:
                    continue
                # Individual stimulus z-maps
                for stim_id, _label in enumerate(self._global_labels):
                    cond_num = stim_id + 1
                    matches = list(
                        sess_dir.glob(
                            f"sub-{subj}_ses-{sess:02d}_cond-condition {cond_num}-*_zmap.nii.gz"
                        )
                    )
                    if matches:
                        records.append(
                            Record(
                                subject_id=subj,
                                session_id=sess,
                                stim_id=stim_id,
                                is_contrast=False,
                                contrast_name=None,
                                zmap_path=matches[0],
                            )
                        )
                # Contrast z-maps — discover via filename
                for contrast_file in sorted(
                    sess_dir.glob(f"sub-{subj}_ses-{sess:02d}_contrast-*_zmap.nii.gz")
                ):
                    cname = self._infer_contrast_name(contrast_file.name)
                    if cname is None:
                        continue
                    records.append(
                        Record(
                            subject_id=subj,
                            session_id=sess,
                            stim_id=-1,
                            is_contrast=True,
                            contrast_name=cname,
                            zmap_path=contrast_file,
                        )
                    )
        return records

    def load_zmap(self, record: Record) -> np.ndarray:
        from nilearn import image

        if self._mask_data is None:
            mask_img = image.load_img(str(self._mask_path))
            self._mask_data = mask_img.get_fdata().astype(bool)
        img = image.load_img(str(record.zmap_path))
        data_3d = img.get_fdata()
        return data_3d[self._mask_data].astype(np.float32)

    def get_cog_vector(self, subject_id: int, session_id: int) -> np.ndarray:
        cols = self._resolve_cog_columns_for_session(session_id)
        rows = self._df_behav[self._df_behav[SUBJECT_COL] == subject_id]
        if rows.empty:
            raise ValueError(f"No behavioral row for subject {subject_id}")
        row = rows.iloc[0]
        out: List[float] = []
        for col in cols:
            v = row[col]
            if pd.isna(v) or v == "":
                raise ValueError(
                    f"Missing cog value subj={subject_id} ses={session_id} col={col!r}"
                )
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Non-numeric cog value subj={subject_id} ses={session_id} col={col!r}"
                )
            if not np.isfinite(f):
                raise ValueError(
                    f"Non-finite cog value subj={subject_id} ses={session_id} col={col!r}"
                )
            out.append(f)
        return np.asarray(out, dtype=np.float64)

    # ---- DFG-specific helpers ----

    def _resolve_cog_columns_for_session(self, session_id: int) -> List[str]:
        """For each base column name, prefer "<base>#<session>" else "<base>"."""
        cols: List[str] = []
        df_cols = set(self._df_behav.columns)
        for base in self.cog_base_cols:
            ses_col = f"{base}#{session_id}"
            if ses_col in df_cols:
                cols.append(ses_col)
            elif base in df_cols:
                cols.append(base)
            else:
                raise KeyError(
                    f"Cog column missing for base={base!r} session={session_id}; "
                    f"tried {ses_col!r} and {base!r}"
                )
        return cols

    def _discover_global_labels(self) -> List[str]:
        all_labels = set()
        for p in sorted(self.metadata_root.glob("session_*_cog2bold.pkl")):
            with open(p, "rb") as f:
                d = pickle.load(f)
            meta = d.get("meta") or {}
            labs = meta.get("stimulus_labels")
            if labs is None:
                raise ValueError(f"{p.name} missing meta['stimulus_labels']")
            all_labels.update(labs)
        if not all_labels:
            raise FileNotFoundError(
                f"No session_*_cog2bold.pkl with stimulus_labels found in {self.metadata_root}"
            )
        return sorted(all_labels)

    def _infer_contrast_name(self, filename: str) -> Optional[str]:
        """Extract the canonical 'a>b' contrast name from an fMRIPrep contrast filename.

        Filenames look like:
            sub-1_ses-01_contrast-condition 2-audio only_vs_condition 3-visual only_zmap.nii.gz
        We match the substring between 'contrast-' and '_zmap', split on '_vs_',
        then map both halves back to canonical labels via prefix matching.
        """
        try:
            after = filename.split("contrast-", 1)[1]
            inside = after.rsplit("_zmap", 1)[0]
            a_raw, b_raw = inside.split("_vs_", 1)
        except (IndexError, ValueError):
            return None
        a = self._match_label(a_raw)
        b = self._match_label(b_raw)
        if a is None or b is None:
            return None
        return f"{a}>{b}"

    def _match_label(self, raw: str) -> Optional[str]:
        """Map 'condition 2-audio only' (raw) to whichever global label starts with that prefix."""
        # global labels look like 'condition 2-audio only' too, so a direct match is usually enough
        if raw in self._global_labels:
            return raw
        for lab in self._global_labels:
            if raw.startswith(lab) or lab.startswith(raw):
                return lab
        return None
