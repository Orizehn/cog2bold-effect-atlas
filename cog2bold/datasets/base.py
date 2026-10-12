"""Dataset adapter abstraction.

To use Cog2BOLD with a new dataset:
  1. subclass DatasetAdapter in cog2bold/datasets/<your_dataset>.py
  2. implement the abstract methods
  3. register it in cog2bold/datasets/registry.py

See cog2bold/datasets/dfg.py for a concrete example.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------- data classes ----------


@dataclass
class Record:
    """Lightweight pointer to a single z-map on disk.

    Adapters return a list of these from `list_records()`. They are deliberately
    minimal — actual data loading happens in `load_zmap(record)`.
    """

    subject_id: int
    session_id: int
    stim_id: int  # -1 for contrast maps
    is_contrast: bool
    contrast_name: Optional[str]
    zmap_path: Path


@dataclass
class ActivationMapSample:
    """One materialised (z-map, cog, stim_encoding) tuple ready for training."""

    map: np.ndarray            # (n_voxels,)
    cog_features: np.ndarray   # (n_cog,)
    stim_encoding: np.ndarray  # (n_stim_dim,) — one-hot OR contrast vector
    stim_id: int               # global stim id (>=0) or -1 for contrast
    is_contrast: bool
    contrast_name: Optional[str]
    subject_id: int
    session_id: int


# ---------- ABC ----------


class DatasetAdapter(ABC):
    """Plug-in interface for a fMRI + cognitive-scores dataset."""

    # --- adapter metadata, set by subclass ---

    name: str = "abstract"  # short identifier used for caching and the registry

    @property
    @abstractmethod
    def global_labels(self) -> List[str]:
        """Ordered list of stimulus labels (length = n_stim)."""

    @property
    def cog_feature_names(self) -> List[str]:
        """Names of the entries in the vector returned by `get_cog_vector`.

        Default: the `cog_base_cols` passed to the adapter (one name per CSV column).
        Override when the adapter expands features (e.g. one-hot encoding), so the
        names stay aligned with the longer feature vector. Used only for display.
        """
        return list(getattr(self, "cog_base_cols", []))

    # --- per-dataset defaults (for UIs/examples; safe to override) ---

    @classmethod
    def default_data_root(cls) -> Optional[Path]:
        """Best-guess data root for this dataset (env var or a relative location).

        Returns None by default; concrete adapters override so the apps can
        auto-fill the path when the dataset is selected.
        """
        return None

    @classmethod
    def default_cog_base_cols(cls) -> List[str]:
        """Sensible default cognitive columns for this dataset (empty by default)."""
        return []

    # --- record discovery + loading ---

    @abstractmethod
    def list_records(self) -> List[Record]:
        """Enumerate every (subject, session, stimulus-or-contrast, file) entry available."""

    @abstractmethod
    def load_zmap(self, record: Record) -> np.ndarray:
        """Load the z-map at `record.zmap_path` and return the masked vector (n_voxels,)."""

    @abstractmethod
    def get_cog_vector(self, subject_id: int, session_id: int) -> np.ndarray:
        """Cognitive feature vector (n_cog,) for a given (subject, session)."""

    # --- mask + masker ---

    @abstractmethod
    def mask_path(self) -> Path:
        """Path to the brain mask NIfTI file."""

    def get_masker(self):
        """Build a fresh NiftiMasker from `mask_path()`.

        Always returns a new masker — we don't pickle/share them, which avoids
        the version-incompatibility footguns in the legacy code.
        """
        from nilearn import image
        from nilearn.maskers import NiftiMasker

        mask_path = self.mask_path()
        if not Path(mask_path).exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask_img = image.load_img(str(mask_path))
        masker = NiftiMasker(mask_img=mask_img)
        masker.fit()
        return masker

    # --- helpers (concrete here so adapters stay small) ---

    @property
    def label2id(self) -> Dict[str, int]:
        return {lab: i for i, lab in enumerate(self.global_labels)}

    def compute_stim_encoding(
        self, stim_id: int, is_contrast: bool = False, contrast_name: Optional[str] = None,
    ) -> np.ndarray:
        """Return a (n_stim,) encoding vector for one record.

        - individual stimulus -> one-hot
        - contrast (e.g. "audio>visual") -> +1 at audio index, -1 at visual index
        """
        n_stim = len(self.global_labels)
        enc = np.zeros(n_stim, dtype=np.float32)
        if not is_contrast:
            enc[stim_id] = 1.0
            return enc
        if contrast_name is None:
            raise ValueError("contrast_name required when is_contrast=True")
        return self._parse_contrast_name(contrast_name)

    def _parse_contrast_name(self, contrast_name: str) -> np.ndarray:
        sep = ">" if ">" in contrast_name else ("_vs_" if "_vs_" in contrast_name else None)
        if sep is None:
            raise ValueError(f"Bad contrast name {contrast_name!r}; expected 'a>b' or 'a_vs_b'")
        a, b = contrast_name.split(sep, 1)
        enc = np.zeros(len(self.global_labels), dtype=np.float32)
        l2i = self.label2id
        if a in l2i:
            enc[l2i[a]] = 1.0
        if b in l2i:
            enc[l2i[b]] = -1.0
        return enc

    def all_pairwise_contrasts(self) -> List[Tuple[str, str]]:
        labs = self.global_labels
        return [(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]

    def resolve_contrast_string(self, s: str) -> str:
        """Accept either a literal name or a vector string '[1 0 0 -1]', return canonical name."""
        s = s.strip()
        if not (s.startswith("[") and s.endswith("]")):
            return s
        inner = s[1:-1].strip()
        try:
            weights = [float(x) for x in inner.split()]
        except ValueError:
            try:
                weights = [float(x) for x in inner.split(",")]
            except ValueError:
                raise ValueError(f"Cannot parse contrast vector {s!r}")
        labs = self.global_labels
        if len(weights) != len(labs):
            raise ValueError(f"Vector has {len(weights)} entries but {len(labs)} stimuli")
        pos = [i for i, w in enumerate(weights) if w > 0]
        neg = [i for i, w in enumerate(weights) if w < 0]
        if len(pos) == 1 and not neg:
            return labs[pos[0]]
        if len(pos) == 1 and len(neg) == 1:
            return f"{labs[pos[0]]}>{labs[neg[0]]}"
        raise ValueError(f"Vector {s!r} must be one-hot or one-vs-one")

    # --- materialisation: where ABC turns into a list of samples ---

    def materialize(
        self,
        cog_base_cols: List[str],
        stim_encoding_mode: str = "onehot",
        contrast_pairs: Optional[List[Tuple[str, str]]] = None,
        subject_session_filter: Optional[Set[Tuple[int, int]]] = None,
        cache_dir: Optional[Path] = None,
        force_recompute: bool = False,
    ) -> List[ActivationMapSample]:
        """Load (or load-and-cache) all samples for this adapter.

        Cache key (fix #4 from review): hash of (adapter name, cog_base_cols,
        stim_encoding_mode, sorted contrast_pairs, sorted filter set). Changes to
        any of these invalidate the cache automatically.
        """
        cache_key = self._cache_key(
            cog_base_cols=cog_base_cols,
            stim_encoding_mode=stim_encoding_mode,
            contrast_pairs=contrast_pairs,
            subject_session_filter=subject_session_filter,
        )

        cache_pkl = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_pkl = cache_dir / f"{self.name}_{cache_key}.pkl"
            if cache_pkl.exists() and not force_recompute:
                with open(cache_pkl, "rb") as f:
                    cached = pickle.load(f)
                print(f"[{self.name}] Loaded {len(cached)} samples from cache {cache_pkl.name}")
                return cached

        load_contrasts = stim_encoding_mode in {"contrast", "contrast_only"}
        load_individuals = stim_encoding_mode in {"onehot", "contrast"}

        records = self.list_records()
        if subject_session_filter is not None:
            records = [r for r in records if (r.subject_id, r.session_id) in subject_session_filter]

        # Filter out individual or contrast records depending on encoding mode
        if not load_individuals:
            records = [r for r in records if r.is_contrast]
        if not load_contrasts:
            records = [r for r in records if not r.is_contrast]
        elif contrast_pairs is not None:
            # Keep only the requested contrasts (match either direction). When
            # contrast_pairs is None we keep *all* contrast records the adapter found.
            wanted = set()
            for a, b in contrast_pairs:
                wanted.add(f"{a}>{b}")
                wanted.add(f"{b}>{a}")
            records = [
                r for r in records if (not r.is_contrast) or (r.contrast_name in wanted)
            ]

        samples: List[ActivationMapSample] = []
        skipped_cog = 0
        for r in records:
            try:
                cog_vec = self.get_cog_vector(r.subject_id, r.session_id)
            except (KeyError, ValueError):
                skipped_cog += 1
                continue
            zmap = self.load_zmap(r)
            if not np.all(np.isfinite(zmap)):
                skipped_cog += 1
                continue
            enc = self.compute_stim_encoding(r.stim_id, r.is_contrast, r.contrast_name)
            samples.append(
                ActivationMapSample(
                    map=zmap.astype(np.float32),
                    cog_features=cog_vec.astype(np.float32),
                    stim_encoding=enc.astype(np.float32),
                    stim_id=r.stim_id,
                    is_contrast=r.is_contrast,
                    contrast_name=r.contrast_name,
                    subject_id=r.subject_id,
                    session_id=r.session_id,
                )
            )

        if skipped_cog:
            print(f"[{self.name}] Skipped {skipped_cog} records (missing cog or non-finite map)")

        if cache_pkl is not None:
            with open(cache_pkl, "wb") as f:
                pickle.dump(samples, f)
            print(f"[{self.name}] Cached {len(samples)} samples to {cache_pkl.name}")

        return samples

    def _cache_key(
        self,
        cog_base_cols: List[str],
        stim_encoding_mode: str,
        contrast_pairs: Optional[List[Tuple[str, str]]],
        subject_session_filter: Optional[Set[Tuple[int, int]]],
    ) -> str:
        payload = {
            "adapter": self.name,
            "cog_base_cols": sorted(cog_base_cols),
            "stim_encoding_mode": stim_encoding_mode,
            "contrast_pairs": sorted(contrast_pairs or []),
            "filter": sorted(subject_session_filter) if subject_session_filter else None,
            "global_labels": list(self.global_labels),
        }
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def filter_samples_by_contrast(
    samples: List[ActivationMapSample],
    single_contrast,
    adapter: DatasetAdapter,
) -> List[ActivationMapSample]:
    """Keep only samples matching the requested stimulus/contrast name(s).

    `single_contrast` may be a single name (e.g. "Words" or "Words>Checkerboard",
    vector strings allowed) OR a list of such names to keep a mixed subset (e.g.
    ["Words>Checkerboard", "Words"]). Note: the relevant maps must already be
    materialised — use stim_encoding="contrast" to load individuals + contrasts.
    """
    names = [single_contrast] if isinstance(single_contrast, str) else list(single_contrast)
    targets = {adapter.resolve_contrast_string(n) for n in names}
    out = []
    for s in samples:
        name = s.contrast_name if s.is_contrast else adapter.global_labels[s.stim_id]
        if name in targets:
            out.append(s)
    if not out:
        raise ValueError(
            f"No samples for single_contrast={sorted(targets)!r}. "
            f"Available: contrasts={sorted({s.contrast_name for s in samples if s.is_contrast})}, "
            f"stims={sorted({adapter.global_labels[s.stim_id] for s in samples if not s.is_contrast})}"
        )
    return out
