"""Atlas-space projection helper used by training and LTO.

Projects masked voxel maps to ROI mean vectors (Schaefer 2018) and back.
Consolidates code that was previously duplicated across `training.py` and
`leave_two_out_classification.py`, and ensures both pipelines use the same
projection so map-shape mismatches can't sneak in.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class AtlasProjector:
    """Projects (n_voxels,) masked maps into Schaefer ROI space and back.

    One projector handles a single (n_rois, mask) pair. Build via:
        projector = AtlasProjector(masker=adapter.get_masker(), n_rois=200)
    """

    def __init__(self, masker, n_rois: int):
        from nilearn import datasets
        from nilearn.maskers import NiftiLabelsMasker

        self.n_rois = n_rois
        self.masker = masker
        self.mask_img = masker.mask_img_
        self.mask_data = self.mask_img.get_fdata().astype(bool)

        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=17)
        self.atlas = atlas
        labels = atlas.labels
        self.labels: List[str] = [
            (lbl.decode() if isinstance(lbl, bytes) else str(lbl)) for lbl in labels
        ]
        # nilearn includes a background entry at index 0 → drop it so indices align
        if len(self.labels) == n_rois + 1:
            self.labels = self.labels[1:]

        self.roi_masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=False)
        self.roi_masker.fit(self.mask_img)

    def voxel_to_roi(self, masked_vec: np.ndarray) -> np.ndarray:
        """(n_voxels,) -> (n_rois,)."""
        from nilearn import image

        if masked_vec.ndim != 1:
            masked_vec = masked_vec.reshape(-1)
        full = np.zeros(self.mask_data.shape, dtype=np.float32)
        full[self.mask_data] = masked_vec.astype(np.float32)
        img = image.new_img_like(self.mask_img, full)
        roi_vec = self.roi_masker.transform(img).reshape(-1)
        return roi_vec

    def voxel_to_roi_batch(self, masked_mat: np.ndarray) -> np.ndarray:
        return np.asarray([self.voxel_to_roi(m) for m in masked_mat])

    def roi_to_voxel_batch(self, roi_mat: np.ndarray) -> np.ndarray:
        """(n_samples, n_rois) -> (n_samples, n_voxels) by uniform broadcast within ROI.

        We extract masked voxels directly from the 4D NIfTI rather than calling
        `masker.transform(...)`. The latter occasionally fails on cached
        NiftiMaskers from older nilearn versions ("missing clean_args_") — the
        manual path is version-independent.
        """
        from nilearn import image  # noqa: F401  (used by inverse_transform)

        roi_2d = roi_mat if roi_mat.ndim == 2 else roi_mat[np.newaxis, :]
        nifti_4d = self.roi_masker.inverse_transform(roi_2d.astype(np.float32))
        data_4d = nifti_4d.get_fdata().astype(np.float32)  # (X, Y, Z, n_samples)
        voxel_by_sample = data_4d[self.mask_data]  # (n_voxels, n_samples)
        return voxel_by_sample.T  # (n_samples, n_voxels)

    def roi_indices_matching(self, patterns: List[str]) -> List[int]:
        return [
            i for i, lbl in enumerate(self.labels)
            if any(p.lower() in lbl.lower() for p in patterns)
        ]


# Cache so a run that needs Schaefer-200 doesn't fetch + fit twice.
_CACHE: Dict[Tuple[int, int], AtlasProjector] = {}


def get_atlas_projector(masker, n_rois: int) -> AtlasProjector:
    key = (n_rois, id(masker))
    if key not in _CACHE:
        _CACHE[key] = AtlasProjector(masker, n_rois=n_rois)
    return _CACHE[key]


def clear_atlas_cache() -> None:
    _CACHE.clear()
