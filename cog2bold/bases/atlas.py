"""Atlas / parcellation basis.

Components are ROI masks. encode() = mean activation per ROI; decode() = broadcast
ROI mean back to voxels. Lossy by design.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cog2bold.bases.base import BasisTransform
from cog2bold.config import BasisConfig


class AtlasBasis(BasisTransform):
    """ROI-based parcellation basis.

    extra_params:
        - atlas_name: 'harvard-oxford' (only built-in option for now) or a custom path
        - atlas_path: optional path to a custom labelled NIfTI atlas
    """

    def __init__(self, config: BasisConfig):
        super().__init__(config)
        self.atlas_name = config.extra_params.get("atlas_name", "harvard-oxford")
        self.atlas_path = config.extra_params.get("atlas_path")
        self.roi_masks: Optional[np.ndarray] = None  # (n_components, n_voxels)

    def fit(self, train_maps: np.ndarray, masker=None, **kwargs) -> "AtlasBasis":
        from nilearn import datasets, image
        import nibabel as nib

        if masker is None:
            raise ValueError("AtlasBasis.fit requires a masker (use adapter.get_masker())")
        self.original_shape = train_maps.shape[1:]
        n_voxels = train_maps.shape[1]

        if self.atlas_path:
            atlas_img = nib.load(str(self.atlas_path))
        elif "harvard-oxford" in self.atlas_name.lower():
            atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
            atlas_img = atlas.maps
        else:
            raise ValueError(f"Unknown atlas_name {self.atlas_name!r}")

        # Resample atlas to fMRI space using the masker's mask image
        mask_img = masker.mask_img_
        dummy = nib.Nifti1Image(np.zeros(mask_img.shape[:3]), mask_img.affine, mask_img.header)
        atlas_resampled = image.resample_to_img(atlas_img, dummy, interpolation="nearest")
        atlas_data = atlas_resampled.get_fdata()
        labels = np.unique(atlas_data)
        labels = labels[labels != 0]

        # Cap or warn on K vs atlas size
        if len(labels) < self.n_components:
            print(
                f"[Atlas] atlas has {len(labels)} ROIs, reducing n_components from {self.n_components}"
            )
            self.n_components = len(labels)
        elif len(labels) > self.n_components:
            sizes = [(lab, int((atlas_data == lab).sum())) for lab in labels]
            sizes.sort(key=lambda x: x[1], reverse=True)
            labels = np.asarray([lab for lab, _ in sizes[: self.n_components]])
            print(f"[Atlas] selected top {self.n_components} ROIs by size")

        mask_data = mask_img.get_fdata().astype(bool)
        atlas_flat = atlas_data[mask_data]
        roi_masks = np.zeros((self.n_components, n_voxels), dtype=np.float32)
        for i, lab in enumerate(labels):
            roi_masks[i] = (atlas_flat == lab).astype(np.float32)
        sizes = roi_masks.sum(axis=1, keepdims=True)
        roi_masks = roi_masks / np.maximum(sizes, 1.0)
        self.roi_masks = roi_masks
        self.is_fitted = True
        return self

    def encode(self, activation_map: np.ndarray, **kwargs) -> np.ndarray:
        return self.roi_masks @ activation_map

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        # Broadcast each ROI mean back across its voxels.
        # Note: roi_masks rows are normalised by ROI size, so this gives a binary-ish mask
        # weighted by the coefficient. Equivalent to constructing a per-voxel value =
        # coefficient_for_its_ROI.
        # Use binary masks (>0) for clean broadcast:
        binary = (self.roi_masks > 0).astype(np.float32)
        return coefficients @ binary

    def get_decode_matrix(self) -> np.ndarray:
        return (self.roi_masks > 0).astype(np.float32)

    def _get_state(self) -> dict:
        return {
            "roi_masks": self.roi_masks,
            "original_shape": self.original_shape,
            "atlas_name": self.atlas_name,
            "n_components": self.n_components,
        }

    def _set_state(self, state: dict) -> None:
        self.roi_masks = state["roi_masks"]
        self.original_shape = state["original_shape"]
        self.atlas_name = state["atlas_name"]
        self.n_components = state["n_components"]
