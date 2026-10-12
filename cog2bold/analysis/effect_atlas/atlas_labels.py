"""Anatomical labelling of MNI coordinates and model-space voxels.

Two atlases, both shipped through nilearn and cached locally after the first fetch:

- **Harvard-Oxford** probabilistic (cortical 48 + subcortical structures) — the classic
  anatomical label of neuroimaging peak tables ("Inferior Frontal Gyrus, pars opercularis (L), 54%").
  Cortical names are lateralised from the sign of x (MNI RAS: x<0 = left); the subcortical
  names already carry a side. The subcortical "Cerebral Cortex"/"White Matter"/"Ventricle"
  catch-all volumes are never used as a region name — white matter is reported as a flag.
- **Schaefer 2018, 200 parcels / Yeo 7 networks** (1 mm) — a functional parcel + network
  label as the second column.

Lookups go coordinate → atlas voxel through the atlas's own inverse affine, so the stats map
is never resampled. Caveat (stated in every report): both atlases live in FSL MNI152
(NLin6Asym); ds000030/AOMIC maps are MNI152NLin2009cAsym — the templates differ by a few mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

# HO subcortical volumes that are tissue classes, not regions
_HO_SUB_EXCLUDE = ("Cerebral White Matter", "Cerebral Cortex", "Lateral Ventricle")

YEO7_NAMES = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience / ventral attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal control",
    "Default": "Default mode",
}

NEAREST_RADIUS_MM = 4.0
MAXPROB_THRESHOLD = 25.0  # % — same convention as FSL's maxprob-thr25 atlases
MIN_PEAK_PROB = 10.0      # % — below this a peak label is replaced by WM / nearest region


def _as_img(maps):
    import nibabel as nib

    return nib.load(maps) if isinstance(maps, str) else maps


def _clean(lbl) -> str:
    return (lbl.decode() if isinstance(lbl, bytes) else str(lbl)).strip()


def _side(x_mm: float) -> str:
    return "L" if x_mm < 0 else "R"


@dataclass
class HarvardOxford:
    prob: np.ndarray          # (X, Y, Z, R) float32, % probability per region
    names: List[str]          # (R,) unlateralised region names
    lateral: np.ndarray       # (R,) bool — True for cortical names (side taken from x)
    wm_prob: np.ndarray       # (X, Y, Z) max(left, right) white-matter probability
    affine: np.ndarray

    def name(self, r: int, x_mm: float) -> str:
        return f"{self.names[r]} ({_side(x_mm)})" if self.lateral[r] else self.names[r]


@dataclass
class Schaefer:
    labels: np.ndarray        # (X, Y, Z) int, 0 = background
    names: List[str]          # index v -> parcel name ('' for background)
    affine: np.ndarray

    @staticmethod
    def network(parcel: str) -> str:
        parts = parcel.split("_")
        return parts[2] if len(parts) > 2 else ""


@lru_cache(maxsize=1)
def harvard_oxford() -> HarvardOxford:
    from nilearn import datasets

    cort = datasets.fetch_atlas_harvard_oxford("cort-prob-2mm", verbose=0)
    sub = datasets.fetch_atlas_harvard_oxford("sub-prob-2mm", verbose=0)
    cimg, simg = _as_img(cort.maps), _as_img(sub.maps)
    cdat = np.asarray(cimg.dataobj, dtype=np.float32)
    sdat = np.asarray(simg.dataobj, dtype=np.float32)
    # labels include a leading 'Background' with no volume
    cnames = [_clean(l) for l in cort.labels][1:]
    snames = [_clean(l) for l in sub.labels][1:]
    assert len(cnames) == cdat.shape[3] and len(snames) == sdat.shape[3]

    keep = [i for i, n in enumerate(snames) if not any(e in n for e in _HO_SUB_EXCLUDE)]
    wm = [i for i, n in enumerate(snames) if "White Matter" in n]
    prob = np.concatenate([cdat, sdat[..., keep]], axis=3)
    names = cnames + [snames[i] for i in keep]
    lateral = np.array([True] * len(cnames) + [False] * len(keep))
    return HarvardOxford(
        prob=prob, names=names, lateral=lateral,
        wm_prob=sdat[..., wm].max(axis=3), affine=cimg.affine,
    )


@lru_cache(maxsize=1)
def schaefer() -> Schaefer:
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=200, yeo_networks=7, resolution_mm=1, verbose=0,
    )
    img = _as_img(atlas.maps)
    names = [_clean(l) for l in atlas.labels]
    if not names[0].lower().startswith("background"):
        names = ["Background"] + names
    names[0] = ""
    return Schaefer(
        labels=np.asarray(img.dataobj).astype(np.int32), names=names, affine=img.affine,
    )


def _mm_to_ijk(xyz_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    xyz = np.atleast_2d(np.asarray(xyz_mm, dtype=np.float64))
    hom = np.c_[xyz, np.ones(len(xyz))]
    return np.rint(hom @ np.linalg.inv(affine).T)[:, :3].astype(int)


def _in_bounds(ijk: np.ndarray, shape) -> np.ndarray:
    return np.all((ijk >= 0) & (ijk < np.asarray(shape[:3])), axis=1)


def _neighbourhood(affine: np.ndarray, radius_mm: float) -> np.ndarray:
    """Integer voxel offsets within `radius_mm`, sorted by distance."""
    vox = np.abs(np.diag(affine)[:3])
    r = np.ceil(radius_mm / vox).astype(int)
    g = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in r], indexing="ij"), -1).reshape(-1, 3)
    d = np.linalg.norm(g * vox, axis=1)
    keep = d <= radius_mm
    order = np.argsort(d[keep])
    return g[keep][order]


# ---------------------------------------------------------------- coordinate lookup


def label_coordinate(xyz_mm) -> Dict[str, object]:
    """Atlas labels for one MNI coordinate (mm).

    Returns keys: ``ho_region``, ``ho_prob`` (%), ``ho_flag`` ('' | 'white matter' |
    'nearest <d> mm' | 'low probability' | 'no label'), ``schaefer_parcel``, ``network``.
    """
    x = float(np.asarray(xyz_mm)[0])
    out: Dict[str, object] = {}

    ho = harvard_oxford()
    ijk = _mm_to_ijk(xyz_mm, ho.affine)
    if not _in_bounds(ijk, ho.prob.shape)[0]:
        out.update(ho_region="outside atlas", ho_prob=0.0, ho_flag="no label")
    else:
        i, j, k = ijk[0]
        p = ho.prob[i, j, k]
        r = int(np.argmax(p))
        if p[r] >= MIN_PEAK_PROB:
            out.update(ho_region=ho.name(r, x), ho_prob=float(p[r]), ho_flag="")
        else:
            out.update(_nearest_ho(ho, ijk[0], x))
            if out["ho_flag"] == "no label" and p[r] > 0:
                out.update(ho_region=ho.name(r, x), ho_prob=float(p[r]), ho_flag="low probability")

    sch = schaefer()
    ijk = _mm_to_ijk(xyz_mm, sch.affine)
    parcel = ""
    if _in_bounds(ijk, sch.labels.shape)[0]:
        for off in _neighbourhood(sch.affine, NEAREST_RADIUS_MM):
            q = ijk[0] + off
            if np.all((q >= 0) & (q < sch.labels.shape)):
                v = int(sch.labels[tuple(q)])
                if v:
                    parcel = sch.names[v]
                    break
    out.update(schaefer_parcel=parcel, network=Schaefer.network(parcel))
    return out


def _nearest_ho(ho: HarvardOxford, ijk: np.ndarray, x: float) -> Dict[str, object]:
    i, j, k = ijk
    if ho.wm_prob[i, j, k] >= 50:
        return dict(ho_region="Cerebral White Matter", ho_prob=float(ho.wm_prob[i, j, k]),
                    ho_flag="white matter")
    vox = np.abs(np.diag(ho.affine)[:3])
    for off in _neighbourhood(ho.affine, NEAREST_RADIUS_MM)[1:]:
        q = ijk + off
        if np.all((q >= 0) & (q < ho.prob.shape[:3])):
            p = ho.prob[tuple(q)]
            r = int(np.argmax(p))
            if p[r] >= MIN_PEAK_PROB:
                d = float(np.linalg.norm(off * vox))
                return dict(ho_region=ho.name(r, x), ho_prob=float(p[r]),
                            ho_flag=f"nearest {d:.0f} mm")
    return dict(ho_region="unlabelled", ho_prob=0.0, ho_flag="no label")


# ---------------------------------------------------------------- model-space voxel maps


@dataclass
class VoxelRegions:
    """Region assignment for every in-mask voxel of a model (C-order boolean indexing).

    ``ho`` / ``yeo7`` hold integer codes into ``ho_names`` / ``yeo7_names``; -1 = unassigned.
    """

    ho: np.ndarray
    ho_names: List[str]
    yeo7: np.ndarray
    yeo7_names: List[str]


def mask_voxel_coords(mask_img) -> np.ndarray:
    """(n_voxels, 3) MNI mm coordinates of in-mask voxels, in masked-vector order."""
    mask = np.asarray(mask_img.dataobj).astype(bool)
    ijk = np.argwhere(mask)  # C order == boolean-index order
    return np.c_[ijk, np.ones(len(ijk))] @ mask_img.affine.T[:, :3]


def voxel_regions(mask_img) -> VoxelRegions:
    """Assign each in-mask voxel to an HO max-prob region (≥25 %) and a Yeo-7 network."""
    xyz = mask_voxel_coords(mask_img)
    side = np.where(xyz[:, 0] < 0, "L", "R")

    ho = harvard_oxford()
    ijk = _mm_to_ijk(xyz, ho.affine)
    ok = _in_bounds(ijk, ho.prob.shape)
    ho_str = np.full(len(xyz), "", dtype=object)
    p = ho.prob[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    r = p.argmax(axis=1)
    pmax = p[np.arange(len(r)), r]
    idx_ok = np.flatnonzero(ok)
    for n, (ri, pm, s) in enumerate(zip(r, pmax, side[ok])):
        if pm >= MAXPROB_THRESHOLD:
            ho_str[idx_ok[n]] = f"{ho.names[ri]} ({s})" if ho.lateral[ri] else ho.names[ri]

    sch = schaefer()
    ijk = _mm_to_ijk(xyz, sch.affine)
    ok = _in_bounds(ijk, sch.labels.shape)
    yeo_str = np.full(len(xyz), "", dtype=object)
    v = sch.labels[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    yeo_str[ok] = [Schaefer.network(sch.names[x]) if x else "" for x in v]

    return VoxelRegions(*_encode(ho_str), *_encode(yeo_str, order=list(YEO7_NAMES)))


def _encode(strings: np.ndarray, order: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    names = order if order is not None else sorted({s for s in strings if s})
    lut = {n: i for i, n in enumerate(names)}
    return np.array([lut.get(s, -1) for s in strings], dtype=np.int32), names


def region_means(values: np.ndarray, codes: np.ndarray, n_regions: int) -> np.ndarray:
    """Mean of `values` (..., n_voxels) within each region code → (..., n_regions)."""
    ok = codes >= 0
    counts = np.bincount(codes[ok], minlength=n_regions).astype(np.float64)
    v = np.atleast_2d(values)[:, ok]
    sums = np.stack([np.bincount(codes[ok], weights=row, minlength=n_regions) for row in v])
    out = sums / np.maximum(counts, 1)
    out[:, counts == 0] = np.nan
    return out if values.ndim > 1 else out[0]


def cluster_composition(codes: np.ndarray, names: List[str], top: int = 3) -> str:
    """'Region A 45%; Region B 30%; …' over the voxels of one cluster."""
    ok = codes >= 0
    if not ok.any():
        return "unlabelled"
    c = np.bincount(codes[ok], minlength=len(names))
    order = np.argsort(c)[::-1][:top]
    n = len(codes)
    return "; ".join(f"{names[i]} {100 * c[i] / n:.0f}%" for i in order if c[i] > 0)
