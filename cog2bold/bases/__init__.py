"""Basis registry + factory."""

from __future__ import annotations

from cog2bold.bases.atlas import AtlasBasis
from cog2bold.bases.base import BasisTransform
from cog2bold.bases.cca import CCABasis
from cog2bold.bases.hybrid import HybridBasis
from cog2bold.bases.ica import ICABasis
from cog2bold.bases.identity import IdentityBasis
from cog2bold.bases.pca import PCABasis
from cog2bold.bases.pls import PLSBasis
from cog2bold.bases.random_basis import RandomOrthonormalBasis
from cog2bold.bases.ridgecol import RidgeColBasis
from cog2bold.config import BasisConfig

_REGISTRY = {
    "identity": IdentityBasis,
    "pca": PCABasis,
    "ica": ICABasis,
    "atlas": AtlasBasis,
    "random": RandomOrthonormalBasis,
    "ridgecol": RidgeColBasis,
    "cca": CCABasis,
    "pls": PLSBasis,
    "hybrid": HybridBasis,
}


def create_basis(config: BasisConfig) -> BasisTransform:
    cls = _REGISTRY.get(config.basis_type)
    if cls is None:
        raise ValueError(f"Unknown basis_type {config.basis_type!r}; known: {sorted(_REGISTRY)}")
    return cls(config)


__all__ = ["BasisTransform", "create_basis"]
