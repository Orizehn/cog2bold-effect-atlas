"""Adapter registry — name → adapter class lookup."""

from __future__ import annotations

from typing import Any, Dict, Type

from cog2bold.datasets.base import DatasetAdapter

_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register_adapter(name: str, cls: Type[DatasetAdapter]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Adapter {name!r} already registered")
    _REGISTRY[name] = cls


def list_adapters() -> Dict[str, Type[DatasetAdapter]]:
    return dict(_REGISTRY)


def create_adapter(name: str, **kwargs: Any) -> DatasetAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown adapter {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


# Eager registration of built-in adapters
def _register_builtins() -> None:
    from cog2bold.datasets.dfg import DFGAdapter
    from cog2bold.datasets.era812 import ERA812Adapter
    from cog2bold.datasets.aomic_piop1 import register_aomic
    from cog2bold.datasets.ds000030 import register_ds000030

    register_adapter("dfg", DFGAdapter)
    register_adapter("era812", ERA812Adapter)
    register_aomic()  # registers aomic_piop1_<task> for each AOMIC task
    register_ds000030()  # registers ds000030_<task> for each CNP task (scap for now)


_register_builtins()
