"""Cog2BOLD — predict fMRI activation maps from (cognitive scores × stimulus)."""

__version__ = "0.1.0"

from cog2bold.config import (
    BasisConfig,
    CorrelationConfig,
    DataConfig,
    PredictorConfig,
    RunConfig,
)
from cog2bold.bases import create_basis
from cog2bold.predictors import create_predictor
from cog2bold.datasets import DatasetAdapter, create_adapter

__all__ = [
    "BasisConfig",
    "CorrelationConfig",
    "DataConfig",
    "PredictorConfig",
    "RunConfig",
    "create_basis",
    "create_predictor",
    "DatasetAdapter",
    "create_adapter",
]
