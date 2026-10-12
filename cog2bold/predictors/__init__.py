"""Predictor registry + factory."""

from cog2bold.config import PredictorConfig
from cog2bold.predictors.base import CoefficientPredictor
from cog2bold.predictors.linear import LinearPredictor


def create_predictor(config: PredictorConfig) -> CoefficientPredictor:
    if config.model_type == "linear":
        return LinearPredictor(config)
    if config.model_type == "mlp":
        # imported lazily so linear-only use (e.g. the deployed app) doesn't need PyTorch
        from cog2bold.predictors.mlp import MLPPredictor

        return MLPPredictor(config)
    raise ValueError(f"Unknown predictor model_type {config.model_type!r}")


__all__ = ["CoefficientPredictor", "create_predictor"]
