"""Pipelines (training + leave-two-out + leave-one-out)."""

from cog2bold.pipelines.lto import LeaveTwoOutClassifier, LTOArgs, run_lto_from_run_dir
from cog2bold.pipelines.loo import LeaveOneOutEvaluator, LOOArgs, run_loo_from_run_dir
from cog2bold.pipelines.training import TrainingPipeline

__all__ = [
    "TrainingPipeline",
    "LeaveTwoOutClassifier",
    "LTOArgs",
    "run_lto_from_run_dir",
    "LeaveOneOutEvaluator",
    "LOOArgs",
    "run_loo_from_run_dir",
]
