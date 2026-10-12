"""Configuration dataclasses for Cog2BOLD.

Every config is JSON-serialisable so a run is reproducible from disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- correlation evaluation spaces ----------


@dataclass
class CorrelationConfig:
    """A single correlation-space spec used in evaluation.

    space=0  voxel-wise (correlate all voxels)
    space=1  Schaefer atlas — correlate ROI mean values (all ROIs)
    space=2  Schaefer atlas subset — only ROIs whose label matches one of `roi_name_patterns`
    """

    name: str
    space: int = 0
    atlas_n_rois: int = 200
    roi_name_patterns: Optional[List[str]] = None

    def __post_init__(self):
        if self.space not in {0, 1, 2}:
            raise ValueError(f"space must be 0/1/2 (got {self.space})")
        if self.space == 2 and not self.roi_name_patterns:
            raise ValueError("space=2 requires roi_name_patterns")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CorrelationConfig":
        return cls(
            name=d["name"],
            space=d.get("space", 0),
            atlas_n_rois=d.get("atlas_n_rois", 200),
            roi_name_patterns=d.get("roi_name_patterns"),
        )


# ---------- basis ----------

VALID_BASES = {"identity", "pca", "ica", "atlas", "random",
               "ridgecol", "cca", "pls", "hybrid"}


@dataclass
class BasisConfig:
    basis_type: str
    n_components: int = 32
    random_seed: int = 0
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.basis_type not in VALID_BASES:
            raise ValueError(
                f"basis_type {self.basis_type!r} must be one of {sorted(VALID_BASES)}"
            )


# ---------- predictor ----------

VALID_PREDICTORS = {"linear", "mlp"}
VALID_LOSS_MODES = {"coefficient", "map"}
VALID_MAP_LOSSES = {"l2", "l1", "correlation"}


@dataclass
class PredictorConfig:
    model_type: str  # "linear" | "mlp"
    output_dim: int = 0  # set during pipeline (= n_components after fit)
    input_dim: Optional[int] = None  # set during pipeline

    # MLP
    hidden_dims: Optional[List[int]] = None
    activation: str = "relu"
    dropout: float = 0.0

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # = ridge α for linear, AdamW WD for mlp
    batch_size: int = 32
    n_epochs: int = 30
    # Max global grad-norm for the MLP optimizer (None = off). Safety net against
    # map-loss divergence; clips before each optimizer step.
    grad_clip_norm: Optional[float] = None

    # Loss
    loss_mode: str = "coefficient"  # see docs/bases.md for recommendation
    map_loss_type: str = "l2"

    random_seed: int = 0

    # If True, predictor uses cognitive features only (no stim_encoding).
    # Requires DataConfig.single_contrast.
    cog_only: bool = False

    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model_type not in VALID_PREDICTORS:
            raise ValueError(
                f"model_type {self.model_type!r} must be one of {sorted(VALID_PREDICTORS)}"
            )
        if self.loss_mode not in VALID_LOSS_MODES:
            raise ValueError(
                f"loss_mode {self.loss_mode!r} must be one of {sorted(VALID_LOSS_MODES)}"
            )
        if self.map_loss_type not in VALID_MAP_LOSSES:
            raise ValueError(
                f"map_loss_type {self.map_loss_type!r} must be one of {sorted(VALID_MAP_LOSSES)}"
            )
        if self.model_type == "mlp" and self.hidden_dims is None:
            self.hidden_dims = [64, 32]


# ---------- data ----------

VALID_STIM_ENCODINGS = {"onehot", "contrast", "contrast_only"}


@dataclass
class DataConfig:
    """Dataset-side configuration.

    The dataset is identified by adapter_name; the adapter knows where files live.
    """

    adapter_name: str = "dfg"
    cog_base_cols: List[str] = field(default_factory=list)

    # Extra keyword arguments forwarded to create_adapter() — persisted to
    # config.json so evaluation rebuilds the adapter identically (e.g. ERA812's
    # include_groups / include_group_onehot). Must be JSON-serialisable.
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)

    stim_encoding: str = "onehot"  # "onehot" | "contrast" | "contrast_only"
    contrast_pairs: Optional[List[Tuple[str, str]]] = None

    # Path to a JSON filter file produced by utils/filter_subjects.py.
    subject_session_filter: Optional[Path] = None

    # Restrict the dataset to a stimulus/contrast NAME, or a list of names to
    # keep a mixed subset, e.g. "Words" or ["Words>Checkerboard", "Words"].
    # (To keep individual stimuli AND contrasts, use stim_encoding="contrast".)
    single_contrast: Optional[Any] = None

    # Force re-load even if a cache exists.
    force_recompute: bool = False

    def __post_init__(self):
        if self.stim_encoding == "mixed":
            self.stim_encoding = "contrast"
        if self.stim_encoding not in VALID_STIM_ENCODINGS:
            raise ValueError(
                f"stim_encoding {self.stim_encoding!r} must be one of {sorted(VALID_STIM_ENCODINGS)}"
            )
        if self.subject_session_filter is not None and isinstance(self.subject_session_filter, str):
            self.subject_session_filter = Path(self.subject_session_filter)
        if self.contrast_pairs and isinstance(self.contrast_pairs[0], list):
            self.contrast_pairs = [tuple(p) for p in self.contrast_pairs]


# ---------- run (top-level) ----------


@dataclass
class RunConfig:
    data_root: Path
    run_dir: Path

    basis_config: BasisConfig
    predictor_config: PredictorConfig
    data_config: DataConfig

    train_fraction: float = 0.8
    random_seed: int = 0

    # Stimulus contrast pairs to compute as evaluation contrasts (not training labels).
    contrast_pairs: Optional[List[Tuple[str, str]]] = None

    plot_debug_plots: bool = False

    # Which correlation spaces to score in LTO / data-informativeness.
    eval_correlation_configs: Optional[List[CorrelationConfig]] = None

    # If set, project all maps into this atlas-space *before* fitting basis/predictor.
    # Use with basis="identity" (ROI means as features) or basis="pca" (PCA on ROI means).
    train_atlas_cfg: Optional[CorrelationConfig] = None

    # If True, model the deviation from the per-contrast training population-mean map:
    # subtract the mean before fitting basis/predictor and add it back at reconstruction.
    # Gives supervised/hybrid bases (whose decode has no mean term) the common component.
    mean_map_offset: bool = False

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.run_dir = Path(self.run_dir)
        if self.contrast_pairs and isinstance(self.contrast_pairs[0], list):
            self.contrast_pairs = [tuple(p) for p in self.contrast_pairs]
        if self.eval_correlation_configs is not None:
            self.eval_correlation_configs = [
                CorrelationConfig.from_dict(c) if isinstance(c, dict) else c
                for c in self.eval_correlation_configs
            ]
        if isinstance(self.train_atlas_cfg, dict):
            self.train_atlas_cfg = CorrelationConfig.from_dict(self.train_atlas_cfg)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        d["run_dir"] = str(self.run_dir)
        if d["data_config"].get("subject_session_filter") is not None:
            d["data_config"]["subject_session_filter"] = str(
                d["data_config"]["subject_session_filter"]
            )
        if self.eval_correlation_configs is not None:
            d["eval_correlation_configs"] = [c.to_dict() for c in self.eval_correlation_configs]
        if self.train_atlas_cfg is not None:
            d["train_atlas_cfg"] = self.train_atlas_cfg.to_dict()
        return d

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        data = json.loads(Path(path).read_text())

        basis_config = BasisConfig(**data["basis_config"])
        predictor_config = PredictorConfig(**data["predictor_config"])
        data_config = DataConfig(**data["data_config"])

        eval_corr_raw = data.get("eval_correlation_configs")
        eval_correlation_configs = (
            [CorrelationConfig.from_dict(c) for c in eval_corr_raw]
            if eval_corr_raw is not None else None
        )

        raw_train_atlas = data.get("train_atlas_cfg")
        train_atlas_cfg = CorrelationConfig.from_dict(raw_train_atlas) if raw_train_atlas else None

        return cls(
            data_root=Path(data["data_root"]),
            run_dir=Path(data["run_dir"]),
            basis_config=basis_config,
            predictor_config=predictor_config,
            data_config=data_config,
            train_fraction=data.get("train_fraction", 0.8),
            random_seed=data.get("random_seed", 0),
            contrast_pairs=data.get("contrast_pairs"),
            plot_debug_plots=data.get("plot_debug_plots", False),
            eval_correlation_configs=eval_correlation_configs,
            train_atlas_cfg=train_atlas_cfg,
            mean_map_offset=data.get("mean_map_offset", False),
        )
