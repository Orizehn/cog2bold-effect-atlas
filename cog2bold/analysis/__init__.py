"""Analysis subpackage."""

from cog2bold.analysis.data_informativeness import (
    DataInformativenessArgs,
    run_data_informativeness,
)
from cog2bold.analysis.feature_informativeness import (
    FeatureScanArgs,
    run_feature_informativeness_scan,
)
from cog2bold.analysis.lto_curve_calibration import (
    calibrated_curve,
    load_verdicts,
    trend_test,
)

__all__ = [
    "DataInformativenessArgs",
    "run_data_informativeness",
    "FeatureScanArgs",
    "run_feature_informativeness_scan",
    "calibrated_curve",
    "load_verdicts",
    "trend_test",
]
