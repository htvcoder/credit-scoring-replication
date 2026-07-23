"""Metric contracts and validated Phase 4 metric implementations."""

from creditrep.metrics.calibration import BRIER_SCORE_SPEC, compute_brier_score
from creditrep.metrics.contract import (
    MetricDirection,
    MetricExactness,
    MetricResult,
    MetricSpecification,
    MetricStatus,
)
from creditrep.metrics.discrimination import PARTIAL_GINI_SPEC, ROC_AUC_SPEC, compute_partial_gini, compute_roc_auc
from creditrep.metrics.profit import EMP_SPEC, compute_emp
from creditrep.metrics.registry import (
    MetricConfig,
    compute_configured_metric,
    get_metric_specification,
    metric_config_hash,
    parse_metric_configs,
)
from creditrep.metrics.validation import MetricInputError, validate_binary_probability_inputs, validate_partial_gini_cutoff

__all__ = [
    "BRIER_SCORE_SPEC",
    "EMP_SPEC",
    "MetricConfig",
    "MetricInputError",
    "MetricDirection",
    "MetricExactness",
    "MetricResult",
    "MetricSpecification",
    "MetricStatus",
    "PARTIAL_GINI_SPEC",
    "ROC_AUC_SPEC",
    "compute_brier_score",
    "compute_configured_metric",
    "compute_emp",
    "compute_partial_gini",
    "compute_roc_auc",
    "get_metric_specification",
    "metric_config_hash",
    "parse_metric_configs",
    "validate_binary_probability_inputs",
    "validate_partial_gini_cutoff",
]
