"""Calibration metrics validated for Phase 4."""

from __future__ import annotations

from typing import Any

import numpy as np

from creditrep.metrics.contract import MetricDirection, MetricExactness, MetricResult, MetricSpecification, MetricStatus
from creditrep.metrics.validation import MetricInputError, validate_binary_probability_inputs

METRIC_VERSION = "1.0"

BRIER_SCORE_SPEC = MetricSpecification(
    metric_id="brier_score",
    metric_version=METRIC_VERSION,
    direction=MetricDirection.MINIMIZE,
    exactness=MetricExactness.EXACT,
    parameters={"labels": [0, 1], "positive_label": 1, "range": [0.0, 1.0], "score_type": "probability_class_1"},
)


def compute_brier_score(y_true: Any, y_score: Any) -> MetricResult:
    try:
        inputs = validate_binary_probability_inputs(y_true, y_score)
    except MetricInputError as exc:
        return MetricResult(
            metric_id=BRIER_SCORE_SPEC.metric_id,
            metric_version=BRIER_SCORE_SPEC.metric_version,
            value=None,
            direction=BRIER_SCORE_SPEC.direction,
            status=MetricStatus.FAILED,
            parameters=BRIER_SCORE_SPEC.parameters,
            exactness=BRIER_SCORE_SPEC.exactness,
            warnings=(str(exc),),
        )

    value = float(np.mean((inputs.y_score - inputs.y_true) ** 2))
    return MetricResult(
        metric_id=BRIER_SCORE_SPEC.metric_id,
        metric_version=BRIER_SCORE_SPEC.metric_version,
        value=value,
        direction=BRIER_SCORE_SPEC.direction,
        status=MetricStatus.VALID,
        parameters=BRIER_SCORE_SPEC.parameters,
        exactness=BRIER_SCORE_SPEC.exactness,
        warnings=(),
    )
