"""Profit-based metric adapters for Phase 4."""

from __future__ import annotations

from typing import Any

from creditrep.metrics.contract import MetricDirection, MetricExactness, MetricResult, MetricSpecification, MetricStatus

METRIC_VERSION = "1.0"

EMP_SPEC = MetricSpecification(
    metric_id="emp",
    metric_version=METRIC_VERSION,
    direction=MetricDirection.MAXIMIZE,
    exactness=MetricExactness.NOT_APPLICABLE,
    parameters={
        "labels": [0, 1],
        "positive_label": 1,
        "score_type": "probability_class_1",
        "source_metric_name": "Expected Maximum Profit",
    },
)

EMP_MISSING_PARAMETERS = (
    "b1",
    "c0",
    "c_star",
    "h(b1,c0)",
    "threshold_selection_policy",
)

EMP_UNSUPPORTED_WARNING = (
    "EMP remains unsupported in Phase 4 because the repository and cited papers do not provide enough validated "
    "business parameter values or distributions to compute a numeric result without inventing assumptions."
)


def compute_emp(y_true: Any, y_score: Any, *, parameters: dict[str, Any] | None = None) -> MetricResult:
    del y_true, y_score
    payload = dict(parameters or {})
    payload.update(
        {
            "decision": "unsupported_due_to_insufficient_specification",
            "missing_parameters": list(EMP_MISSING_PARAMETERS),
            "primary_source_formula": "MP = max_t P(t; b1, c0, c*) and EMP = E_h[MP]",
        }
    )
    return MetricResult(
        metric_id=EMP_SPEC.metric_id,
        metric_version=EMP_SPEC.metric_version,
        value=None,
        direction=EMP_SPEC.direction,
        status=MetricStatus.UNSUPPORTED,
        parameters=payload,
        exactness=MetricExactness.NOT_APPLICABLE,
        warnings=(EMP_UNSUPPORTED_WARNING,),
    )
