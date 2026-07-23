"""Discrimination metrics validated for Phase 4."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from creditrep.metrics.contract import MetricDirection, MetricExactness, MetricResult, MetricSpecification, MetricStatus
from creditrep.metrics.validation import (
    MetricInputError,
    ValidatedBinaryProbabilityInputs,
    validate_binary_probability_inputs,
    validate_partial_gini_cutoff,
)

METRIC_VERSION = "1.0"

ROC_AUC_SPEC = MetricSpecification(
    metric_id="roc_auc",
    metric_version=METRIC_VERSION,
    direction=MetricDirection.MAXIMIZE,
    exactness=MetricExactness.EXACT,
    parameters={"labels": [0, 1], "positive_label": 1, "score_type": "probability_class_1"},
)

PARTIAL_GINI_SPEC = MetricSpecification(
    metric_id="partial_gini",
    metric_version=METRIC_VERSION,
    direction=MetricDirection.MAXIMIZE,
    exactness=MetricExactness.EXACT,
    parameters={
        "b": 0.4,
        "labels": [0, 1],
        "normalization": "2 * roc_auc(subset) - 1",
        "positive_label": 1,
        "score_region": "y_score <= b",
        "score_type": "probability_class_1",
    },
)


def _result_from_spec(
    specification: MetricSpecification,
    *,
    value: float | None,
    status: MetricStatus | str,
    parameters: dict[str, Any] | None = None,
    warnings: tuple[str, ...] | list[str] = (),
    exactness: MetricExactness | str | None = None,
) -> MetricResult:
    return MetricResult(
        metric_id=specification.metric_id,
        metric_version=specification.metric_version,
        value=value,
        direction=specification.direction,
        status=status,
        parameters=parameters if parameters is not None else specification.parameters,
        exactness=exactness if exactness is not None else specification.exactness,
        warnings=warnings,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _roc_auc_from_validated(inputs: ValidatedBinaryProbabilityInputs) -> float | None:
    if inputs.positive_count == 0 or inputs.negative_count == 0:
        return None
    ranks = _average_ranks(inputs.y_score)
    positive_rank_sum = float(np.sum(ranks[inputs.y_true == 1]))
    auc = (
        positive_rank_sum - (inputs.positive_count * (inputs.positive_count + 1) / 2.0)
    ) / (inputs.positive_count * inputs.negative_count)
    return float(auc)


def compute_roc_auc(y_true: Any, y_score: Any) -> MetricResult:
    try:
        inputs = validate_binary_probability_inputs(y_true, y_score)
    except MetricInputError as exc:
        return _result_from_spec(ROC_AUC_SPEC, value=None, status=MetricStatus.FAILED, warnings=(str(exc),))

    auc = _roc_auc_from_validated(inputs)
    if auc is None:
        return _result_from_spec(
            ROC_AUC_SPEC,
            value=None,
            status=MetricStatus.UNDEFINED,
            warnings=("ROC AUC is undefined when y_true contains only one class.",),
        )
    return _result_from_spec(ROC_AUC_SPEC, value=auc, status=MetricStatus.VALID)


def partial_gini_specification(*, b: float = 0.4) -> MetricSpecification:
    cutoff = validate_partial_gini_cutoff(b)
    parameters = dict(PARTIAL_GINI_SPEC.parameters)
    parameters["b"] = cutoff
    return replace(PARTIAL_GINI_SPEC, parameters=parameters)


def compute_partial_gini(y_true: Any, y_score: Any, *, b: float = 0.4) -> MetricResult:
    try:
        cutoff = validate_partial_gini_cutoff(b)
        inputs = validate_binary_probability_inputs(y_true, y_score)
    except MetricInputError as exc:
        return _result_from_spec(
            partial_gini_specification(),
            value=None,
            status=MetricStatus.FAILED,
            warnings=(str(exc),),
        )

    specification = partial_gini_specification(b=cutoff)
    region_mask = inputs.y_score <= cutoff
    if not np.any(region_mask):
        return _result_from_spec(
            specification,
            value=None,
            status=MetricStatus.UNDEFINED,
            warnings=(f"Partial Gini is undefined because no observations satisfy y_score <= {cutoff}.",),
        )
    subset = ValidatedBinaryProbabilityInputs(y_true=inputs.y_true[region_mask], y_score=inputs.y_score[region_mask])
    auc = _roc_auc_from_validated(subset)
    if auc is None:
        return _result_from_spec(
            specification,
            value=None,
            status=MetricStatus.UNDEFINED,
            warnings=(f"Partial Gini is undefined because y_score <= {cutoff} contains only one class.",),
        )
    return _result_from_spec(specification, value=(2.0 * auc) - 1.0, status=MetricStatus.VALID)
