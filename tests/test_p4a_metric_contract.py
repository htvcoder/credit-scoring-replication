from __future__ import annotations

import json
import math

import numpy as np
import pytest

from creditrep.evaluation.metrics import compute_binary_metrics
from creditrep.metrics import (
    MetricDirection,
    MetricExactness,
    MetricResult,
    MetricSpecification,
    MetricStatus,
)


def test_metric_result_serializes_to_json_compatible_dict() -> None:
    result = MetricResult(
        metric_id="roc_auc",
        metric_version="1.0",
        value=0.75,
        direction=MetricDirection.MAXIMIZE,
        status=MetricStatus.VALID,
        exactness=MetricExactness.EXACT,
        parameters={"labels": [0, 1]},
        warnings=("outer test labels were used only for evaluation",),
    )

    payload = result.to_dict()

    assert payload == {
        "direction": "maximize",
        "exactness": "exact",
        "metric_id": "roc_auc",
        "metric_version": "1.0",
        "parameters": {"labels": [0, 1]},
        "status": "valid",
        "value": 0.75,
        "warnings": ["outer test labels were used only for evaluation"],
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_metric_specification_serializes_parameters_deterministically() -> None:
    spec = MetricSpecification(
        metric_id="partial_gini_b_0_4",
        metric_version="1.0",
        direction="maximize",
        exactness="approximate",
        parameters={"b": 0.4, "labels": [0, 1]},
    )

    assert list(spec.to_dict()["parameters"]) == ["b", "labels"]
    assert spec.to_dict() == spec.to_dict()


def test_invalid_enum_status_and_exactness_are_rejected() -> None:
    with pytest.raises(ValueError, match="status"):
        MetricResult("auc", "1.0", 0.5, "maximize", "ok")
    with pytest.raises(ValueError, match="direction"):
        MetricResult("auc", "1.0", 0.5, "bigger", "valid")
    with pytest.raises(ValueError, match="exactness"):
        MetricSpecification("auc", "1.0", "maximize", "paper_exact")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invalid_numeric_value_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricResult("brier_score", "1.0", value, "minimize", "valid")


def test_valid_status_requires_value_but_undefined_may_store_none() -> None:
    with pytest.raises(ValueError, match="required"):
        MetricResult("roc_auc", "1.0", None, "maximize", "valid")

    undefined = MetricResult(
        "roc_auc",
        "1.0",
        None,
        "maximize",
        "undefined",
        warnings=("single-class fold",),
    )
    assert undefined.to_dict()["value"] is None
    assert undefined.to_dict()["warnings"] == ["single-class fold"]


def test_parameters_reject_nan_and_non_string_keys() -> None:
    with pytest.raises(ValueError, match="NaN"):
        MetricResult("emp", "1.0", None, "maximize", "unsupported", parameters={"alpha": math.nan})
    with pytest.raises(ValueError, match="keys"):
        MetricResult("emp", "1.0", None, "maximize", "unsupported", parameters={1: "bad"})


def test_exact_and_approximate_status_are_serialized() -> None:
    exact = MetricResult("roc_auc", "1.0", 1.0, "maximize", "valid", exactness="exact")
    approximate = MetricResult("emp", "1.0", None, "maximize", "unsupported", exactness="approximate")
    assert exact.to_dict()["exactness"] == "exact"
    assert approximate.to_dict()["exactness"] == "approximate"


def test_smoke_metrics_backward_compatibility_shape_is_unchanged() -> None:
    metrics = compute_binary_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.2, 0.8, 0.9]),
        y_pred=np.array([0, 0, 1, 1]),
        threshold=0.5,
    )

    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "test_row_count" in metrics
    assert "confusion_matrix" in metrics
    assert "metric_id" not in metrics
