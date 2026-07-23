from __future__ import annotations

import json

from creditrep.metrics import MetricStatus, compute_emp


def test_emp_returns_unsupported_with_stable_missing_parameter_metadata() -> None:
    result = compute_emp([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])

    assert result.metric_id == "emp"
    assert result.status == MetricStatus.UNSUPPORTED
    assert result.value is None
    assert result.exactness == "not_applicable"
    assert result.parameters["decision"] == "unsupported_due_to_insufficient_specification"
    assert result.parameters["missing_parameters"] == [
        "b1",
        "c0",
        "c_star",
        "h(b1,c0)",
        "threshold_selection_policy",
    ]
    assert "do not provide enough validated business parameter values" in result.warnings[0]
    assert result.to_dict() == result.to_dict()
    json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)


def test_emp_preserves_explicit_parameters_without_creating_numeric_value() -> None:
    result = compute_emp(
        [0, 1],
        [0.2, 0.8],
        parameters={"source_note": "kept for provenance only"},
    )

    assert result.status == MetricStatus.UNSUPPORTED
    assert result.value is None
    assert result.parameters["source_note"] == "kept for provenance only"
    assert "missing_parameters" in result.parameters
