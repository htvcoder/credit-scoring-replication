from __future__ import annotations

import json
import math
from itertools import product

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from creditrep.metrics import MetricStatus, compute_brier_score, compute_partial_gini, compute_roc_auc


def pairwise_auc_reference(y_true: list[int], y_score: list[float]) -> float | None:
    positives = [score for label, score in zip(y_true, y_score, strict=True) if label == 1]
    negatives = [score for label, score in zip(y_true, y_score, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive_score, negative_score in product(positives, negatives):
        total += 1
        if positive_score > negative_score:
            wins += 1.0
        elif positive_score == negative_score:
            wins += 0.5
    return wins / total


def partial_gini_reference(y_true: list[int], y_score: list[float], *, b: float) -> float | None:
    subset = [(label, score) for label, score in zip(y_true, y_score, strict=True) if score <= b]
    if not subset:
        return None
    subset_true = [label for label, _ in subset]
    subset_score = [score for _, score in subset]
    auc = pairwise_auc_reference(subset_true, subset_score)
    if auc is None:
        return None
    return (2.0 * auc) - 1.0


def test_roc_auc_matches_reference_and_sklearn_on_known_case() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.10, 0.40, 0.35, 0.80, 0.90]

    result = compute_roc_auc(y_true, y_score)

    assert result.status == MetricStatus.VALID
    assert result.metric_id == "roc_auc"
    assert result.direction == "maximize"
    assert result.exactness == "exact"
    assert result.value == pytest.approx(pairwise_auc_reference(y_true, y_score))
    assert result.value == pytest.approx(roc_auc_score(y_true, y_score))


def test_roc_auc_handles_perfect_reversed_constant_and_imbalanced_cases() -> None:
    perfect = compute_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    reversed_case = compute_roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    constant = compute_roc_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
    imbalanced = compute_roc_auc([0, 0, 0, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.8])

    assert perfect.value == pytest.approx(1.0)
    assert reversed_case.value == pytest.approx(0.0)
    assert constant.value == pytest.approx(0.5)
    assert imbalanced.value == pytest.approx(1.0)


def test_roc_auc_reports_undefined_and_failed_inputs() -> None:
    single_class = compute_roc_auc([1, 1, 1], [0.2, 0.3, 0.4])
    mismatch = compute_roc_auc([0, 1], [0.2])
    nan_input = compute_roc_auc([0, 1], [0.2, math.nan])
    bad_probability = compute_roc_auc([0, 1], [0.2, 1.2])
    bad_labels = compute_roc_auc([0, 2], [0.2, 0.8])

    assert single_class.status == MetricStatus.UNDEFINED
    assert single_class.value is None
    assert mismatch.status == MetricStatus.FAILED
    assert nan_input.status == MetricStatus.FAILED
    assert bad_probability.status == MetricStatus.FAILED
    assert bad_labels.status == MetricStatus.FAILED


def test_brier_score_matches_hand_computed_values_and_range() -> None:
    perfect = compute_brier_score([0, 1], [0.0, 1.0])
    constant = compute_brier_score([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
    known = compute_brier_score([0, 1, 1], [0.1, 0.8, 0.6])

    assert perfect.status == MetricStatus.VALID
    assert perfect.metric_id == "brier_score"
    assert perfect.direction == "minimize"
    assert perfect.exactness == "exact"
    assert perfect.value == pytest.approx(0.0)
    assert constant.value == pytest.approx(0.25)
    assert known.value == pytest.approx(((0.1 - 0.0) ** 2 + (0.8 - 1.0) ** 2 + (0.6 - 1.0) ** 2) / 3.0)
    assert 0.0 <= known.value <= 1.0


def test_brier_score_rejects_invalid_inputs() -> None:
    single_class = compute_brier_score([1, 1, 1], [0.2, 0.3, 0.4])
    mismatch = compute_brier_score([0, 1], [0.2])
    bad_probability = compute_brier_score([0, 1], [0.2, -0.1])
    inf_input = compute_brier_score([0, 1], [0.2, math.inf])

    assert single_class.status == MetricStatus.VALID
    assert single_class.value == pytest.approx(np.mean((np.array([0.2, 0.3, 0.4]) - 1.0) ** 2))
    assert mismatch.status == MetricStatus.FAILED
    assert bad_probability.status == MetricStatus.FAILED
    assert inf_input.status == MetricStatus.FAILED


def test_partial_gini_matches_independent_reference_and_documents_parameters() -> None:
    y_true = [0, 1, 0, 1, 0, 1, 0]
    y_score = [0.05, 0.08, 0.10, 0.22, 0.24, 0.35, 0.55]

    result = compute_partial_gini(y_true, y_score, b=0.4)

    assert result.status == MetricStatus.VALID
    assert result.metric_id == "partial_gini"
    assert result.direction == "maximize"
    assert result.exactness == "exact"
    assert result.parameters["b"] == 0.4
    assert result.parameters["normalization"] == "2 * roc_auc(subset) - 1"
    assert result.value == pytest.approx(partial_gini_reference(y_true, y_score, b=0.4))


def test_partial_gini_handles_perfect_reversed_constant_and_parameterized_b() -> None:
    perfect = compute_partial_gini([0, 0, 1, 1], [0.05, 0.10, 0.15, 0.20], b=0.25)
    reversed_case = compute_partial_gini([0, 0, 1, 1], [0.20, 0.15, 0.10, 0.05], b=0.25)
    constant = compute_partial_gini([0, 1, 0, 1], [0.20, 0.20, 0.20, 0.20], b=0.25)
    broader_b = compute_partial_gini([0, 1, 0, 1, 0, 1], [0.05, 0.18, 0.20, 0.32, 0.35, 0.39], b=0.4)

    assert perfect.value == pytest.approx(1.0)
    assert reversed_case.value == pytest.approx(-1.0)
    assert constant.value == pytest.approx(0.0)
    assert broader_b.value == pytest.approx(partial_gini_reference([0, 1, 0, 1, 0, 1], [0.05, 0.18, 0.20, 0.32, 0.35, 0.39], b=0.4))


def test_partial_gini_is_row_permutation_invariant_with_ties() -> None:
    y_true = [0, 1, 0, 1, 0, 1]
    y_score = [0.10, 0.10, 0.20, 0.20, 0.30, 0.30]
    forward = compute_partial_gini(y_true, y_score, b=0.3)
    reversed_order = compute_partial_gini(list(reversed(y_true)), list(reversed(y_score)), b=0.3)

    assert forward.status == MetricStatus.VALID
    assert reversed_order.status == MetricStatus.VALID
    assert forward.value == pytest.approx(reversed_order.value)


def test_partial_gini_reports_invalid_and_undefined_cases() -> None:
    invalid_b = compute_partial_gini([0, 1], [0.1, 0.2], b=0.0)
    no_rows = compute_partial_gini([0, 1], [0.2, 0.3], b=0.1)
    single_class_region = compute_partial_gini([0, 0, 1], [0.1, 0.2, 0.9], b=0.2)
    mismatch = compute_partial_gini([0, 1], [0.2], b=0.4)
    nan_input = compute_partial_gini([0, 1], [0.2, math.nan], b=0.4)

    assert invalid_b.status == MetricStatus.FAILED
    assert no_rows.status == MetricStatus.UNDEFINED
    assert single_class_region.status == MetricStatus.UNDEFINED
    assert mismatch.status == MetricStatus.FAILED
    assert nan_input.status == MetricStatus.FAILED


def test_metric_results_are_json_serializable_and_deterministic() -> None:
    results = [
        compute_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        compute_brier_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        compute_partial_gini([0, 1, 0, 1], [0.05, 0.10, 0.15, 0.20], b=0.25),
    ]

    payload = [result.to_dict() for result in results]
    assert payload == [result.to_dict() for result in results]
    json.dumps(payload, sort_keys=True, allow_nan=False)
