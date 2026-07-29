"""JSON-safe contracts for non-publishable P6C neural training evidence.

These validators deliberately do not write artifacts.  P6C.1B-b owns capture and
publication once the nested-CV runner has an artifact boundary.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from creditrep.artifacts.exceptions import ArtifactError

NEURAL_ARTIFACT_SCHEMA_VERSION = "1.0"
RESULT_SCOPE = "mlp_nested_cv_validation"
PROBABILITY_SEMANTICS = "P(class 1) = P(bad/default)"
_MODEL_DEPTHS = {"mlp_1": 1, "mlp_3": 3, "mlp_5": 5}
_TRAINING_SCOPES = {"inner_candidate", "final_refit"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NeuralTrainingSummary(TypedDict):
    schema_version: str
    experiment_id: str
    dataset_id: str
    model_id: Literal["mlp_1", "mlp_3", "mlp_5"]
    outer_fold_id: str
    inner_fold_id: str | None
    candidate_id: str | int | None
    training_scope: Literal["inner_candidate", "final_refit"]
    hidden_depth: int
    hidden_layers: list[int]
    parameter_count: int
    framework: str
    framework_version: str
    requested_device: str
    resolved_device: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    early_stopping_enabled: bool
    early_stopping_triggered: bool
    patience: int
    min_delta: float
    stop_reason: str
    best_weights_restored: bool
    duration_seconds: float
    model_seed: int
    early_stopping_split_seed: int
    fair_budget_id: str
    probability_semantics: str
    publishable: Literal[False]
    result_scope: Literal["mlp_nested_cv_validation"]
    warnings: list[str]


class NeuralEpochRecord(TypedDict):
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float
    duration_seconds: float
    improved: bool
    finite: Literal[True]


class NeuralTrainingHistory(TypedDict):
    schema_version: str
    experiment_id: str
    dataset_id: str
    model_id: Literal["mlp_1", "mlp_3", "mlp_5"]
    outer_fold_id: str
    inner_fold_id: str | None
    candidate_id: str | int | None
    training_scope: Literal["inner_candidate", "final_refit"]
    epochs: list[NeuralEpochRecord]


class NeuralEarlyStoppingSplitMetadata(TypedDict):
    schema_version: str
    experiment_id: str
    dataset_id: str
    model_id: Literal["mlp_1", "mlp_3", "mlp_5"]
    outer_fold_id: str
    inner_fold_id: str | None
    candidate_id: str | int | None
    split_scope: Literal["inner_candidate", "final_refit"]
    source_partition: str
    strategy: Literal["stratified_holdout"]
    validation_fraction: float
    shuffle: Literal[True]
    split_seed: int
    split_hash: str
    source_row_count: int
    train_row_count: int
    validation_row_count: int
    source_class_counts: dict[str, int]
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int]
    overlap_count: Literal[0]
    union_matches_source: Literal[True]
    publishable: Literal[False]
    result_scope: Literal["mlp_nested_cv_validation"]


def _fail(field: str, detail: str) -> None:
    raise ArtifactError(f"Neural artifact {field} {detail}.")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(label, "must be an object")
    if not all(isinstance(key, str) for key in value):
        _fail(label, "must use string keys")
    return value


def _required(payload: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(payload))
    if missing:
        _fail(label, f"is missing required fields: {', '.join(missing)}")


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(field, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(field, f"must be >= {minimum}")
    return value


def _number(
    value: Any, field: str, *, minimum: float | None = None, positive: bool = False
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        _fail(field, "must be a finite number")
    result = float(value)
    if positive and result <= 0:
        _fail(field, "must be > 0")
    if minimum is not None and result < minimum:
        _fail(field, f"must be >= {minimum}")
    return result


def _string(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value:
        _fail(field, "must be a non-empty string")
    return value


def _seed(value: Any, field: str) -> None:
    _integer(value, field, minimum=0)


def _json_safe(value: Any, field: str = "payload") -> None:
    """Reject runtime objects before serialisation can produce an opaque error."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            _fail(field, "contains NaN or Infinity")
        return
    if isinstance(value, Mapping):
        if "state_dict" in value:
            _fail(field, "must not contain a state_dict")
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(field, "contains a non-string key")
            _json_safe(item, f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_safe(item, f"{field}[{index}]")
        return
    module = type(value).__module__
    if module.startswith("torch") or hasattr(value, "state_dict"):
        _fail(field, "must not contain a tensor, state_dict, or model object")
    _fail(field, f"contains unsupported object type {type(value).__name__}")


def _validate_context(
    payload: Mapping[str, Any], *, scope_key: str, label: str
) -> None:
    _required(
        payload,
        {
            "schema_version",
            "experiment_id",
            "dataset_id",
            "model_id",
            "outer_fold_id",
            "inner_fold_id",
            "candidate_id",
            scope_key,
        },
        label,
    )
    if payload["schema_version"] != NEURAL_ARTIFACT_SCHEMA_VERSION:
        _fail("schema_version", f"must equal {NEURAL_ARTIFACT_SCHEMA_VERSION!r}")
    for field in ("experiment_id", "dataset_id", "outer_fold_id"):
        _string(payload[field], field)
    model_id = payload["model_id"]
    if model_id not in _MODEL_DEPTHS:
        _fail("model_id", "must be one of mlp_1, mlp_3, mlp_5")
    _string(payload["inner_fold_id"], "inner_fold_id", nullable=True)
    candidate = payload["candidate_id"]
    if candidate is not None and (
        not isinstance(candidate, (str, int)) or isinstance(candidate, bool)
    ):
        _fail("candidate_id", "must be a string, integer, or null")
    if payload[scope_key] not in _TRAINING_SCOPES:
        _fail(scope_key, "must be inner_candidate or final_refit")


def _validate_non_publishable(payload: Mapping[str, Any]) -> None:
    if payload.get("publishable") is not False:
        _fail("publishable", "must be false")
    if payload.get("result_scope") != RESULT_SCOPE:
        _fail("result_scope", f"must equal {RESULT_SCOPE!r}")


def validate_neural_training_summary(value: Any) -> dict[str, Any]:
    """Validate one completed neural training summary without persisting it."""
    payload = _mapping(value, "summary")
    required = set(NeuralTrainingSummary.__annotations__)
    _required(payload, required, "summary")
    _json_safe(payload)
    _validate_context(payload, scope_key="training_scope", label="summary")
    model_id = payload["model_id"]
    depth = _integer(payload["hidden_depth"], "hidden_depth", minimum=1)
    if depth != _MODEL_DEPTHS[model_id]:
        _fail("hidden_depth", f"does not match {model_id}")
    layers = payload["hidden_layers"]
    if (
        not isinstance(layers, Sequence)
        or isinstance(layers, (str, bytes))
        or len(layers) != depth
    ):
        _fail("hidden_layers", "must have exactly hidden_depth entries")
    for index, layer in enumerate(layers):
        _integer(layer, f"hidden_layers[{index}]", minimum=1)
    _integer(payload["parameter_count"], "parameter_count", minimum=1)
    for field in (
        "framework",
        "framework_version",
        "requested_device",
        "resolved_device",
        "optimizer",
        "stop_reason",
        "fair_budget_id",
    ):
        _string(payload[field], field)
    _number(payload["learning_rate"], "learning_rate", positive=True)
    _number(payload["weight_decay"], "weight_decay", minimum=0)
    _integer(payload["batch_size"], "batch_size", minimum=1)
    max_epochs = _integer(payload["max_epochs"], "max_epochs", minimum=1)
    completed = _integer(payload["epochs_completed"], "epochs_completed", minimum=1)
    if completed > max_epochs:
        _fail("epochs_completed", "must be <= max_epochs")
    best_epoch = _integer(payload["best_epoch"], "best_epoch", minimum=1)
    if best_epoch > completed:
        _fail("best_epoch", "must be <= epochs_completed")
    _number(payload["best_validation_loss"], "best_validation_loss")
    for field in (
        "early_stopping_enabled",
        "early_stopping_triggered",
        "best_weights_restored",
    ):
        if not isinstance(payload[field], bool):
            _fail(field, "must be boolean")
    _integer(payload["patience"], "patience", minimum=1)
    _number(payload["min_delta"], "min_delta", minimum=0)
    _number(payload["duration_seconds"], "duration_seconds", minimum=0)
    _seed(payload["model_seed"], "model_seed")
    _seed(payload["early_stopping_split_seed"], "early_stopping_split_seed")
    if payload["probability_semantics"] != PROBABILITY_SEMANTICS:
        _fail("probability_semantics", f"must equal {PROBABILITY_SEMANTICS!r}")
    warnings = payload["warnings"]
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        _fail("warnings", "must be a list of strings")
    _validate_non_publishable(payload)
    return dict(payload)


def validate_neural_training_history(value: Any) -> dict[str, Any]:
    """Validate completed epoch-level evidence for one neural training run."""
    payload = _mapping(value, "history")
    _required(payload, set(NeuralTrainingHistory.__annotations__), "history")
    _json_safe(payload)
    _validate_context(payload, scope_key="training_scope", label="history")
    epochs = payload["epochs"]
    if not isinstance(epochs, list):
        _fail("epochs", "must be a list")
    for expected_epoch, record in enumerate(epochs, start=1):
        item = _mapping(record, f"epochs[{expected_epoch - 1}]")
        _required(
            item,
            set(NeuralEpochRecord.__annotations__),
            f"epochs[{expected_epoch - 1}]",
        )
        if (
            _integer(item["epoch"], f"epochs[{expected_epoch - 1}].epoch", minimum=1)
            != expected_epoch
        ):
            _fail(
                "epochs",
                "must start at 1 and increase consecutively without duplicates",
            )
        _number(item["train_loss"], f"epochs[{expected_epoch - 1}].train_loss")
        _number(
            item["validation_loss"], f"epochs[{expected_epoch - 1}].validation_loss"
        )
        _number(
            item["learning_rate"],
            f"epochs[{expected_epoch - 1}].learning_rate",
            positive=True,
        )
        _number(
            item["duration_seconds"],
            f"epochs[{expected_epoch - 1}].duration_seconds",
            minimum=0,
        )
        if not isinstance(item["improved"], bool):
            _fail(f"epochs[{expected_epoch - 1}].improved", "must be boolean")
        if item["finite"] is not True:
            _fail(
                f"epochs[{expected_epoch - 1}].finite",
                "must be true for a completed artifact",
            )
    return dict(payload)


def validate_neural_early_stopping_split_metadata(value: Any) -> dict[str, Any]:
    """Validate provenance for a train-only stratified early-stopping holdout."""
    payload = _mapping(value, "early-stopping split metadata")
    _required(
        payload,
        set(NeuralEarlyStoppingSplitMetadata.__annotations__),
        "early-stopping split metadata",
    )
    _json_safe(payload)
    _validate_context(
        payload, scope_key="split_scope", label="early-stopping split metadata"
    )
    if payload["strategy"] != "stratified_holdout":
        _fail("strategy", "must equal 'stratified_holdout'")
    _number(payload["validation_fraction"], "validation_fraction", positive=True)
    if float(payload["validation_fraction"]) >= 1:
        _fail("validation_fraction", "must be < 1")
    if payload["shuffle"] is not True:
        _fail("shuffle", "must be true")
    _seed(payload["split_seed"], "split_seed")
    if not isinstance(payload["split_hash"], str) or not _SHA256.fullmatch(
        payload["split_hash"]
    ):
        _fail("split_hash", "must be a lowercase 64-character SHA-256 hex digest")
    source = _integer(payload["source_row_count"], "source_row_count", minimum=0)
    train = _integer(payload["train_row_count"], "train_row_count", minimum=0)
    validation = _integer(
        payload["validation_row_count"], "validation_row_count", minimum=0
    )
    if source != train + validation:
        _fail("source_row_count", "must equal train_row_count + validation_row_count")
    class_counts: dict[str, Mapping[str, int]] = {}
    for field in (
        "source_class_counts",
        "train_class_counts",
        "validation_class_counts",
    ):
        counts = _mapping(payload[field], field)
        class_counts[field] = counts
        for label, count in counts.items():
            _integer(count, f"{field}.{label}", minimum=0)
    if set(class_counts["source_class_counts"]) != set(
        class_counts["train_class_counts"]
    ) or set(class_counts["source_class_counts"]) != set(
        class_counts["validation_class_counts"]
    ):
        _fail(
            "class_counts", "must use the same classes in source, train, and validation"
        )
    for label, source_count in class_counts["source_class_counts"].items():
        if (
            source_count
            != class_counts["train_class_counts"][label]
            + class_counts["validation_class_counts"][label]
        ):
            _fail("class_counts", f"do not reconcile for class {label!r}")
    if (
        sum(class_counts["source_class_counts"].values()) != source
        or sum(class_counts["train_class_counts"].values()) != train
        or sum(class_counts["validation_class_counts"].values()) != validation
    ):
        _fail("class_counts", "do not reconcile with row counts")
    if _integer(payload["overlap_count"], "overlap_count", minimum=0) != 0:
        _fail("overlap_count", "must be zero")
    if payload["union_matches_source"] is not True:
        _fail("union_matches_source", "must be true")
    source_partition = _string(payload["source_partition"], "source_partition")
    if source_partition in {"outer_test", "test"}:
        _fail("source_partition", "must not be outer_test or test")
    _validate_non_publishable(payload)
    return dict(payload)


def validate_training_summary_history(
    summary: Any, history: Any, *, tolerance: float = 1e-9
) -> None:
    """Validate that summary and history identify and describe the same training run."""
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or tolerance < 0
        or not math.isfinite(float(tolerance))
    ):
        _fail("tolerance", "must be a finite non-negative number")
    summary_payload = validate_neural_training_summary(summary)
    history_payload = validate_neural_training_history(history)
    for field in (
        "experiment_id",
        "dataset_id",
        "model_id",
        "outer_fold_id",
        "inner_fold_id",
        "candidate_id",
        "training_scope",
    ):
        if summary_payload[field] != history_payload[field]:
            _fail(field, "does not match between summary and history")
    records = history_payload["epochs"]
    if len(records) != summary_payload["epochs_completed"]:
        _fail("epochs", "count does not match summary epochs_completed")
    best_epoch = summary_payload["best_epoch"]
    if best_epoch > len(records):
        _fail("best_epoch", "is absent from history")
    best_loss = records[best_epoch - 1]["validation_loss"]
    if not math.isclose(
        float(best_loss),
        float(summary_payload["best_validation_loss"]),
        rel_tol=float(tolerance),
        abs_tol=float(tolerance),
    ):
        _fail("best_validation_loss", "does not match history at best_epoch")


def ensure_neural_payload_json_safe(value: Any) -> None:
    """Public JSON-safety guard for capture code added in P6C.1B-b."""
    _json_safe(value)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            "Neural artifact payload is not JSON-serializable."
        ) from exc


def validate_neural_fold_artifact_set(
    path: Any, *, expected_model_id: str | None = None
) -> None:
    """Validate the complete optional neural evidence set within one P5C fold.

    Classical folds deliberately have no neural directory.  A neural fold is
    identified by registry/model metadata, never by a filename convention.
    """
    from pathlib import Path

    root = Path(path)
    model = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    model_id = expected_model_id or model.get("model_id")
    if model_id not in _MODEL_DEPTHS:
        if (root / "neural").exists():
            _fail("neural", "is present for a non-neural fold")
        return
    manifest_path = root / "neural" / "neural_manifest.json"
    if not manifest_path.is_file():
        _fail("neural manifest", "is missing for a neural fold")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("Neural artifact neural manifest is malformed.") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != NEURAL_ARTIFACT_SCHEMA_VERSION
    ):
        _fail("neural manifest", "has an unsupported schema version")
    if (
        manifest.get("model_id") != model_id
        or manifest.get("publishable") is not False
        or manifest.get("result_scope") != RESULT_SCOPE
    ):
        _fail("neural manifest", "has inconsistent model or publication metadata")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        _fail("neural manifest candidates", "must be a non-empty list")
    ids = [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
    if len(ids) != len(candidates) or len({str(item) for item in ids}) != len(ids):
        _fail("neural manifest candidate_id", "must be unique")
    selected = manifest.get("selected_candidate_id")
    if selected not in ids:
        _fail("neural manifest selected_candidate_id", "does not identify a candidate")
    fair_budget_id = manifest.get("fair_budget_id")
    if not isinstance(fair_budget_id, str) or not fair_budget_id:
        _fail("neural manifest fair_budget_id", "must be a non-empty string")

    def load(relative: Any, label: str) -> dict[str, Any]:
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in relative.replace("\\", "/").split("/")
        ):
            _fail(label, "must be a safe relative path")
        candidate = root / relative
        if not candidate.is_file():
            _fail(label, "is missing")
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"Neural artifact {label} is malformed.") from exc

    expected_context = {
        "experiment_id": manifest.get("experiment_id"),
        "dataset_id": manifest.get("dataset_id"),
        "model_id": model_id,
        "outer_fold_id": manifest.get("outer_fold_id"),
    }
    for candidate in candidates:
        runs = candidate.get("runs") if isinstance(candidate, dict) else None
        if not isinstance(runs, list) or not runs:
            _fail("neural manifest candidate runs", "must be a non-empty list")
        for run in runs:
            summary = load(run.get("training_summary"), "candidate training_summary")
            history = load(run.get("training_history"), "candidate training_history")
            split = load(
                run.get("early_stopping_split"), "candidate early_stopping_split"
            )
            validate_neural_training_summary(summary)
            validate_neural_training_history(history)
            validate_neural_early_stopping_split_metadata(split)
            validate_training_summary_history(summary, history)
            for key, value in expected_context.items():
                if (
                    summary.get(key) != value
                    or history.get(key) != value
                    or split.get(key) != value
                ):
                    _fail(key, "does not match neural manifest")
            if (
                summary["candidate_id"] != candidate["candidate_id"]
                or history["candidate_id"] != candidate["candidate_id"]
                or split["candidate_id"] != candidate["candidate_id"]
            ):
                _fail("candidate_id", "does not match candidate manifest entry")
            if (
                summary["training_scope"] != "inner_candidate"
                or history["training_scope"] != "inner_candidate"
                or split["split_scope"] != "inner_candidate"
            ):
                _fail("training_scope", "is invalid for candidate evidence")
            if summary["fair_budget_id"] != fair_budget_id:
                _fail("fair_budget_id", "does not match neural manifest")
    final = manifest.get("final_refit")
    if not isinstance(final, dict) or final.get("selected_candidate_id") != selected:
        _fail("neural manifest final_refit", "does not match selected candidate")
    summary = load(final.get("training_summary"), "final-refit training_summary")
    history = load(final.get("training_history"), "final-refit training_history")
    split = load(final.get("early_stopping_split"), "final-refit early_stopping_split")
    validate_neural_training_summary(summary)
    validate_neural_training_history(history)
    validate_neural_early_stopping_split_metadata(split)
    validate_training_summary_history(summary, history)
    if (
        summary.get("training_scope") != "final_refit"
        or history.get("training_scope") != "final_refit"
        or split.get("split_scope") != "final_refit"
    ):
        _fail("training_scope", "is invalid for final-refit evidence")
    if summary.get("fair_budget_id") != fair_budget_id:
        _fail("fair_budget_id", "does not match between final refit and candidates")
