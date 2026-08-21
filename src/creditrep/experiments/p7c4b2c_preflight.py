"""Reusable execution and artifact lifecycle for P7C.4B.2c."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
from queue import Empty
import re
import shutil
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import psutil
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.registry import find_repo_root
from creditrep.locked_runtime_inputs import (
    LockedRuntimeInputError,
    OUTER_PROJECTION_RUNTIME_DATASETS,
    ValidatedRuntimeInputs,
    validate_locked_runtime_inputs,
)
from creditrep.process_tree import (
    close_process_queue,
    isolate_process_group,
    terminate_and_reap,
)
from creditrep.protocols.p7c4b2c import (
    ADDITIVE_COMPONENTS,
    EXECUTION_CLASSES,
    MODES,
    P7C4B2CError,
    SCHEMA_VERSION,
    SCIENTIFIC_COVERAGE_REASON_CODES,
    canonical_digest,
    project_validated,
    summarize,
    validate_canonical_plan,
    validate_plan,
    validate_sample,
)
from creditrep.protocols.p7c4b2d import (
    P7C4B2DError,
    PROJECTION_MAXIMUM_MONETARY_BUDGET,
    PROJECTION_MAXIMUM_RUNTIME_HOURS,
    TARGET_PROJECTION_PREFLIGHT_STAGE,
    dataset_binding_digest,
    normalize_target_output,
    projection_preflight_resource_policy,
    select_authorized_scope,
    TARGET_CANARY_STAGE,
    TARGET_EXECUTION_STAGES,
    validate_effective_authorization,
)
from creditrep.strict_json import StrictJSONError, load_strict_json_object

EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_INCOMPLETE = 3
EXIT_AUTHORIZATION = 4
Workload = Callable[[dict[str, Any], Path], dict[str, Any]]
PROVENANCE_SCHEMA_VERSION = 1
RUNTIME_STATE_SCHEMA_VERSION = 2
RUNTIME_CLOCK_SKEW_SECONDS = 5.0


def _lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None
    )


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, codes: list[str]) -> Any:
    try:
        return load_strict_json_object(path)
    except StrictJSONError as exc:
        codes.append(str(exc))
    return None


def _safe_digest(value: Any, codes: list[str]) -> str | None:
    try:
        return sha256_canonical(value)
    except (TypeError, ValueError):
        codes.append("non_canonical_or_non_finite_artifact")
        return None


def git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root.as_posix()}", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if result.returncode or len(value) != 40:
        raise P7C4B2CError("git_provenance_missing")
    return value


def capture_environment(repo_root: Path) -> dict[str, Any]:
    import platform
    import sys

    value = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_head(repo_root),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "captured_utc": _utc(),
    }
    value["environment_digest"] = canonical_digest(value, "environment_digest")
    return value


def synthetic_outer_refit(task: dict[str, Any], _repo_root: Path) -> dict[str, Any]:
    """Tiny real preprocessing/fit/predict/metric/serialization fixture path."""
    rng = np.random.default_rng(task["seed"])
    rows = 36
    features = rng.normal(size=(rows, 5))
    features[0, 0] = np.nan
    target = (np.nan_to_num(features[:, 0]) + 0.25 * features[:, 1] > 0).astype(int)
    train, test = np.arange(24), np.arange(24, rows)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    transformer = Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler())])
    train_x = transformer.fit_transform(features[train])
    test_x = transformer.transform(features[test])
    timings["preprocessing_elapsed_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    estimator = LogisticRegression(random_state=task["seed"], max_iter=50)
    estimator.fit(train_x, target[train])
    timings["model_fit_elapsed_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    scores = estimator.predict_proba(test_x)[:, 1]
    timings["prediction_elapsed_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    metric = float(roc_auc_score(target[test], scores))
    timings["metric_elapsed_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    serialized = json.dumps(
        {"coefficient": estimator.coef_.tolist(), "metric": metric},
        sort_keys=True,
    )
    timings["other_measured_orchestration_elapsed_seconds"] = (
        time.perf_counter() - started
    )
    return {
        "timings": timings,
        "result": {"metric": metric, "serialized_bytes": len(serialized)},
        "preprocessing_identity": "synthetic_simple_imputer_standard_scaler",
        "input_identity": {
            "fixture": "p7c4b2c_tiny_classification_v1",
            "seed": task["seed"],
            "rows": rows,
            "features": 5,
        },
        "limitations": [
            "synthetic_model_is_logistic_regression_not_mlp",
            "synthetic_fixture_is_not_target_runtime_evidence",
        ],
    }


def canonical_outer_refit(
    task: dict[str, Any],
    repo_root: Path,
    *,
    locked_runtime_inputs: ValidatedRuntimeInputs,
) -> dict[str, Any]:
    """Run the canonical preprocessing/refit primitive for one authorized target unit."""
    from creditrep.datasets import load_dataset
    from creditrep.experiments.model_validation import _fit_for_partition
    from creditrep.experiments.p7c3_feasibility import adapt_mlp_feasibility_candidate
    from creditrep.splitting import create_nested_cv_definition

    dataset_id = task["dataset_id"]
    try:
        source_hash = locked_runtime_inputs.source_hashes[dataset_id]
    except KeyError as exc:
        raise P7C4B2CError("locked_runtime_input_dataset_missing") from exc
    dataset = load_dataset(
        dataset_id,
        repo_root=repo_root,
        registry=locked_runtime_inputs.registry,
        expected_source_sha256=source_hash,
    )
    nested = create_nested_cv_definition(
        dataset,
        dataset_checksum=source_hash,
        outer_n_repeats=task["outer_repeat"] + 1,
        outer_n_splits=2,
        inner_n_splits=5,
        random_seed=42,
    )
    outer = nested.outer_folds[task["outer_repeat"] * 2 + task["outer_fold"]]
    parameters = adapt_mlp_feasibility_candidate(
        task["model_id"],
        {"id": task["candidate_id"], **task["candidate"]},
        {
            "optimizer": "adam",
            "batch_size": 32,
            "max_epochs": 200,
            "device_policy": "cpu",
            "early_stopping": {"enabled": True, "patience": 20, "min_delta": 0.0001},
        },
    )
    timings: dict[str, float] = {}
    estimator, pipeline, test, _ = _fit_for_partition(
        dataset=dataset,
        model_id=task["model_id"],
        parameters=parameters,
        seed=task["seed"],
        outer_id=outer.outer_fold_id,
        inner_id="final_refit",
        candidate_id=task["candidate_id"],
        train_indices=outer.train_indices,
        evaluation_indices=outer.test_indices,
        protocol_config=locked_runtime_inputs.protocol_config,
        model_stage="outer_refit",
        timing_sink=timings,
    )
    started = time.perf_counter()
    scores = estimator.predict_proba(test)[:, 1]
    timings["prediction_elapsed_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    metric = float(roc_auc_score(dataset.target.iloc[list(outer.test_indices)], scores))
    timings["metric_elapsed_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    model_metadata = {
        "class": estimator.__class__.__name__,
        "parameters": parameters,
        "preprocessing": pipeline.get_metadata(),
        "metric": metric,
    }
    json.dumps(model_metadata, sort_keys=True, default=str)
    timings["other_measured_orchestration_elapsed_seconds"] = (
        time.perf_counter() - started
    )
    return {
        "timings": timings,
        "result": {"metric": metric},
        "preprocessing_identity": sha256_canonical(pipeline.get_metadata()),
        "input_identity": {
            "dataset_id": task["dataset_id"],
            "dataset_checksum": source_hash,
            "outer_split_hash": outer.split_hash,
        },
        "limitations": [
            "candidate_is_complexity_proxy_not_observed_canonical_selection"
        ],
    }


def _task_subset(
    plan: dict[str, Any], mode: str, max_samples: int | None
) -> list[dict[str, Any]]:
    tasks = [task for task in plan["tasks"] if task["mode"] == mode]
    if max_samples is None:
        return tasks
    if max_samples < 1:
        raise P7C4B2CError("invalid_sample_budget")
    warmups = [task for task in tasks if task["classification"] == "warmup"]
    measured = [task for task in tasks if task["classification"] == "measured"]
    return warmups[: min(max_samples, len(warmups))] + measured[:max_samples]


def _workload_for_execution_class(execution_class: str) -> Workload:
    """Closed production mapping; callers never select a workload callable."""
    mapping: dict[str, Workload] = {
        "synthetic_validation": synthetic_outer_refit,
        "target_preflight": canonical_outer_refit,
    }
    try:
        return mapping[execution_class]
    except KeyError as exc:
        raise P7C4B2CError("unsupported_execution_class") from exc


def _parse_utc(value: Any, code: str) -> datetime:
    if not isinstance(value, str):
        raise P7C4B2CError(code)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise P7C4B2CError(code) from exc
    if parsed.tzinfo is None:
        raise P7C4B2CError(code)
    return parsed.astimezone(timezone.utc)


def _existing_filesystem_anchor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise P7C4B2CError("live_disk_lookup_failed")
        candidate = parent
    return candidate


def _require_live_disk(
    output: Path,
    minimum_free_bytes: int,
    disk_usage_provider: Callable[[Path], Any],
) -> None:
    if (
        not isinstance(minimum_free_bytes, int)
        or isinstance(minimum_free_bytes, bool)
        or minimum_free_bytes < 1
    ):
        raise P7C4B2CError("disk_policy_mismatch")
    try:
        usage = disk_usage_provider(_existing_filesystem_anchor(output))
        free = usage.free
    except (OSError, AttributeError, TypeError):
        raise P7C4B2CError("live_disk_lookup_failed") from None
    if not isinstance(free, int) or isinstance(free, bool) or free < minimum_free_bytes:
        raise P7C4B2CError("insufficient_live_disk")


def _canonical_authorized_tasks(
    plan: dict[str, Any], authorization: dict[str, Any], mode: str
) -> list[dict[str, Any]]:
    """Resolve the only target task payloads from the validated canonical plan."""
    validate_plan(plan)
    if authorization.get("execution_mode") != mode:
        raise P7C4B2CError("execution_mode_mismatch")
    task_ids = authorization.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids)):
        raise P7C4B2CError("authorization_task_scope_mismatch")
    by_id = {task["sample_id"]: task for task in plan["tasks"] if task["mode"] == mode}
    try:
        tasks = [by_id[task_id] for task_id in task_ids]
    except (KeyError, TypeError) as exc:
        raise P7C4B2CError("authorization_task_scope_mismatch") from exc
    try:
        expected_ids = select_authorized_scope(
            plan, mode, authorization.get("execution_stage")
        )["task_ids"]
    except P7C4B2DError as exc:
        raise P7C4B2CError(str(exc)) from exc
    if task_ids != expected_ids:
        raise P7C4B2CError("authorization_task_scope_mismatch")
    return tasks


def _task_set_digest(tasks: list[dict[str, Any]]) -> str:
    return sha256_canonical(tasks)


def _authorization_provenance(
    authorization: dict[str, Any],
    normalized_output: str,
    worker_count: int,
    canonical_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    value = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "authorization_digest": authorization["authorization_digest"],
        "proposal_digest": authorization["proposal_digest"],
        "target_environment_digest": authorization["target_environment_digest"],
        "locked_runtime_inputs_digest": authorization["locked_runtime_inputs_digest"],
        "plan_digest": authorization["plan_digest"],
        "git_commit": authorization["git_commit"],
        "authorization_created_at": authorization["created_at"],
        "authorization_expires_at": authorization["expires_at"],
        "execution_stage": authorization["execution_stage"],
        "execution_mode": authorization["execution_mode"],
        "task_ids": list(authorization["task_ids"]),
        "canonical_task_set_digest": _task_set_digest(canonical_tasks),
        "normalized_output_directory": normalized_output,
        "vm_count": authorization["vm_count"],
        "worker_count": worker_count,
        "maximum_task_count": authorization["maximum_task_count"],
        "maximum_runtime_hours": authorization["maximum_runtime_hours"],
        "maximum_monetary_budget": authorization["maximum_monetary_budget"],
        "hourly_price": authorization["hourly_price"],
        "currency": authorization["currency"],
        "minimum_free_disk_bytes": authorization["minimum_free_disk_bytes"],
        "resource_policy": authorization["resource_policy"],
    }
    if authorization["execution_stage"] == TARGET_PROJECTION_PREFLIGHT_STAGE:
        value.update(
            {
                "dataset_ids": list(authorization["dataset_ids"]),
                "dataset_hashes": dict(authorization["dataset_hashes"]),
                "dataset_binding_digest": authorization["dataset_binding_digest"],
            }
        )
    return value


def _validate_resume_provenance(
    manifest: dict[str, Any],
    authorization: dict[str, Any],
    normalized_output: str,
    canonical_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    stored = manifest.get("authorization_provenance")
    expected = _authorization_provenance(
        authorization, normalized_output, manifest.get("worker_count"), canonical_tasks
    )
    if (
        not isinstance(stored, dict)
        or stored.get("schema_version") != PROVENANCE_SCHEMA_VERSION
    ):
        raise P7C4B2CError("target_resume_provenance_missing")
    if stored != expected:
        raise P7C4B2CError("authorization_provenance_mismatch")
    # The persisted manifest is evidence, never an alternate workload source.
    if manifest.get("expected_tasks") != canonical_tasks:
        raise P7C4B2CError("canonical_task_manifest_mismatch")
    return stored


def _runtime_state_digest(state: dict[str, Any]) -> str:
    return sha256_canonical(
        {key: value for key, value in state.items() if key != "state_digest"}
    )


def _runtime_checkpoint(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation": state["generation"],
        "accumulated_elapsed_seconds": state["accumulated_elapsed_seconds"],
        "consumed_monetary_budget": state["consumed_monetary_budget"],
        "failed_task_count": state["failed_task_count"],
        "last_accounted_at": state["last_accounted_at"],
        "state_digest": state["state_digest"],
    }


def _scientific_coverage_result(coverage: dict[str, Any]) -> dict[str, Any]:
    reason_codes = (
        list(SCIENTIFIC_COVERAGE_REASON_CODES)
        if coverage.get("incomplete_strata")
        else []
    )
    return {
        "valid": not reason_codes,
        "reason_codes": reason_codes,
        "minimum_repetitions": coverage.get("minimum_repetitions"),
        "incomplete_strata": coverage.get("incomplete_strata", []),
    }


def _validate_persisted_target_contract(
    run_dir: Path,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    expected_tasks: list[dict[str, Any]],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    """Validate immutable target-preflight bindings without re-authorizing a run."""
    if manifest.get("execution_class") != "target_preflight":
        return []
    codes: list[str] = []
    provenance = manifest.get("authorization_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION
    ):
        return ["target_resume_provenance_missing"]
    expected_ids = [task.get("sample_id") for task in expected_tasks]
    try:
        canonical_stage_ids = select_authorized_scope(
            plan,
            manifest.get("mode"),
            provenance.get("execution_stage"),
        )["task_ids"]
    except P7C4B2DError:
        canonical_stage_ids = []
        codes.append("authorization_provenance_mismatch")
    if (
        provenance.get("execution_stage") not in TARGET_EXECUTION_STAGES
        or provenance.get("execution_mode") != manifest.get("mode")
        or provenance.get("plan_digest") != plan.get("plan_digest")
        or provenance.get("task_ids") != expected_ids
        or expected_ids != canonical_stage_ids
        or provenance.get("canonical_task_set_digest")
        != _task_set_digest(expected_tasks)
        or not isinstance(provenance.get("maximum_task_count"), int)
        or isinstance(provenance.get("maximum_task_count"), bool)
        or len(expected_tasks) > provenance.get("maximum_task_count", -1)
        or manifest.get("run_id") != run_dir.name
        or os.path.normcase(provenance.get("normalized_output_directory", ""))
        != os.path.normcase(str(run_dir))
        or not _lower_hex(provenance.get("authorization_digest"), 64)
        or not _lower_hex(provenance.get("proposal_digest"), 64)
        or not _lower_hex(provenance.get("target_environment_digest"), 64)
        or not isinstance(provenance.get("locked_runtime_inputs_digest"), str)
        or re.fullmatch(
            r"[0-9a-fA-F]{64}", provenance.get("locked_runtime_inputs_digest", "")
        )
        is None
    ):
        codes.append("authorization_provenance_mismatch")
    if provenance.get("execution_stage") == TARGET_PROJECTION_PREFLIGHT_STAGE:
        try:
            expected_policy = projection_preflight_resource_policy(manifest.get("mode"))
        except P7C4B2DError:
            expected_policy = None
        if provenance.get("resource_policy") != expected_policy:
            codes.append("authorization_provenance_mismatch")
        dataset_ids = provenance.get("dataset_ids")
        dataset_hashes = provenance.get("dataset_hashes")
        try:
            expected_dataset_binding = dataset_binding_digest(
                dataset_ids=OUTER_PROJECTION_RUNTIME_DATASETS,
                dataset_hashes=dataset_hashes,
                locked_runtime_inputs_digest=provenance.get(
                    "locked_runtime_inputs_digest"
                ),
                plan_digest=provenance.get("plan_digest"),
            )
        except (TypeError, ValueError):
            expected_dataset_binding = None
        if (
            dataset_ids != list(OUTER_PROJECTION_RUNTIME_DATASETS)
            or not isinstance(dataset_hashes, dict)
            # Dataset hashes are a mapping: JSON serialization may reorder
            # its keys, while dataset_ids deliberately remains ordered.
            or set(dataset_hashes) != set(OUTER_PROJECTION_RUNTIME_DATASETS)
            or provenance.get("dataset_binding_digest") != expected_dataset_binding
        ):
            codes.append("outer_dataset_binding_mismatch")
        try:
            locked = validate_locked_runtime_inputs(
                provenance.get("locked_runtime_inputs_digest"),
                repo_root or find_repo_root(),
                dataset_ids=OUTER_PROJECTION_RUNTIME_DATASETS,
            )
            if dataset_hashes != locked.source_hashes:
                codes.append("outer_dataset_hash_mismatch")
        except LockedRuntimeInputError:
            codes.append("locked_runtime_input_mismatch")
    runtime_codes: list[str] = []
    state = _read_json(run_dir / "authorization_runtime.json", runtime_codes)
    if runtime_codes or not state:
        codes.append("runtime_state_invalid")
        return codes
    required_state_fields = {
        "schema_version",
        "run_id",
        "authorization_digest",
        "proposal_digest",
        "target_environment_digest",
        "plan_digest",
        "normalized_output_directory",
        "maximum_runtime_hours",
        "maximum_monetary_budget",
        "hourly_price",
        "vm_count",
        "resource_policy",
        "runtime_started_at",
        "last_accounted_at",
        "accumulated_elapsed_seconds",
        "consumed_monetary_budget",
        "failed_task_count",
        "generation",
        "state_digest",
    }
    if (
        set(state) != required_state_fields
        or state.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION
    ):
        codes.append("runtime_state_invalid")
        return codes
    expected_binding = {
        "run_id": manifest.get("run_id"),
        "authorization_digest": provenance.get("authorization_digest"),
        "proposal_digest": provenance.get("proposal_digest"),
        "target_environment_digest": provenance.get("target_environment_digest"),
        "plan_digest": provenance.get("plan_digest"),
        "normalized_output_directory": provenance.get("normalized_output_directory"),
        "maximum_runtime_hours": provenance.get("maximum_runtime_hours"),
        "maximum_monetary_budget": provenance.get("maximum_monetary_budget"),
        "hourly_price": provenance.get("hourly_price"),
        "vm_count": provenance.get("vm_count"),
        "resource_policy": provenance.get("resource_policy"),
        "runtime_started_at": (manifest.get("runtime_origin") or {}).get(
            "runtime_started_at"
        ),
    }
    if any(state.get(key) != value for key, value in expected_binding.items()):
        codes.append("runtime_state_provenance_mismatch")
    if state.get("state_digest") != _runtime_state_digest(state) or manifest.get(
        "runtime_checkpoint"
    ) != _runtime_checkpoint(state):
        codes.append("runtime_state_rollback_or_integrity_failure")
    elapsed = state.get("accumulated_elapsed_seconds")
    generation = state.get("generation")
    maximum_hours = state.get("maximum_runtime_hours")
    consumed = state.get("consumed_monetary_budget")
    failed = state.get("failed_task_count")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed < 0
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
        or not isinstance(maximum_hours, (int, float))
        or isinstance(maximum_hours, bool)
        or not math.isfinite(maximum_hours)
        or maximum_hours <= 0
        or elapsed > maximum_hours * 3600.0
        or not isinstance(consumed, str)
        or not isinstance(failed, int)
        or isinstance(failed, bool)
        or failed < 0
        or failed > 0
    ):
        codes.append("runtime_state_invalid")
    try:
        consumed_value = Decimal(consumed)
        budget_value = Decimal(str(state.get("maximum_monetary_budget")))
    except (InvalidOperation, TypeError, ValueError):
        consumed_value = budget_value = Decimal("-1")
    if (
        not consumed_value.is_finite()
        or consumed_value < 0
        or not budget_value.is_finite()
        or budget_value <= 0
        or consumed_value > budget_value
    ):
        codes.append("runtime_state_invalid")
    try:
        expected_consumed = (
            Decimal(str(state.get("hourly_price")))
            * Decimal(state.get("vm_count"))
            * Decimal(str(elapsed))
            / Decimal(3600)
        )
    except (InvalidOperation, TypeError, ValueError):
        expected_consumed = Decimal("-1")
    if consumed_value != expected_consumed:
        codes.append("runtime_state_invalid")
    try:
        started = _parse_utc(state["runtime_started_at"], "runtime_state_invalid")
        last = _parse_utc(state["last_accounted_at"], "runtime_state_invalid")
        created = _parse_utc(
            provenance["authorization_created_at"], "runtime_state_invalid"
        )
        manifest_created = _parse_utc(
            manifest.get("created_utc"), "runtime_state_invalid"
        )
        if started < created or started > manifest_created or last < started:
            codes.append("runtime_state_timestamp_invalid")
        if elapsed + RUNTIME_CLOCK_SKEW_SECONDS < (last - started).total_seconds():
            codes.append("runtime_state_invalid")
    except (KeyError, P7C4B2CError):
        codes.append("runtime_state_invalid")
    return codes


def _validate_target_source_identity(
    manifest: dict[str, Any],
    environment: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[str]:
    """Cross-bind persisted target artifacts to the authorized source commit."""
    if manifest.get("execution_class") != "target_preflight":
        return []
    provenance = manifest.get("authorization_provenance")
    if not isinstance(provenance, dict):
        return []  # The persisted-contract validator reports the missing provenance.
    authorized_commit = provenance.get("git_commit")
    commits = [authorized_commit, environment.get("git_commit")]
    commits.extend(record.get("git_commit") for record in records)
    if not all(_lower_hex(commit, 40) for commit in commits) or any(
        commit != authorized_commit for commit in commits
    ):
        return ["git_provenance_mismatch"]
    return []


def _validate_runtime_state(
    state: dict[str, Any],
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    wall_now: datetime,
) -> float:
    origin = manifest.get("runtime_origin")
    checkpoint = manifest.get("runtime_checkpoint")
    if (
        not isinstance(state, dict)
        or not isinstance(origin, dict)
        or not isinstance(checkpoint, dict)
    ):
        raise P7C4B2CError("runtime_state_invalid")
    required = {
        "schema_version",
        "run_id",
        "authorization_digest",
        "proposal_digest",
        "target_environment_digest",
        "plan_digest",
        "normalized_output_directory",
        "maximum_runtime_hours",
        "maximum_monetary_budget",
        "hourly_price",
        "vm_count",
        "resource_policy",
        "runtime_started_at",
        "last_accounted_at",
        "accumulated_elapsed_seconds",
        "consumed_monetary_budget",
        "failed_task_count",
        "generation",
        "state_digest",
    }
    if (
        set(state) != required
        or state.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION
    ):
        raise P7C4B2CError("runtime_state_invalid")
    expected_binding = {
        "run_id": manifest.get("run_id"),
        "authorization_digest": provenance.get("authorization_digest"),
        "proposal_digest": provenance.get("proposal_digest"),
        "target_environment_digest": provenance.get("target_environment_digest"),
        "plan_digest": provenance.get("plan_digest"),
        "normalized_output_directory": provenance.get("normalized_output_directory"),
        "maximum_runtime_hours": provenance.get("maximum_runtime_hours"),
        "maximum_monetary_budget": provenance.get("maximum_monetary_budget"),
        "hourly_price": provenance.get("hourly_price"),
        "vm_count": provenance.get("vm_count"),
        "resource_policy": provenance.get("resource_policy"),
        "runtime_started_at": origin.get("runtime_started_at"),
    }
    if any(state.get(key) != value for key, value in expected_binding.items()):
        raise P7C4B2CError("runtime_state_provenance_mismatch")
    if state.get("state_digest") != _runtime_state_digest(
        state
    ) or checkpoint != _runtime_checkpoint(state):
        raise P7C4B2CError("runtime_state_rollback_or_integrity_failure")
    elapsed = state.get("accumulated_elapsed_seconds")
    generation = state.get("generation")
    consumed = state.get("consumed_monetary_budget")
    failed = state.get("failed_task_count")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed < 0
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
        or not isinstance(consumed, str)
        or not isinstance(failed, int)
        or isinstance(failed, bool)
        or failed < 0
        or failed > 0
    ):
        raise P7C4B2CError("runtime_state_invalid")
    started = _parse_utc(state["runtime_started_at"], "runtime_state_invalid")
    last = _parse_utc(state["last_accounted_at"], "runtime_state_invalid")
    created = _parse_utc(
        provenance["authorization_created_at"], "runtime_state_invalid"
    )
    manifest_created = _parse_utc(manifest.get("created_utc"), "runtime_state_invalid")
    if wall_now < last - timedelta(seconds=RUNTIME_CLOCK_SKEW_SECONDS):
        raise P7C4B2CError("runtime_clock_rollback")
    if (
        started < created
        or started > manifest_created
        or started > wall_now + timedelta(seconds=RUNTIME_CLOCK_SKEW_SECONDS)
    ):
        raise P7C4B2CError("runtime_state_timestamp_invalid")
    if last < started or last > wall_now + timedelta(
        seconds=RUNTIME_CLOCK_SKEW_SECONDS
    ):
        raise P7C4B2CError("runtime_state_timestamp_invalid")
    if float(elapsed) + RUNTIME_CLOCK_SKEW_SECONDS < (last - started).total_seconds():
        raise P7C4B2CError("runtime_state_invalid")
    maximum = float(provenance["maximum_runtime_hours"]) * 3600.0
    if elapsed > maximum:
        raise P7C4B2CError("runtime_state_invalid")
    try:
        consumed_value = Decimal(consumed)
        budget = Decimal(str(provenance["maximum_monetary_budget"]))
    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        raise P7C4B2CError("runtime_state_invalid") from exc
    if (
        not consumed_value.is_finite()
        or consumed_value < 0
        or not budget.is_finite()
        or budget <= 0
        or consumed_value > budget
    ):
        raise P7C4B2CError("runtime_state_invalid")
    expected_consumed = (
        Decimal(str(provenance["hourly_price"]))
        * Decimal(provenance["vm_count"])
        * Decimal(str(elapsed))
        / Decimal(3600)
    )
    if consumed_value != expected_consumed:
        raise P7C4B2CError("runtime_state_invalid")
    return max(float(elapsed), (wall_now - started).total_seconds())


def _persist_runtime_state(
    run_dir: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    elapsed: float,
    wall_now: datetime,
) -> None:
    state["accumulated_elapsed_seconds"] = max(
        float(state["accumulated_elapsed_seconds"]), elapsed
    )
    rate = Decimal(str(state["hourly_price"])) * Decimal(state["vm_count"])
    consumed = rate * Decimal(str(state["accumulated_elapsed_seconds"])) / Decimal(3600)
    state["consumed_monetary_budget"] = format(consumed, "f")
    state["last_accounted_at"] = wall_now.astimezone(timezone.utc).isoformat()
    state["generation"] += 1
    state["state_digest"] = _runtime_state_digest(state)
    _atomic_json(run_dir / "authorization_runtime.json", state)
    manifest["runtime_checkpoint"] = _runtime_checkpoint(state)
    _atomic_json(run_dir / "manifest.json", manifest)


def _deadline(
    provenance: dict[str, Any],
    state: dict[str, Any],
    manifest: dict[str, Any],
    wall_now: datetime,
    monotonic_now: float,
) -> tuple[float, float]:
    expiry = _parse_utc(provenance["authorization_expires_at"], "authorization_expired")
    expiry_remaining = (expiry - wall_now).total_seconds()
    elapsed = _validate_runtime_state(state, manifest, provenance, wall_now)
    maximum_seconds = float(provenance["maximum_runtime_hours"]) * 3600.0
    runtime_remaining = maximum_seconds - elapsed
    hourly = Decimal(str(provenance["hourly_price"])) * Decimal(provenance["vm_count"])
    budget = Decimal(str(provenance["maximum_monetary_budget"]))
    monetary_seconds = float(budget / hourly * Decimal(3600))
    monetary_remaining = monetary_seconds - elapsed
    remaining = min(expiry_remaining, runtime_remaining, monetary_remaining)
    if remaining <= 0:
        if expiry_remaining <= 0:
            code = "authorization_expired"
        elif runtime_remaining <= 0:
            code = "runtime_budget_exceeded"
        else:
            code = "monetary_budget_exceeded"
        raise P7C4B2CError(code)
    return monotonic_now + remaining, elapsed


def _elapsed_budget_violation(provenance: dict[str, Any], elapsed: float) -> str | None:
    runtime_seconds = float(provenance["maximum_runtime_hours"]) * 3600.0
    if elapsed >= runtime_seconds:
        return "runtime_budget_exceeded"
    hourly = Decimal(str(provenance["hourly_price"])) * Decimal(provenance["vm_count"])
    consumed = hourly * Decimal(str(elapsed)) / Decimal(3600)
    if consumed >= Decimal(str(provenance["maximum_monetary_budget"])):
        return "monetary_budget_exceeded"
    return None


def _quarantine(path: Path, run_dir: Path, reason: str) -> None:
    destination = run_dir / "quarantine" / f"{path.name}-{reason}-{uuid4().hex[:8]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))


def _clean_stale_temps(run_dir: Path) -> None:
    temp_root = run_dir / "tmp"
    if not temp_root.exists():
        return
    for path in list(temp_root.iterdir()):
        _quarantine(path, run_dir, "stale_temporary_attempt")


def _execute_task(
    task: dict[str, Any],
    *,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    attempt: int,
    dispatch_started: float | None = None,
    authorization_deadline_monotonic: float | None = None,
    commit_artifact: bool = True,
) -> dict[str, Any]:
    execution_class = manifest.get("execution_class")
    workload = _workload_for_execution_class(execution_class)
    locked_runtime_inputs = None
    if execution_class == "target_preflight":
        provenance = manifest.get("authorization_provenance")
        if (
            not isinstance(provenance, dict)
            or authorization_deadline_monotonic is None
            or task.get("sample_id") not in provenance.get("task_ids", [])
            or str(run_dir.resolve()) != provenance.get("normalized_output_directory")
        ):
            raise P7C4B2CError("task_authorization_context_invalid")
        try:
            locked_runtime_inputs = validate_locked_runtime_inputs(
                provenance.get("locked_runtime_inputs_digest"),
                repo_root,
                dataset_ids=(
                    OUTER_PROJECTION_RUNTIME_DATASETS
                    if provenance.get("execution_stage")
                    == TARGET_PROJECTION_PREFLIGHT_STAGE
                    else ("AC", "GMC")
                ),
            )
        except LockedRuntimeInputError as exc:
            raise P7C4B2CError("locked_runtime_input_mismatch") from exc
    if (
        authorization_deadline_monotonic is not None
        and time.perf_counter() >= authorization_deadline_monotonic
    ):
        raise P7C4B2CError("runtime_budget_exceeded")
    attempt_identity = {
        "sample_id": task["sample_id"],
        "attempt": attempt,
        "run_id": manifest["run_id"],
    }
    attempt_id = sha256_canonical(attempt_identity)
    temporary = run_dir / "tmp" / attempt_id
    if temporary.exists():
        _quarantine(temporary, run_dir, "stale_temporary_attempt")
    temporary.mkdir(parents=True)
    dispatched = (
        dispatch_started if dispatch_started is not None else time.perf_counter()
    )
    started = time.perf_counter()
    started_utc = _utc()
    try:
        detail = (
            canonical_outer_refit(
                task,
                repo_root,
                locked_runtime_inputs=locked_runtime_inputs,
            )
            if execution_class == "target_preflight"
            else workload(task, repo_root)
        )
        completed_workload = time.perf_counter()
        if (
            authorization_deadline_monotonic is not None
            and completed_workload >= authorization_deadline_monotonic
        ):
            raise P7C4B2CError("runtime_budget_exceeded")
        timings = detail["timings"]
        for field in ADDITIVE_COMPONENTS:
            timings.setdefault(
                field, 0.0 if field == "artifact_write_elapsed_seconds" else None
            )
        if any(
            timings[field] is None
            for field in ADDITIVE_COMPONENTS
            if field != "artifact_write_elapsed_seconds"
        ):
            raise P7C4B2CError("required_component_unknown")
        aggregate_before_write = sum(
            float(timings[field]) for field in ADDITIVE_COMPONENTS
        )
        record = {
            **task,
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "attempt": attempt,
            "attempt_id": attempt_id,
            "execution_class": manifest["execution_class"],
            "git_commit": environment["git_commit"],
            "plan_digest": manifest["plan_digest"],
            "preprocessing_identity": detail["preprocessing_identity"],
            "input_identity": detail["input_identity"],
            "input_hash": sha256_canonical(detail["input_identity"]),
            "started_utc": started_utc,
            "completed_utc": _utc(),
            "started_monotonic": started,
            "completed_monotonic": completed_workload,
            "warmup_elapsed_seconds": completed_workload - started
            if task["classification"] == "warmup"
            else 0.0,
            "measured_phase_elapsed_seconds": completed_workload - dispatched
            if task["classification"] == "measured"
            else 0.0,
            "aggregate_measured_fit_runtime_seconds": timings[
                "model_fit_elapsed_seconds"
            ]
            if task["classification"] == "measured"
            else 0.0,
            "worker_process_startup_elapsed_seconds": started - dispatched,
            "task_dispatch_or_queue_elapsed_seconds": started - dispatched,
            **timings,
            "aggregate_outer_refit_runtime_seconds": aggregate_before_write,
            "status": "completed",
            "projection_eligible": False,
            "limitations": detail.get("limitations", []),
            "result": detail["result"],
        }
        write_started = time.perf_counter()
        _atomic_json(temporary / "telemetry.json", record)
        record["artifact_write_elapsed_seconds"] = time.perf_counter() - write_started
        record["aggregate_outer_refit_runtime_seconds"] = sum(
            float(record[field]) for field in ADDITIVE_COMPONENTS
        )
        record["completed_monotonic"] = time.perf_counter()
        record["completed_utc"] = _utc()
        _atomic_json(temporary / "telemetry.json", record)
        _atomic_json(temporary / "result.json", record)
        _atomic_json(
            temporary / "COMPLETED.json",
            {"record_digest": sha256_canonical(record), "attempt_id": attempt_id},
        )
        codes = validate_sample(record, task, manifest)
        if codes:
            raise P7C4B2CError(",".join(codes))
        if commit_artifact:
            destination = run_dir / "samples" / task["sample_id"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise P7C4B2CError("output_collision")
            os.replace(temporary, destination)
        return record
    except Exception as exc:
        failure = {
            **attempt_identity,
            "attempt_id": attempt_id,
            "execution_class": manifest["execution_class"],
            "git_commit": environment["git_commit"],
            "plan_digest": manifest["plan_digest"],
            "failed_utc": _utc(),
            "exception_type": type(exc).__name__,
            "message": str(exc)[:300],
        }
        _atomic_json(run_dir / "failures" / f"{attempt_id}.json", failure)
        if temporary.exists():
            _quarantine(temporary, run_dir, "failed_attempt")
        raise


def _task_process_entry(result_queue: Any, kwargs: dict[str, Any]) -> None:
    """Spawn-safe boundary for one killable projection-preflight task."""
    try:
        isolate_process_group()
        result_queue.put({"status": "completed", "record": _execute_task(**kwargs)})
    except BaseException as exc:  # child must report stable failure to its supervisor
        result_queue.put(
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc)[:300],
            }
        )


def _aggregate_process_tree_rss(root_pid: int) -> int:
    try:
        root = psutil.Process(root_pid)
        processes = {
            process.pid: process for process in [root, *root.children(recursive=True)]
        }
        return sum(process.memory_info().rss for process in processes.values())
    except (psutil.Error, OSError):
        raise P7C4B2CError("memory_sampler_failure") from None


def _terminate_and_reap(processes: list[Any]) -> bool:
    return terminate_and_reap(processes)


def _persist_supervisor_failure(
    run_dir: Path,
    manifest: dict[str, Any],
    task: dict[str, Any],
    reason_code: str,
) -> None:
    identity = {
        "sample_id": task["sample_id"],
        "attempt": 1,
        "run_id": manifest["run_id"],
    }
    attempt_id = sha256_canonical(identity)
    failure = {
        **identity,
        "attempt_id": attempt_id,
        "execution_class": manifest["execution_class"],
        "git_commit": manifest["authorization_provenance"]["git_commit"],
        "plan_digest": manifest["plan_digest"],
        "failed_utc": _utc(),
        "exception_type": "SupervisionFailure",
        "message": reason_code,
        "reason_code": reason_code,
    }
    failure_path = run_dir / "failures" / f"{attempt_id}.json"
    if failure_path.exists():
        codes: list[str] = []
        original = _read_json(failure_path, codes)
        if isinstance(original, dict) and not codes:
            failure = {**original, "supervisor_reason_code": reason_code}
    _atomic_json(failure_path, failure)
    for path in (
        run_dir / "tmp" / attempt_id,
        run_dir / "samples" / task["sample_id"],
    ):
        if path.exists():
            _quarantine(path, run_dir, reason_code)


def _run_supervised_projection_phase(
    phase: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    run_dir: Path,
    repo_root: Path,
    provenance: dict[str, Any],
    runtime_state: dict[str, Any],
    target_task_gate: Callable[[], None],
    authorization_deadline: Callable[[], float],
    monotonic_clock: Callable[[], float],
    rss_sampler: Callable[[int], int] = _aggregate_process_tree_rss,
    context_provider: Callable[[str], Any] = get_context,
    poll_interval_seconds: float = 0.05,
) -> list[dict[str, Any]]:
    policy = provenance.get("resource_policy")
    if not isinstance(policy, dict):
        raise P7C4B2CError("invalid_projection_resource_policy")
    maximum_in_flight = policy.get("maximum_in_flight_tasks")
    if (
        provenance.get("execution_stage") != TARGET_PROJECTION_PREFLIGHT_STAGE
        or policy.get("maximum_runtime_hours") != PROJECTION_MAXIMUM_RUNTIME_HOURS
        or policy.get("maximum_monetary_budget") != PROJECTION_MAXIMUM_MONETARY_BUDGET
        or maximum_in_flight != manifest.get("worker_count")
        or not isinstance(maximum_in_flight, int)
        or isinstance(maximum_in_flight, bool)
        or maximum_in_flight < 1
        or maximum_in_flight > 2
        or policy.get("maximum_failed_task_count") != 0
    ):
        raise P7C4B2CError("invalid_projection_resource_policy")
    timeout_seconds = policy.get("per_task_timeout_seconds")
    rss_limit = policy.get("aggregate_process_tree_rss_limit_bytes")
    if not isinstance(timeout_seconds, int) or not isinstance(rss_limit, int):
        raise P7C4B2CError("invalid_projection_resource_policy")
    ctx = context_provider("spawn")
    pending = list(phase)
    active: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    supervisor_pid = os.getpid()

    def abort(reason: str, failed_item: dict[str, Any] | None = None) -> None:
        cleanup_ok = _terminate_and_reap([item["process"] for item in active])
        queue_results = [close_process_queue(item["queue"]) for item in active]
        queues_ok = all(queue_results)
        cleanup_reason = None if cleanup_ok and queues_ok else "process_cleanup_failure"
        affected = [item["task"] for item in active]
        if failed_item is not None and failed_item["task"] not in affected:
            affected.append(failed_item["task"])
        if not affected and pending:
            affected.append(pending[0])
        for task in affected:
            _persist_supervisor_failure(run_dir, manifest, task, reason)
            if cleanup_reason is not None:
                _persist_supervisor_failure(run_dir, manifest, task, cleanup_reason)
        runtime_state["failed_task_count"] = int(runtime_state["failed_task_count"]) + 1
        now = _parse_utc(runtime_state["last_accounted_at"], "runtime_state_invalid")
        _persist_runtime_state(
            run_dir,
            manifest,
            runtime_state,
            max(float(runtime_state["accumulated_elapsed_seconds"]), 0.0),
            now,
        )
        if cleanup_reason is not None:
            raise P7C4B2CError(f"{reason};{cleanup_reason}")
        raise P7C4B2CError(reason)

    while pending or active:
        if active and monotonic_clock() >= authorization_deadline():
            try:
                target_task_gate()
            except P7C4B2CError as exc:
                abort(str(exc))
        if rss_sampler(supervisor_pid) > rss_limit:
            abort("memory_limit_exceeded")
        while pending and len(active) < maximum_in_flight:
            target_task_gate()
            if rss_sampler(supervisor_pid) > rss_limit:
                abort("memory_limit_exceeded")
            task = pending.pop(0)
            result_queue = ctx.Queue()
            kwargs = {
                "task": task,
                "manifest": manifest,
                "environment": environment,
                "run_dir": run_dir,
                "repo_root": repo_root,
                "attempt": 1,
                "dispatch_started": monotonic_clock(),
                "authorization_deadline_monotonic": authorization_deadline(),
                "commit_artifact": False,
            }
            process = ctx.Process(
                target=_task_process_entry, args=(result_queue, kwargs)
            )
            process.start()
            active.append(
                {
                    "task": task,
                    "process": process,
                    "queue": result_queue,
                    "started": monotonic_clock(),
                    "outcome": None,
                }
            )
        progressed = False
        for item in list(active):
            now_monotonic = monotonic_clock()
            if now_monotonic - item["started"] >= timeout_seconds:
                abort("task_timeout_exceeded", item)
            process = item["process"]
            if item["outcome"] is None:
                try:
                    item["outcome"] = item["queue"].get_nowait()
                except (Empty, AttributeError):
                    pass
            if process.is_alive():
                continue
            process.join(timeout=2.0)
            outcome = item["outcome"]
            if outcome is None:
                try:
                    outcome = item["queue"].get(timeout=1.0)
                except Exception:
                    outcome = {"status": "failed", "message": "child_result_missing"}
            close = getattr(item["queue"], "close", None)
            if close is not None:
                close()
            active.remove(item)
            progressed = True
            try:
                target_task_gate()
            except P7C4B2CError as exc:
                abort(str(exc), item)
            if outcome.get("status") != "completed" or process.exitcode != 0:
                abort("failure_rate_threshold_exceeded", item)
            record = outcome.get("record")
            if not isinstance(record, dict):
                abort("failure_rate_threshold_exceeded", item)
            if rss_sampler(supervisor_pid) > rss_limit:
                abort("memory_limit_exceeded", item)
            attempt_id = record.get("attempt_id")
            temporary = run_dir / "tmp" / str(attempt_id)
            destination = run_dir / "samples" / item["task"]["sample_id"]
            if (
                not _lower_hex(attempt_id, 64)
                or not temporary.is_dir()
                or destination.exists()
            ):
                abort("artifact_validation_failure", item)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(temporary, destination)
            except OSError:
                abort("artifact_validation_failure", item)
            completed.append(record)
        if active and not progressed:
            time.sleep(poll_interval_seconds)
    return completed


def run(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    execution_class: str,
    mode: str,
    repo_root: Path | None = None,
    target_environment: dict[str, Any] | None = None,
    authorization_proposal: dict[str, Any] | None = None,
    effective_authorization: dict[str, Any] | None = None,
    target_authorized: bool = False,
    authorization_plan_digest: str | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    return _run_impl(
        plan,
        output_dir,
        execution_class=execution_class,
        mode=mode,
        repo_root=repo_root,
        target_environment=target_environment,
        authorization_proposal=authorization_proposal,
        effective_authorization=effective_authorization,
        target_authorized=target_authorized,
        authorization_plan_digest=authorization_plan_digest,
        max_samples=max_samples,
        wall_clock=lambda: datetime.now(timezone.utc),
        monotonic_clock=time.perf_counter,
        disk_usage_provider=shutil.disk_usage,
    )


def _run_impl(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    execution_class: str,
    mode: str,
    repo_root: Path | None,
    target_environment: dict[str, Any] | None,
    authorization_proposal: dict[str, Any] | None,
    effective_authorization: dict[str, Any] | None,
    target_authorized: bool,
    authorization_plan_digest: str | None,
    max_samples: int | None,
    wall_clock: Callable[[], datetime],
    monotonic_clock: Callable[[], float],
    disk_usage_provider: Callable[[Path], Any],
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    requested_output = Path(output_dir)
    validate_plan(plan)
    if execution_class not in EXECUTION_CLASSES:
        raise P7C4B2CError("unsupported_execution_class")
    _workload_for_execution_class(execution_class)
    if mode not in MODES:
        raise P7C4B2CError("unsupported_execution_mode")
    provenance = None
    runtime_state = None
    canonical_tasks = None
    if execution_class == "target_preflight":
        canonical_plan = validate_canonical_plan(plan, root)
        if target_authorized or authorization_plan_digest is not None:
            raise P7C4B2CError("legacy_target_authorization_flags_forbidden")
        report = validate_effective_authorization(
            effective_authorization,
            authorization_proposal,
            target_environment or {},
            canonical_plan,
            repo_root=root,
        )
        if not report["valid"]:
            raise P7C4B2CError(",".join(report["reason_codes"]))
        if mode != effective_authorization["execution_mode"]:
            raise P7C4B2CError("execution_mode_mismatch")
        canonical_tasks = _canonical_authorized_tasks(
            canonical_plan, effective_authorization, mode
        )
        plan = canonical_plan
        try:
            normalized_output = normalize_target_output(requested_output, root)
            authorized_output = normalize_target_output(
                effective_authorization["output_directory"], root
            )
        except P7C4B2DError as exc:
            raise P7C4B2CError(str(exc)) from exc
        if normalized_output != authorized_output:
            raise P7C4B2CError("authorized_output_mismatch")
        output_dir = Path(normalized_output)
        _require_live_disk(
            output_dir,
            effective_authorization["minimum_free_disk_bytes"],
            disk_usage_provider,
        )
        now = wall_clock().astimezone(timezone.utc)
        provenance = _authorization_provenance(
            effective_authorization, normalized_output, MODES[mode], canonical_tasks
        )
        runtime_state = {
            "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
            "run_id": output_dir.name,
            "authorization_digest": provenance["authorization_digest"],
            "proposal_digest": provenance["proposal_digest"],
            "target_environment_digest": provenance["target_environment_digest"],
            "plan_digest": provenance["plan_digest"],
            "normalized_output_directory": normalized_output,
            "maximum_runtime_hours": provenance["maximum_runtime_hours"],
            "maximum_monetary_budget": provenance["maximum_monetary_budget"],
            "hourly_price": provenance["hourly_price"],
            "vm_count": provenance["vm_count"],
            "resource_policy": provenance["resource_policy"],
            "runtime_started_at": now.isoformat(),
            "last_accounted_at": now.isoformat(),
            "accumulated_elapsed_seconds": 0.0,
            "consumed_monetary_budget": "0.0",
            "failed_task_count": 0,
            "generation": 0,
        }
        runtime_state["state_digest"] = _runtime_state_digest(runtime_state)
    else:
        output_dir = requested_output.resolve()
    if execution_class == "target_preflight" and max_samples is not None:
        raise P7C4B2CError("target_sample_truncation_forbidden")
    if output_dir.exists():
        raise P7C4B2CError("output_collision")
    output_dir.mkdir(parents=True)
    if (
        execution_class == "target_preflight"
        and str(output_dir.resolve()) != provenance["normalized_output_directory"]
    ):
        raise P7C4B2CError("authorized_output_changed")
    environment = capture_environment(root)
    expected = (
        canonical_tasks
        if execution_class == "target_preflight"
        else _task_subset(plan, mode, max_samples)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": output_dir.name,
        "execution_class": execution_class,
        "mode": mode,
        "worker_count": MODES[mode],
        "plan_digest": plan["plan_digest"],
        "scientific_manifest_digest": plan["scientific_manifest_digest"],
        "expected_tasks": expected,
        "created_utc": _utc(),
        "projection_use": "forbidden"
        if execution_class == "synthetic_validation"
        else "validator_required",
    }
    if provenance is not None:
        manifest["authorization_provenance"] = provenance
        manifest["runtime_origin"] = {
            "runtime_started_at": runtime_state["runtime_started_at"]
        }
        manifest["runtime_checkpoint"] = _runtime_checkpoint(runtime_state)
        _deadline(provenance, runtime_state, manifest, now, monotonic_clock())
    _atomic_json(output_dir / "plan.json", plan)
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(output_dir / "environment.json", environment)
    if runtime_state is not None:
        _atomic_json(output_dir / "authorization_runtime.json", runtime_state)
    return _resume_impl(
        output_dir,
        repo_root=root,
        target_environment=target_environment,
        authorization_proposal=authorization_proposal,
        effective_authorization=effective_authorization,
        target_authorized=False,
        authorization_plan_digest=None,
        wall_clock=wall_clock,
        monotonic_clock=monotonic_clock,
        disk_usage_provider=disk_usage_provider,
        initial_authorization_validated=True,
    )


def resume(
    run_dir: Path,
    *,
    repo_root: Path | None = None,
    target_environment: dict[str, Any] | None = None,
    authorization_proposal: dict[str, Any] | None = None,
    effective_authorization: dict[str, Any] | None = None,
    target_authorized: bool = False,
    authorization_plan_digest: str | None = None,
) -> dict[str, Any]:
    return _resume_impl(
        run_dir,
        repo_root=repo_root,
        target_environment=target_environment,
        authorization_proposal=authorization_proposal,
        effective_authorization=effective_authorization,
        target_authorized=target_authorized,
        authorization_plan_digest=authorization_plan_digest,
        wall_clock=lambda: datetime.now(timezone.utc),
        monotonic_clock=time.perf_counter,
        disk_usage_provider=shutil.disk_usage,
        initial_authorization_validated=False,
    )


def _resume_impl(
    run_dir: Path,
    *,
    repo_root: Path | None,
    target_environment: dict[str, Any] | None,
    authorization_proposal: dict[str, Any] | None,
    effective_authorization: dict[str, Any] | None,
    target_authorized: bool,
    authorization_plan_digest: str | None,
    wall_clock: Callable[[], datetime],
    monotonic_clock: Callable[[], float],
    disk_usage_provider: Callable[[Path], Any],
    initial_authorization_validated: bool,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    run_dir = run_dir.resolve()
    codes: list[str] = []
    plan = _read_json(run_dir / "plan.json", codes)
    manifest = _read_json(run_dir / "manifest.json", codes)
    environment = _read_json(run_dir / "environment.json", codes)
    if codes or not plan or not manifest or not environment:
        raise P7C4B2CError("incompatible_resume")
    execution_class = manifest.get("execution_class")
    _workload_for_execution_class(execution_class)
    authorization_deadline_monotonic = None
    provenance = None
    runtime_state = None
    canonical_tasks = None
    session_started_monotonic = monotonic_clock()
    session_started_elapsed = 0.0
    if execution_class == "target_preflight":
        # Rebuild from locked inputs and compare the persisted representation before
        # any authorization-dependent operation or filesystem cleanup. Dispatch uses
        # only task objects from the rebuilt plan returned here.
        plan = validate_canonical_plan(plan, root)
        if target_authorized or authorization_plan_digest is not None:
            raise P7C4B2CError("legacy_target_authorization_flags_forbidden")
        if not initial_authorization_validated:
            report = validate_effective_authorization(
                effective_authorization,
                authorization_proposal,
                target_environment or {},
                plan,
                repo_root=root,
                allow_existing_output=True,
            )
            if not report["valid"]:
                raise P7C4B2CError(",".join(report["reason_codes"]))
        if not isinstance(effective_authorization, dict):
            raise P7C4B2CError("authorization_missing")
        try:
            authorized_output = normalize_target_output(
                effective_authorization["output_directory"], root
            )
        except (KeyError, P7C4B2DError) as exc:
            raise P7C4B2CError("authorized_output_mismatch") from exc
        if str(run_dir) != authorized_output:
            raise P7C4B2CError("authorized_output_mismatch")
        if manifest.get("mode") != effective_authorization.get("execution_mode"):
            raise P7C4B2CError("execution_mode_mismatch")
        canonical_tasks = _canonical_authorized_tasks(
            plan, effective_authorization, manifest.get("mode")
        )
        provenance = _validate_resume_provenance(
            manifest, effective_authorization, authorized_output, canonical_tasks
        )
        runtime_codes: list[str] = []
        runtime_state = _read_json(
            run_dir / "authorization_runtime.json", runtime_codes
        )
        if runtime_codes or not runtime_state:
            raise P7C4B2CError("runtime_state_invalid")
        if provenance.get("execution_stage") == TARGET_PROJECTION_PREFLIGHT_STAGE and (
            int(runtime_state.get("failed_task_count", -1)) > 0
            or any((run_dir / "failures").glob("*.json"))
        ):
            raise P7C4B2CError("failure_rate_threshold_exceeded")
        now = wall_clock().astimezone(timezone.utc)
        authorization_deadline_monotonic, session_started_elapsed = _deadline(
            provenance, runtime_state, manifest, now, session_started_monotonic
        )
        _require_live_disk(
            run_dir,
            provenance["minimum_free_disk_bytes"],
            disk_usage_provider,
        )
    if (
        execution_class == "target_preflight"
        and not initial_authorization_validated
        and (run_dir / "COMPLETED.json").exists()
    ):
        final_report = validate_artifacts(run_dir, repo_root=root)
        return {
            "run_id": manifest["run_id"],
            "executed": 0,
            "skipped": final_report.get("completed", 0),
            "validation": final_report,
        }

    def target_task_gate() -> None:
        nonlocal authorization_deadline_monotonic
        if provenance is None or runtime_state is None:
            return
        now = wall_clock().astimezone(timezone.utc)
        session_elapsed = session_started_elapsed + max(
            0.0, monotonic_clock() - session_started_monotonic
        )
        conservative_elapsed = max(
            _validate_runtime_state(runtime_state, manifest, provenance, now),
            session_elapsed,
        )
        violation = _elapsed_budget_violation(provenance, conservative_elapsed)
        if violation is not None:
            raise P7C4B2CError(violation)
        _persist_runtime_state(
            run_dir, manifest, runtime_state, conservative_elapsed, now
        )
        authorization_deadline_monotonic, _ = _deadline(
            provenance, runtime_state, manifest, now, monotonic_clock()
        )
        _require_live_disk(
            run_dir,
            provenance["minimum_free_disk_bytes"],
            disk_usage_provider,
        )

    records = []
    pending: list[dict[str, Any]] = []
    skipped = executed = 0
    tasks_for_dispatch = (
        canonical_tasks if canonical_tasks is not None else manifest["expected_tasks"]
    )
    for task in tasks_for_dispatch:
        directory = run_dir / "samples" / task["sample_id"]
        if directory.exists():
            local_codes: list[str] = []
            record = _read_json(directory / "result.json", local_codes)
            marker = _read_json(directory / "COMPLETED.json", local_codes)
            if (
                local_codes
                or not record
                or not marker
                or marker.get("record_digest") != _safe_digest(record, local_codes)
                or validate_sample(record, task, manifest)
            ):
                _quarantine(directory, run_dir, "corrupt_completed_sample")
            else:
                records.append(record)
                skipped += 1
                continue
        pending.append(task)
    if not initial_authorization_validated and not pending:
        final_report = validate_artifacts(run_dir, repo_root=root)
        return {
            "run_id": manifest["run_id"],
            "executed": 0,
            "skipped": skipped,
            "validation": final_report,
        }
    _clean_stale_temps(run_dir)
    # Warmups finish before the measured pool starts. Authorization and live disk
    # are rechecked before every submission; already-running work is not hard-killed.
    for classification in ("warmup", "measured"):
        phase = [task for task in pending if task["classification"] == classification]
        if not phase:
            continue
        if (
            provenance is not None
            and provenance.get("execution_stage") == TARGET_PROJECTION_PREFLIGHT_STAGE
            and runtime_state is not None
        ):
            phase_records = _run_supervised_projection_phase(
                phase,
                manifest=manifest,
                environment=environment,
                run_dir=run_dir,
                repo_root=root,
                provenance=provenance,
                runtime_state=runtime_state,
                target_task_gate=target_task_gate,
                authorization_deadline=lambda: authorization_deadline_monotonic,
                monotonic_clock=monotonic_clock,
            )
            records.extend(phase_records)
            executed += len(phase_records)
            target_task_gate()
            continue
        with ProcessPoolExecutor(
            max_workers=manifest["worker_count"], mp_context=get_context("spawn")
        ) as executor:
            futures = {}
            for task in phase:
                target_task_gate()
                future = executor.submit(
                    _execute_task,
                    task,
                    manifest=manifest,
                    environment=environment,
                    run_dir=run_dir,
                    repo_root=root,
                    attempt=1,
                    dispatch_started=monotonic_clock(),
                    authorization_deadline_monotonic=authorization_deadline_monotonic,
                )
                futures[future] = task
            for future in as_completed(futures):
                records.append(future.result())
                executed += 1
        target_task_gate()
    records.sort(key=lambda item: item["sample_id"])
    if provenance is not None and runtime_state is not None:
        target_task_gate()
    coverage, stratum_summary = summarize(
        records, plan, expected_tasks=manifest["expected_tasks"]
    )
    _atomic_json(run_dir / "coverage.json", coverage)
    _atomic_json(run_dir / "stratum_summary.json", stratum_summary)
    preliminary = _validate_artifacts(
        run_dir,
        allow_missing_derived=True,
        allow_missing_marker=True,
        repo_root=root,
    )
    projection = project_validated(
        records,
        plan,
        artifact_validation=preliminary,
        execution_class=manifest["execution_class"],
    )
    _atomic_json(run_dir / "projection.json", projection)
    _atomic_json(
        run_dir / "eligibility.json",
        {
            "schema_version": SCHEMA_VERSION,
            "execution_plan_eligible": projection["execution_plan_eligible"],
            "reason_codes": projection["reason_codes"],
        },
    )
    report = _validate_artifacts(run_dir, allow_missing_marker=True, repo_root=root)
    _atomic_json(run_dir / "validation.json", report)
    if report["valid"] and report["completed"] == report["expected"]:
        _atomic_json(
            run_dir / "COMPLETED.json",
            {
                "validation_digest": sha256_canonical(report),
                "run_id": manifest["run_id"],
            },
        )
        report = validate_artifacts(run_dir, repo_root=root)
    return {
        "run_id": manifest["run_id"],
        "executed": executed,
        "skipped": skipped,
        "validation": report,
    }


def _validate_artifacts(
    run_dir: Path,
    *,
    allow_missing_derived: bool = False,
    allow_missing_marker: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    codes: list[str] = []
    plan = _read_json(run_dir / "plan.json", codes)
    manifest = _read_json(run_dir / "manifest.json", codes)
    environment = _read_json(run_dir / "environment.json", codes)
    if not plan or not manifest or not environment:
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "reason_codes": sorted(set(codes)),
        }
    try:
        validate_plan(plan)
    except P7C4B2CError as exc:
        codes.extend(str(exc).split(","))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or environment.get("schema_version") != SCHEMA_VERSION
    ):
        codes.append("unsupported_schema_version")
    if manifest.get("plan_digest") != plan.get("plan_digest"):
        codes.append("plan_hash_mismatch")
    if not environment.get("git_commit"):
        codes.append("git_provenance_missing")
    if environment.get("environment_digest") != canonical_digest(
        environment, "environment_digest"
    ):
        codes.append("environment_hash_mismatch")
    expected = manifest.get("expected_tasks", [])
    if not isinstance(expected, list):
        codes.append("canonical_task_manifest_mismatch")
        expected = []
    codes.extend(
        _validate_persisted_target_contract(
            run_dir, manifest, plan, expected, repo_root=repo_root
        )
    )
    expected_ids = [task.get("sample_id") for task in expected]
    if len(expected_ids) != len(set(expected_ids)):
        codes.append("duplicate_sample_identity")
    actual_dirs = (
        sorted((run_dir / "samples").glob("*"))
        if (run_dir / "samples").exists()
        else []
    )
    if any(path.name not in set(expected_ids) for path in actual_dirs):
        codes.append("unexpected_sample")
    if set(expected_ids) - {path.name for path in actual_dirs}:
        codes.append("missing_planned_sample")
    records = []
    task_by_id = {task["sample_id"]: task for task in expected}
    for directory in actual_dirs:
        record = _read_json(directory / "result.json", codes)
        telemetry = _read_json(directory / "telemetry.json", codes)
        marker = _read_json(directory / "COMPLETED.json", codes)
        if not record:
            continue
        records.append(record)
        if telemetry and (
            telemetry.get("attempt_id") != record.get("attempt_id")
            or telemetry != record
        ):
            codes.append("telemetry_identity_mismatch")
        if not marker or marker.get("record_digest") != _safe_digest(record, codes):
            codes.append("complete_marker_integrity_failure")
        codes.extend(
            validate_sample(record, task_by_id.get(directory.name, {}), manifest)
        )
        provenance = manifest.get("authorization_provenance") or {}
        if provenance.get("execution_stage") == TARGET_PROJECTION_PREFLIGHT_STAGE:
            task = task_by_id.get(directory.name, {})
            dataset_id = task.get("dataset_id")
            expected_hash = (provenance.get("dataset_hashes") or {}).get(dataset_id)
            if (record.get("input_identity") or {}).get("dataset_id") != dataset_id or (
                record.get("input_identity") or {}
            ).get("dataset_checksum") != expected_hash:
                codes.append("outer_dataset_hash_mismatch")
    identities = [record.get("sample_id") for record in records]
    if len(identities) != len(set(identities)):
        codes.append("duplicate_sample_identity")
    codes.extend(_validate_target_source_identity(manifest, environment, records))
    coverage, summary = summarize(records, plan, expected_tasks=expected)
    scientific_coverage = _scientific_coverage_result(coverage)
    is_target = manifest.get("execution_class") == "target_preflight"
    execution_stage = (manifest.get("authorization_provenance") or {}).get(
        "execution_stage"
    )
    is_target_canary = is_target and execution_stage == TARGET_CANARY_STAGE
    is_projection_preflight = is_target and execution_stage in (
        TARGET_EXECUTION_STAGES - {TARGET_CANARY_STAGE}
    )
    if not is_target:
        codes.extend(scientific_coverage["reason_codes"])
    if not allow_missing_derived:
        stored_coverage = _read_json(run_dir / "coverage.json", codes)
        stored_summary = _read_json(run_dir / "stratum_summary.json", codes)
        projection = _read_json(run_dir / "projection.json", codes)
        gate = _read_json(run_dir / "eligibility.json", codes)
        if stored_coverage != coverage or stored_summary != summary:
            codes.append("derived_artifact_mismatch")
        evidence_digest = _safe_digest(records, codes)
        if projection and projection.get("source_evidence_digest") != evidence_digest:
            codes.append("invalid_projection_source")
        if (
            gate
            and gate.get("execution_plan_eligible") is True
            and (
                any(
                    record.get(field) is None
                    for record in records
                    for field in ADDITIVE_COMPONENTS
                )
                or manifest.get("execution_class") != "target_preflight"
            )
        ):
            codes.append("invalid_true_eligibility")
    if (run_dir / "failures").exists() and any((run_dir / "failures").glob("*.json")):
        codes.append("failed_attempt_present")
    if (
        is_target_canary
        and (run_dir / "quarantine").exists()
        and any((run_dir / "quarantine").iterdir())
    ):
        codes.append("quarantined_attempt_present")
    completion = (
        _read_json(run_dir / "COMPLETED.json", codes)
        if (run_dir / "COMPLETED.json").exists()
        else None
    )
    if completion:
        validation = _read_json(run_dir / "validation.json", codes)
        if (
            not validation
            or completion.get("validation_digest") != sha256_canonical(validation)
            or completion.get("run_id") != manifest.get("run_id")
            or len(records) != len(expected)
        ):
            codes.append("run_complete_marker_integrity_failure")
    elif not allow_missing_marker:
        codes.append("completed_run_missing_marker")
    evidence_digest = _safe_digest(records, codes)
    target_canary_acceptance = {
        "applicable": is_target_canary,
        "accepted": is_target_canary and not codes,
        "reason_codes": sorted(set(codes)) if is_target_canary else [],
        "scientific_projection_eligible": False,
        "canonical_scientific_execution_authorized": False,
    }
    target_projection_preflight_acceptance = {
        "applicable": is_projection_preflight,
        "accepted": is_projection_preflight and not codes,
        "reason_codes": sorted(set(codes)) if is_projection_preflight else [],
        "scientific_projection_eligible": False,
        "canonical_scientific_execution_authorized": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not codes,
        "reason_codes": sorted(set(codes)),
        "expected": len(expected),
        "completed": len(records),
        "evidence_digest": evidence_digest,
        "execution_class": manifest.get("execution_class"),
        "target_canary_acceptance": target_canary_acceptance,
        "target_projection_preflight_acceptance": target_projection_preflight_acceptance,
        "scientific_coverage": scientific_coverage,
    }


def validate_artifacts(
    run_dir: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate a persisted run; completed runs always require their run marker."""
    return _validate_artifacts(run_dir, repo_root=repo_root)
