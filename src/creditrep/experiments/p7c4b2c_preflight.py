"""Reusable execution and artifact lifecycle for P7C.4B.2c."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.registry import find_repo_root
from creditrep.protocols.p7c4b2c import (
    ADDITIVE_COMPONENTS,
    EXECUTION_CLASSES,
    MODES,
    P7C4B2CError,
    SCHEMA_VERSION,
    canonical_digest,
    project_validated,
    summarize,
    validate_plan,
    validate_sample,
)

EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_INCOMPLETE = 3
EXIT_AUTHORIZATION = 4
Workload = Callable[[dict[str, Any], Path], dict[str, Any]]


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
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        codes.append("missing_required_artifact")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        codes.append("malformed_artifact")
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


def canonical_outer_refit(task: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Run the canonical preprocessing/refit primitive for one authorized target unit."""
    from creditrep.datasets import load_dataset
    from creditrep.experiments.model_validation import _fit_for_partition
    from creditrep.experiments.p7c3_feasibility import adapt_mlp_feasibility_candidate
    from creditrep.preprocessing import load_protocol_a_config
    from creditrep.splitting import create_nested_cv_definition

    dataset = load_dataset(task["dataset_id"], repo_root=repo_root)
    with (repo_root / "data" / "checksums-sha256.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        checksums = {row["Path"]: row["Hash"] for row in csv.DictReader(handle)}
    source_hash = checksums[dataset.metadata["source_file"]]
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
        protocol_config=load_protocol_a_config(repo_root=repo_root),
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
    workload: Workload,
    attempt: int,
    dispatch_started: float | None = None,
) -> dict[str, Any]:
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
        detail = workload(task, repo_root)
        completed_workload = time.perf_counter()
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


def run(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    execution_class: str,
    mode: str,
    repo_root: Path | None = None,
    target_authorized: bool = False,
    authorization_plan_digest: str | None = None,
    max_samples: int | None = None,
    workload: Workload | None = None,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    output_dir = output_dir.resolve()
    validate_plan(plan)
    if execution_class not in EXECUTION_CLASSES:
        raise P7C4B2CError("unsupported_execution_class")
    if mode not in MODES:
        raise P7C4B2CError("unsupported_execution_mode")
    if execution_class == "target_preflight" and (
        not target_authorized or authorization_plan_digest != plan["plan_digest"]
    ):
        raise P7C4B2CError("target_authorization_missing_or_mismatch")
    if execution_class == "target_preflight" and max_samples is not None:
        raise P7C4B2CError("target_sample_truncation_forbidden")
    if output_dir.exists():
        raise P7C4B2CError("output_collision")
    output_dir.mkdir(parents=True)
    environment = capture_environment(root)
    expected = _task_subset(plan, mode, max_samples)
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
    _atomic_json(output_dir / "plan.json", plan)
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(output_dir / "environment.json", environment)
    return resume(
        output_dir,
        repo_root=root,
        target_authorized=target_authorized,
        authorization_plan_digest=authorization_plan_digest,
        workload=workload,
    )


def resume(
    run_dir: Path,
    *,
    repo_root: Path | None = None,
    target_authorized: bool = False,
    authorization_plan_digest: str | None = None,
    workload: Workload | None = None,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    run_dir = run_dir.resolve()
    codes: list[str] = []
    plan = _read_json(run_dir / "plan.json", codes)
    manifest = _read_json(run_dir / "manifest.json", codes)
    environment = _read_json(run_dir / "environment.json", codes)
    if codes or not plan or not manifest or not environment:
        raise P7C4B2CError("incompatible_resume")
    if manifest["execution_class"] == "target_preflight" and (
        not target_authorized or authorization_plan_digest != plan["plan_digest"]
    ):
        raise P7C4B2CError("target_authorization_missing_or_mismatch")
    _clean_stale_temps(run_dir)
    adapter = workload or (
        synthetic_outer_refit
        if manifest["execution_class"] == "synthetic_validation"
        else canonical_outer_refit
    )
    records = []
    pending: list[dict[str, Any]] = []
    skipped = executed = 0
    for task in manifest["expected_tasks"]:
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
    if workload is not None:
        for task in pending:
            records.append(
                _execute_task(
                    task,
                    manifest=manifest,
                    environment=environment,
                    run_dir=run_dir,
                    repo_root=root,
                    workload=adapter,
                    attempt=1,
                )
            )
            executed += 1
    else:
        # Warmups finish before the measured pool starts. Each submitted unit runs
        # in a spawned process, so startup/queue time has the same boundary in
        # synthetic validation and an authorized target preflight.
        for classification in ("warmup", "measured"):
            phase = [
                task for task in pending if task["classification"] == classification
            ]
            with ProcessPoolExecutor(
                max_workers=manifest["worker_count"],
                mp_context=get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(
                        _execute_task,
                        task,
                        manifest=manifest,
                        environment=environment,
                        run_dir=run_dir,
                        repo_root=root,
                        workload=adapter,
                        attempt=1,
                        dispatch_started=time.perf_counter(),
                    ): task
                    for task in phase
                }
                for future in as_completed(futures):
                    records.append(future.result())
                    executed += 1
    records.sort(key=lambda item: item["sample_id"])
    coverage, stratum_summary = summarize(
        records, plan, expected_tasks=manifest["expected_tasks"]
    )
    _atomic_json(run_dir / "coverage.json", coverage)
    _atomic_json(run_dir / "stratum_summary.json", stratum_summary)
    preliminary = validate_artifacts(run_dir, allow_missing_derived=True)
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
    report = validate_artifacts(run_dir, allow_missing_marker=True)
    _atomic_json(run_dir / "validation.json", report)
    if report["valid"] and report["completed"] == report["expected"]:
        _atomic_json(
            run_dir / "COMPLETED.json",
            {
                "validation_digest": sha256_canonical(report),
                "run_id": manifest["run_id"],
            },
        )
    return {
        "run_id": manifest["run_id"],
        "executed": executed,
        "skipped": skipped,
        "validation": report,
    }


def validate_artifacts(
    run_dir: Path,
    *,
    allow_missing_derived: bool = False,
    allow_missing_marker: bool = False,
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
        if telemetry and telemetry.get("attempt_id") != record.get("attempt_id"):
            codes.append("telemetry_identity_mismatch")
        if not marker or marker.get("record_digest") != _safe_digest(record, codes):
            codes.append("complete_marker_integrity_failure")
        codes.extend(
            validate_sample(record, task_by_id.get(directory.name, {}), manifest)
        )
    identities = [record.get("sample_id") for record in records]
    if len(identities) != len(set(identities)):
        codes.append("duplicate_sample_identity")
    coverage, summary = summarize(records, plan, expected_tasks=expected)
    if coverage["incomplete_strata"]:
        codes.extend(["incomplete_required_stratum", "insufficient_repetitions"])
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
            or len(records) != len(expected)
        ):
            codes.append("run_complete_marker_integrity_failure")
    elif not allow_missing_marker and not codes and len(records) == len(expected):
        codes.append("completed_run_missing_marker")
    evidence_digest = _safe_digest(records, codes)
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not codes,
        "reason_codes": sorted(set(codes)),
        "expected": len(expected),
        "completed": len(records),
        "evidence_digest": evidence_digest,
        "execution_class": manifest.get("execution_class"),
    }
