"""Engineering-only RF/XGBoost pilot harness for P7C.2; no ranking or refit."""

from __future__ import annotations

import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import psutil
import sklearn
import xgboost

from creditrep.checksums import get_dataset_checksum
from creditrep.config.loader import sha256_canonical
from creditrep.datasets.loader import load_dataset
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.nested_cv import _fit_preprocessing
from creditrep.experiments.p7b_cart import ProcessRssSampler, capture_git_provenance
from creditrep.models.factory import create_model
from creditrep.preprocessing import load_protocol_a_config
from creditrep.protocols.p7c2 import load_pilot_plan, plan_digest
from creditrep.splitting import create_nested_cv_definition


class P7C2HarnessError(ValueError):
    """The immutable P7C.2 execution or artifact contract was violated."""


EXPECTED_OUTER = "repeat_00_fold_00"
REQUIRED_TELEMETRY = {
    "process_id",
    "configured_thread_count",
    "effective_thread_count",
    "process_rss_start_bytes",
    "process_rss_peak_bytes",
    "process_rss_delta_peak_bytes",
    "process_cpu_seconds",
    "started_at",
    "completed_at",
    "duration_seconds",
    "library_versions",
}
FORBIDDEN_PAYLOAD_KEYS = {
    "predictions",
    "model_weights",
    "raw_rows",
    "transformed_matrix",
    "feature_values",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def stable_fit_id(identity: dict[str, Any]) -> str:
    required = {
        "plan_digest",
        "model_id",
        "dataset_id",
        "outer_repeat_index",
        "outer_fold_index",
        "candidate_id",
        "inner_fold_index",
        "seed",
    }
    if set(identity) != required:
        raise P7C2HarnessError(
            "fit identity fields do not match the immutable contract"
        )
    return sha256_canonical(identity)


def _portable_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if FORBIDDEN_PAYLOAD_KEYS & set(payload):
        raise P7C2HarnessError("forbidden training payload in artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_error(exc: Exception) -> dict[str, str]:
    message = str(exc).replace("\\", "/")
    if ":/" in message or ":\\" in str(exc):
        message = "sanitized local-path error"
    return {"type": type(exc).__name__, "message": message[:300]}


def _effective_parameters(
    model_id: str, parameters: dict[str, Any], feature_count: int, seed: int
) -> dict[str, Any]:
    if model_id == "random_forest":
        multiplier = parameters["max_features_multiplier_of_sqrt_m"]
        max_features = max(
            1, min(feature_count, math.ceil(multiplier * math.sqrt(feature_count)))
        )
        return {
            "n_estimators": parameters["n_estimators"],
            "max_features": max_features,
            "n_jobs": 1,
            "random_state": seed,
        }
    return {
        **parameters,
        "n_jobs": 1,
        "random_state": seed,
        "tree_method": "hist",
        "eval_metric": "logloss",
    }


def build_execution_plan(
    spec_path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    spec = load_pilot_plan(spec_path, repo_root=root)
    digest = plan_digest(spec)
    fits: list[dict[str, Any]] = []
    for dataset_id in spec["datasets"]:
        dataset = load_dataset(dataset_id, repo_root=root)
        source = _portable_relative(dataset.source_path, root)
        checksum = get_dataset_checksum(
            dataset_id, source, repo_root=root
        ).actual_sha256
        nested = create_nested_cv_definition(
            dataset,
            dataset_checksum=checksum,
            outer_n_repeats=1,
            outer_n_splits=2,
            inner_n_splits=5,
            random_seed=spec["seed"],
        )
        outer = next(
            item for item in nested.outer_folds if item.outer_fold_id == EXPECTED_OUTER
        )
        for model in spec["models"]:
            for candidate in model["candidates"]:
                for inner_index, inner in enumerate(outer.inner_folds):
                    identity = {
                        "plan_digest": digest,
                        "model_id": model["model_id"],
                        "dataset_id": dataset_id,
                        "outer_repeat_index": 0,
                        "outer_fold_index": 0,
                        "candidate_id": candidate["id"],
                        "inner_fold_index": inner_index,
                        "seed": inner.seed,
                    }
                    fit_id = stable_fit_id(identity)
                    fits.append(
                        {
                            **identity,
                            "fit_id": fit_id,
                            "parameters": candidate["parameters"],
                            "dataset_checksum": checksum,
                            "inner_split_hash": inner.split_hash,
                            "artifact_path": f"fits/{fit_id}/result.json",
                        }
                    )
    if len(fits) != spec["expected_fits"]["total"] or len(
        {x["fit_id"] for x in fits}
    ) != len(fits):
        raise P7C2HarnessError("execution plan count/identity mismatch")
    return {
        "schema_version": 1,
        "checkpoint_id": "P7C.2.1",
        "purpose": "engineering_feasibility_only",
        "plan_digest": digest,
        "threading": spec["threading"],
        "artifact_root": spec["artifact_root"],
        "retry_policy": spec["retry_policy"],
        "fits": fits,
    }


def validate_execution_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if (
        plan.get("schema_version") != 1
        or plan.get("purpose") != "engineering_feasibility_only"
    ):
        raise P7C2HarnessError("invalid execution plan schema/purpose")
    if plan.get("threading") != {
        "fits_parallelism": 1,
        "estimator_threads": 1,
        "allow_n_jobs_minus_one": False,
        "gpu_enabled": False,
    }:
        raise P7C2HarnessError("execution threading policy mismatch")
    fits = plan.get("fits")
    if not isinstance(fits, list) or len(fits) != 60:
        raise P7C2HarnessError("execution plan must contain exactly 60 fits")
    ids: list[str] = []
    for fit in fits:
        identity = {
            key: fit[key]
            for key in (
                "plan_digest",
                "model_id",
                "dataset_id",
                "outer_repeat_index",
                "outer_fold_index",
                "candidate_id",
                "inner_fold_index",
                "seed",
            )
        }
        expected = stable_fit_id(identity)
        if fit.get("fit_id") != expected or fit.get("plan_digest") != plan.get(
            "plan_digest"
        ):
            raise P7C2HarnessError("fit identity or plan digest mismatch")
        ids.append(expected)
    if len(set(ids)) != 60:
        raise P7C2HarnessError("duplicate fit identity")
    return {"valid": True, "expected_fits": 60, "unique_fit_ids": 60}


def _allowed_output(output_dir: Path, plan: dict[str, Any], root: Path) -> None:
    allowed = (root / plan["artifact_root"]).resolve()
    target = output_dir.resolve()
    if target == allowed or allowed not in target.parents:
        raise P7C2HarnessError(
            f"output directory must be a child of {plan['artifact_root']}"
        )


def _execute_fit(
    fit: dict[str, Any], dataset: Any, inner: Any, protocol: Any
) -> dict[str, Any]:
    sampler = ProcessRssSampler(interval_seconds=0.05)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    started = _now()
    sampler.start()
    try:
        _, train_matrix, validation_matrix = _fit_preprocessing(
            dataset,
            train_indices=inner.train_indices,
            transform_indices=inner.validation_indices,
            protocol_config=protocol,
        )
        parameters = _effective_parameters(
            fit["model_id"], fit["parameters"], train_matrix.shape[1], fit["seed"]
        )
        estimator = create_model(fit["model_id"], parameters, random_seed=fit["seed"])
        estimator.fit(train_matrix, dataset.target.iloc[list(inner.train_indices)])
        if list(estimator.classes_) != [0, 1]:
            raise P7C2HarnessError("estimator class order must be [0, 1]")
        probabilities = np.asarray(estimator.predict_proba(validation_matrix))
        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != 2
            or not np.isfinite(probabilities[:, 1]).all()
        ):
            raise P7C2HarnessError(
                "predict_proba must expose finite P(class=1) in column 1"
            )
        rss = sampler.stop()
        return {
            "schema_version": 1,
            "status": "completed",
            "outcome": "completed",
            "fit_id": fit["fit_id"],
            "plan_digest": fit["plan_digest"],
            "model_id": fit["model_id"],
            "dataset_id": fit["dataset_id"],
            "candidate_id": fit["candidate_id"],
            "outer_repeat_index": fit["outer_repeat_index"],
            "outer_fold_index": fit["outer_fold_index"],
            "inner_fold_index": fit["inner_fold_index"],
            "seed": fit["seed"],
            "configured_thread_count": 1,
            "effective_thread_count": int(parameters["n_jobs"]),
            "tree_method": parameters.get("tree_method"),
            "started_at": started,
            "completed_at": _now(),
            "duration_seconds": time.perf_counter() - wall_start,
            "process_cpu_seconds": time.process_time() - cpu_start,
            "library_versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
            "error": None,
            **rss,
        }
    except BaseException:
        sampler.stop()
        raise


def _identity_matches(payload: dict[str, Any], fit: dict[str, Any]) -> bool:
    return all(
        payload.get(key) == fit.get(key)
        for key in (
            "fit_id",
            "plan_digest",
            "model_id",
            "dataset_id",
            "candidate_id",
            "outer_repeat_index",
            "outer_fold_index",
            "inner_fold_index",
            "seed",
        )
    )


def _provenance_valid(payload: dict[str, Any]) -> bool:
    provenance = payload.get("git_provenance")
    if not isinstance(provenance, dict):
        return False
    git_head = provenance.get("git_head")
    details = provenance.get("working_tree_details")
    return (
        isinstance(git_head, str)
        and len(git_head) == 40
        and all(character in "0123456789abcdef" for character in git_head.lower())
        and provenance.get("working_tree") in {"clean", "dirty"}
        and isinstance(details, dict)
        and isinstance(details.get("is_dirty"), bool)
        and isinstance(details.get("porcelain_v1"), list)
    )


def _valid_completed(path: Path, fit: dict[str, Any]) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "completed"
        and value.get("outcome") == "completed"
        and _identity_matches(value, fit)
        and not (REQUIRED_TELEMETRY - set(value))
        and not (FORBIDDEN_PAYLOAD_KEYS & set(value))
        and _provenance_valid(value)
    )


def validate_artifacts(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_execution_plan(plan)
    expected = {item["fit_id"]: item for item in plan["fits"]}
    completed = failed = retryable_failed = 0
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    if not output_dir.exists():
        return {
            "valid": False,
            "resumable": False,
            "expected": 60,
            "completed": 0,
            "failed": 0,
            "missing": 60,
            "unexpected": 0,
            "errors": [{"code": "missing_output"}],
        }
    try:
        saved_plan = json.loads(
            (output_dir / "execution_plan.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        saved_plan = None
        errors.append({"code": "missing_or_corrupt_plan"})
    if saved_plan != plan:
        errors.append({"code": "plan_mismatch"})
    try:
        environment = json.loads(
            (output_dir / "environment.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        environment = None
        errors.append({"code": "missing_or_corrupt_environment"})
    if environment is not None and (
        environment.get("plan_digest") != plan["plan_digest"]
        or environment.get("threading") != plan["threading"]
        or not isinstance(environment.get("git_head"), str)
        or len(environment.get("git_head", "")) != 40
    ):
        errors.append({"code": "environment_mismatch"})
    for temporary in output_dir.rglob("*.tmp"):
        errors.append(
            {
                "code": "incomplete_temporary",
                "path": temporary.relative_to(output_dir).as_posix(),
            }
        )
    for path in output_dir.glob("fits/*/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(
                {
                    "code": "corrupt_json",
                    "path": path.relative_to(output_dir).as_posix(),
                }
            )
            continue
        fit_id = payload.get("fit_id")
        if fit_id not in expected:
            errors.append(
                {
                    "code": "unexpected_fit",
                    "path": path.relative_to(output_dir).as_posix(),
                }
            )
            continue
        if fit_id in seen:
            errors.append(
                {
                    "code": "duplicate_fit",
                    "path": path.relative_to(output_dir).as_posix(),
                }
            )
            continue
        seen.add(fit_id)
        fit = expected[fit_id]
        if not _identity_matches(payload, fit) or FORBIDDEN_PAYLOAD_KEYS & set(payload):
            errors.append(
                {
                    "code": "invalid_payload",
                    "path": path.relative_to(output_dir).as_posix(),
                }
            )
            continue
        if payload.get("status") == "completed":
            missing_fields = REQUIRED_TELEMETRY - set(payload)
            if (
                missing_fields
                or payload.get("outcome") != "completed"
                or not _provenance_valid(payload)
            ):
                errors.append(
                    {
                        "code": "invalid_telemetry",
                        "path": path.relative_to(output_dir).as_posix(),
                    }
                )
            else:
                completed += 1
        elif payload.get("status") == "failed":
            if (
                REQUIRED_TELEMETRY - set(payload)
                or not isinstance(payload.get("attempt_count"), int)
                or not _provenance_valid(payload)
            ):
                errors.append(
                    {
                        "code": "invalid_failure_telemetry",
                        "path": path.relative_to(output_dir).as_posix(),
                    }
                )
            elif (
                payload["attempt_count"]
                > plan["retry_policy"]["max_retry_attempts"] + 1
            ):
                errors.append(
                    {
                        "code": "retry_budget_exceeded",
                        "path": path.relative_to(output_dir).as_posix(),
                    }
                )
            else:
                failed += 1
                error_type = payload.get("error", {}).get("type")
                if payload["attempt_count"] <= plan["retry_policy"][
                    "max_retry_attempts"
                ] and (
                    not plan["retry_policy"].get("transient_errors_only")
                    or error_type in {"OSError", "TimeoutError"}
                ):
                    retryable_failed += 1
        else:
            errors.append(
                {
                    "code": "invalid_status",
                    "path": path.relative_to(output_dir).as_posix(),
                }
            )
    missing = len(expected) - len(seen)
    blocking = any(
        error["code"]
        in {
            "plan_mismatch",
            "missing_or_corrupt_plan",
            "missing_or_corrupt_environment",
            "environment_mismatch",
            "unexpected_fit",
            "duplicate_fit",
            "invalid_payload",
            "corrupt_json",
            "incomplete_temporary",
            "invalid_telemetry",
            "invalid_failure_telemetry",
            "retry_budget_exceeded",
            "invalid_status",
        }
        for error in errors
    )
    return {
        "valid": not blocking,
        "resumable": not blocking and (missing > 0 or retryable_failed > 0),
        "expected": len(expected),
        "completed": completed,
        "failed": failed,
        "missing": missing,
        "unexpected": sum(error["code"] == "unexpected_fit" for error in errors),
        "errors": errors,
        "completion_status": "completed"
        if completed == len(expected) and failed == 0 and not errors
        else "incomplete",
    }


def run(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    resume: bool = False,
    execute_fit: Callable[
        [dict[str, Any], Any, Any, Any], dict[str, Any]
    ] = _execute_fit,
    dataset_loader: Callable[..., Any] = load_dataset,
) -> dict[str, Any]:
    validate_execution_plan(plan)
    root = (repo_root or find_repo_root()).resolve()
    _allowed_output(output_dir, plan, root)
    if output_dir.exists() and not resume:
        raise P7C2HarnessError("refusing to overwrite an existing output directory")
    if resume:
        state = validate_artifacts(plan, output_dir)
        if not state["valid"] or not state["resumable"]:
            raise P7C2HarnessError("artifact set is not safely resumable")
    provenance = capture_git_provenance(root, required=True)
    if not resume:
        output_dir.mkdir(parents=True)
        _atomic_json(output_dir / "execution_plan.json", plan)
        _atomic_json(
            output_dir / "environment.json",
            {
                "schema_version": 1,
                "plan_digest": plan["plan_digest"],
                "threading": plan["threading"],
                **provenance,
            },
        )
    else:
        environment = json.loads(
            (output_dir / "environment.json").read_text(encoding="utf-8")
        )
        if (
            environment.get("plan_digest") != plan["plan_digest"]
            or environment.get("threading") != plan["threading"]
        ):
            raise P7C2HarnessError("resume plan/threading policy mismatch")
        if any(
            environment.get(key) != provenance.get(key)
            for key in ("git_head", "working_tree", "working_tree_details")
        ):
            raise P7C2HarnessError("resume Git provenance mismatch")
    protocol = load_protocol_a_config(repo_root=root)
    dataset_cache: dict[str, Any] = {}
    outer_cache: dict[str, Any] = {}
    completed = skipped = failed = 0
    for fit in plan["fits"]:
        result_path = output_dir / fit["artifact_path"]
        if resume and _valid_completed(result_path, fit):
            skipped += 1
            continue
        if fit["dataset_id"] not in dataset_cache:
            dataset = dataset_loader(fit["dataset_id"], repo_root=root)
            dataset_cache[fit["dataset_id"]] = dataset
            nested = create_nested_cv_definition(
                dataset,
                dataset_checksum=fit["dataset_checksum"],
                outer_n_repeats=1,
                outer_n_splits=2,
                inner_n_splits=5,
                random_seed=42,
            )
            outer_cache[fit["dataset_id"]] = next(
                x for x in nested.outer_folds if x.outer_fold_id == EXPECTED_OUTER
            )
        inner = outer_cache[fit["dataset_id"]].inner_folds[fit["inner_fold_index"]]
        failure_path = result_path
        previous_attempts = 0
        previous_error_type: str | None = None
        if failure_path.exists():
            try:
                previous_failure = json.loads(failure_path.read_text(encoding="utf-8"))
                previous_attempts = int(previous_failure.get("attempt_count", 0))
                previous_error_type = previous_failure.get("error", {}).get("type")
            except (OSError, json.JSONDecodeError):
                raise P7C2HarnessError(
                    "corrupt failure artifact must be preserved and inspected"
                )
        if previous_attempts and (
            previous_attempts > plan["retry_policy"]["max_retry_attempts"]
            or (
                plan["retry_policy"].get("transient_errors_only")
                and previous_error_type not in {"OSError", "TimeoutError"}
            )
        ):
            failed += 1
            continue
        attempt = previous_attempts + 1
        try:
            payload = execute_fit(
                fit, dataset_cache[fit["dataset_id"]], inner, protocol
            )
            payload.update(
                {
                    "attempt_count": attempt,
                    "git_provenance": provenance,
                    "artifact_bytes": 0,
                }
            )
            _atomic_json(result_path, payload)
            completed += 1
        except KeyboardInterrupt:
            _atomic_json(
                output_dir / "engineering_summary.json",
                {
                    "completion_status": "interrupted",
                    "expected": 60,
                    "completed": completed + skipped,
                    "failed": failed,
                    "pending": 60 - completed - skipped - failed,
                    "plan_digest": plan["plan_digest"],
                },
            )
            raise
        except Exception as exc:
            process = psutil.Process()
            rss = int(process.memory_info().rss)
            failure = {
                "schema_version": 1,
                "status": "failed",
                "outcome": "failed",
                "fit_id": fit["fit_id"],
                "plan_digest": fit["plan_digest"],
                "model_id": fit["model_id"],
                "dataset_id": fit["dataset_id"],
                "candidate_id": fit["candidate_id"],
                "outer_repeat_index": fit["outer_repeat_index"],
                "outer_fold_index": fit["outer_fold_index"],
                "inner_fold_index": fit["inner_fold_index"],
                "seed": fit["seed"],
                "attempt_count": attempt,
                "error": _safe_error(exc),
                "git_provenance": provenance,
                "process_id": process.pid,
                "configured_thread_count": 1,
                "effective_thread_count": 1,
                "process_rss_start_bytes": rss,
                "process_rss_peak_bytes": rss,
                "process_rss_delta_peak_bytes": 0,
                "process_cpu_seconds": float(time.process_time()),
                "started_at": _now(),
                "completed_at": _now(),
                "duration_seconds": 0.0,
                "library_versions": {
                    "python": platform.python_version(),
                    "scikit_learn": sklearn.__version__,
                    "xgboost": xgboost.__version__,
                },
            }
            _atomic_json(failure_path, failure)
            failed += 1
    summary = {
        "completion_status": "completed"
        if completed + skipped == 60 and failed == 0
        else "completed_with_failures",
        "expected": 60,
        "completed": completed + skipped,
        "skipped": skipped,
        "failed": failed,
        "pending": 60 - completed - skipped - failed,
        "plan_digest": plan["plan_digest"],
    }
    _atomic_json(output_dir / "engineering_summary.json", summary)
    return summary
