"""Execution harness for the non-publishable P7C.3 MLP feasibility sample.

The module deliberately separates immutable-plan validation from execution.  It
does not calculate predictive metrics and records one atomic, auditable record
per inner-fold fit.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import psutil

from creditrep.config.loader import sha256_canonical
from creditrep.datasets import load_dataset
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.model_validation import _fit_for_partition
from creditrep.preprocessing import ProtocolAConfig
from creditrep.protocols.p7c import load_mlp_feasibility_plan
from creditrep.splitting import create_nested_cv_definition


class P7C3HarnessError(RuntimeError):
    pass


ARTIFACT_ROOT = "artifacts/p7c3-mlp-feasibility"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
FAILURE_CLASSES = {
    "timeout",
    "oom_resource_exhaustion",
    "transient_infrastructure",
    "implementation_data_configuration",
    "interrupted_cancelled",
    "policy_violation",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def stable_fit_id(identity: dict[str, Any]) -> str:
    return sha256_canonical(identity)


def _dataset_checksums(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (root / "data" / "checksums-sha256.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            result[row["Path"]] = row["Hash"]
    return result


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def capture_provenance(root: Path) -> dict[str, Any]:
    try:
        dirty = _git(root, "status", "--porcelain=v1").splitlines()
        return {
            "git_head": _git(root, "rev-parse", "HEAD"),
            "working_tree": "dirty" if dirty else "clean",
            "working_tree_details": {"is_dirty": bool(dirty), "porcelain_v1": dirty},
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        raise P7C3HarnessError(f"Git provenance unavailable: {exc}") from exc


def build_execution_plan(
    plan_path: str | Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    source = load_mlp_feasibility_plan(plan_path, repo_root=root)
    digest = source["lock"]["plan_sha256"]
    checksums = _dataset_checksums(root)
    fits: list[dict[str, Any]] = []
    for dataset_id in source["datasets"]:
        dataset = load_dataset(dataset_id, repo_root=root)
        source_path = dataset.metadata["source_file"]
        if source_path not in checksums:
            raise P7C3HarnessError(f"dataset checksum is not registered: {source_path}")
        for model in source["models"]:
            for candidate in model["candidates"]:
                for inner_fold_index in range(source["inner_folds"]):
                    identity = {
                        "plan_digest": digest,
                        "dataset_id": dataset_id,
                        "model_id": model["model_id"],
                        "candidate_id": candidate["id"],
                        "outer_repeat_index": 0,
                        "outer_fold_index": 0,
                        "inner_fold_index": inner_fold_index,
                        "seed": 42 + inner_fold_index,
                    }
                    fit_id = stable_fit_id(identity)
                    fits.append(
                        identity
                        | {
                            "fit_id": fit_id,
                            "parameters": {
                                key: value
                                for key, value in candidate.items()
                                if key != "id"
                            },
                            "dataset_checksum": checksums[source_path],
                            "artifact_path": f"fits/{fit_id}/result.json",
                        }
                    )
    execution = {
        "schema_version": 1,
        "checkpoint_id": "P7C.3",
        "purpose": "engineering_feasibility_only",
        "plan_digest": digest,
        "artifact_root": ARTIFACT_ROOT,
        "fits": fits,
        "threading": {
            "fits_parallelism": 1,
            "torch_intraop_threads": 2,
            "blas_openmp_threads": 2,
            "nested_parallelism": False,
        },
        "limits": deepcopy(source["compute_policy"]),
        "retry_policy": {"max_retry_attempts": 1, "transient_errors_only": True},
    }
    validate_execution_plan(execution)
    return execution


def validate_execution_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if (
        plan.get("schema_version") != 1
        or plan.get("checkpoint_id") != "P7C.3"
        or plan.get("purpose") != "engineering_feasibility_only"
    ):
        raise P7C3HarnessError("invalid P7C.3 execution identity")
    if plan.get("artifact_root") != ARTIFACT_ROOT or plan.get("threading") != {
        "fits_parallelism": 1,
        "torch_intraop_threads": 2,
        "blas_openmp_threads": 2,
        "nested_parallelism": False,
    }:
        raise P7C3HarnessError("sequential two-thread policy mismatch")
    limits = plan.get("limits", {})
    if (
        limits.get("per_fit_timeout_seconds") != 1800
        or limits.get("total_wall_time_seconds") != 43200
        or limits.get("rss_warning_bytes") != 10 * 1024**3
        or limits.get("rss_hard_bytes") != int(11.5 * 1024**3)
        or limits.get("disk_free_floor_bytes") != 15 * 1024**3
    ):
        raise P7C3HarnessError("compute policy limits mismatch")
    fits = plan.get("fits")
    if (
        not isinstance(fits, list)
        or len(fits) != 60
        or len({fit.get("fit_id") for fit in fits}) != 60
    ):
        raise P7C3HarnessError("expected exactly 60 unique fits")
    if {fit.get("model_id") for fit in fits} != {"mlp_1", "mlp_3", "mlp_5"}:
        raise P7C3HarnessError("all MLP depths must be covered")
    for fit in fits:
        identity = {
            key: fit[key]
            for key in (
                "plan_digest",
                "dataset_id",
                "model_id",
                "candidate_id",
                "outer_repeat_index",
                "outer_fold_index",
                "inner_fold_index",
                "seed",
            )
        }
        if fit.get("fit_id") != stable_fit_id(identity):
            raise P7C3HarnessError("fit identity digest mismatch")
    return {
        "valid": True,
        "expected_fits": 60,
        "unique_fit_ids": 60,
        "plan_digest": plan["plan_digest"],
    }


def configure_threads() -> dict[str, Any]:
    for name in THREAD_ENV:
        os.environ[name] = "2"
    from creditrep.models.neural.runtime import require_torch

    torch = require_torch()
    torch.set_num_threads(2)
    interop = None
    try:
        torch.set_num_interop_threads(1)
        interop = torch.get_num_interop_threads()
    except RuntimeError:
        interop = "already_initialized"
    return {
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": interop,
        "environment": {name: os.environ[name] for name in THREAD_ENV},
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }


def preflight(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    validate_execution_plan(plan)
    root = (repo_root or find_repo_root()).resolve()
    if output_dir.resolve().parent != (root / ARTIFACT_ROOT).resolve():
        raise P7C3HarnessError(
            "output directory must be a direct child of the ignored P7C.3 artifact root"
        )
    if output_dir.exists() and not resume:
        raise P7C3HarnessError("refusing to overwrite an existing output directory")
    disk = shutil.disk_usage(root).free
    if disk < plan["limits"]["disk_free_floor_bytes"]:
        raise P7C3HarnessError("disk free space is below policy floor")
    for dataset_id in ("TC", "GMC"):
        load_dataset(dataset_id, repo_root=root)
    return {
        "valid": True,
        "plan_digest": plan["plan_digest"],
        "expected_fits": 60,
        "disk_free_bytes": disk,
        "device_policy": "cpu",
        "concurrency": 1,
        "thread_policy": {"requested": 2, "applied_at_worker_start": True},
        "provenance": capture_provenance(root),
    }


def _tree_rss(process: psutil.Process) -> int:
    try:
        return sum(
            member.memory_info().rss
            for member in [process, *process.children(recursive=True)]
        )
    except (psutil.Error, OSError):
        return 0


def _child(queue, fit: dict[str, Any], root_text: str) -> None:
    try:
        threads = configure_threads()
        root = Path(root_text)
        dataset = load_dataset(fit["dataset_id"], repo_root=root)
        nested = create_nested_cv_definition(
            dataset,
            dataset_checksum=fit["dataset_checksum"],
            outer_n_repeats=1,
            outer_n_splits=2,
            inner_n_splits=5,
            random_seed=42,
        )
        outer = nested.outer_folds[0]
        inner = outer.inner_folds[fit["inner_fold_index"]]
        _fit_for_partition(
            dataset=dataset,
            model_id=fit["model_id"],
            parameters=fit["parameters"],
            seed=fit["seed"],
            outer_id=outer.outer_fold_id,
            inner_id=inner.inner_fold_id,
            candidate_id=fit["candidate_id"],
            train_indices=inner.train_indices,
            evaluation_indices=inner.validation_indices,
            protocol_config=ProtocolAConfig(),
            model_stage="p7c3_feasibility",
        )
        queue.put({"ok": True, "threads": threads})
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc).splitlines()[0][:200],
            }
        )


def _classify(error_type: str) -> str:
    if error_type in {"TimeoutError"}:
        return "timeout"
    if error_type in {"MemoryError"}:
        return "oom_resource_exhaustion"
    if error_type in {"OSError", "ConnectionError"}:
        return "transient_infrastructure"
    return "implementation_data_configuration"


def _valid_completed(path: Path, fit: dict[str, Any]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and all(
            payload.get(key) == fit.get(key)
            for key in (
                "fit_id",
                "plan_digest",
                "dataset_id",
                "model_id",
                "candidate_id",
                "inner_fold_index",
                "seed",
            )
        )
        and isinstance(payload.get("provenance"), dict)
    )


def run(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    root = (repo_root or find_repo_root()).resolve()
    base = preflight(plan, output_dir, repo_root=root, resume=resume)
    if resume and not validate_artifacts(plan, output_dir).get("resumable"):
        raise P7C3HarnessError("artifact set is not safely resumable")
    if not resume:
        output_dir.mkdir(parents=True)
        _atomic_json(output_dir / "execution_plan.json", plan)
        _atomic_json(output_dir / "environment.json", base)
    started = time.monotonic()
    completed = skipped = failed = 0
    for fit in plan["fits"]:
        if time.monotonic() - started > plan["limits"]["total_wall_time_seconds"]:
            break
        path = output_dir / fit["artifact_path"]
        if resume and _valid_completed(path, fit):
            skipped += 1
            continue
        attempts = 0
        while True:
            attempts += 1
            wall = time.monotonic()
            queue = get_context("spawn").Queue()
            child = get_context("spawn").Process(
                target=_child, args=(queue, fit, str(root))
            )
            child.start()
            proc = psutil.Process(child.pid)
            peak = _tree_rss(proc)
            timeout = False
            while (
                child.is_alive()
                and time.monotonic() - wall <= plan["limits"]["per_fit_timeout_seconds"]
            ):
                peak = max(peak, _tree_rss(proc))
                time.sleep(0.1)
                if peak > plan["limits"]["rss_hard_bytes"]:
                    child.terminate()
                    break
            if child.is_alive():
                timeout = True
                child.terminate()
            child.join(10)
            peak = max(peak, _tree_rss(proc))
            message = (
                queue.get()
                if not queue.empty()
                else {
                    "ok": False,
                    "error_type": "TimeoutError" if timeout else "WorkerExit",
                    "message": "worker ended without result",
                }
            )
            status = (
                "completed"
                if message.get("ok")
                and not timeout
                and peak <= plan["limits"]["rss_hard_bytes"]
                else "failed"
            )
            classification = (
                "policy_violation"
                if peak > plan["limits"]["rss_hard_bytes"]
                else (
                    "timeout"
                    if timeout
                    else _classify(message.get("error_type", "WorkerExit"))
                )
            )
            record = fit | {
                "schema_version": 1,
                "status": status,
                "outcome": status,
                "attempt_count": attempts,
                "started_at": _now(),
                "completed_at": _now(),
                "wall_clock_seconds": time.monotonic() - wall,
                "process_exit_code": child.exitcode,
                "peak_rss_bytes_process_tree": peak,
                "rss_threshold_state": "hard_stop"
                if peak > plan["limits"]["rss_hard_bytes"]
                else (
                    "warning" if peak >= plan["limits"]["rss_warning_bytes"] else "ok"
                ),
                "timeout": timeout,
                "failure_classification": None
                if status == "completed"
                else classification,
                "effective_threads": message.get("threads"),
                "provenance": base["provenance"],
                "device": "cpu",
                "error": None
                if status == "completed"
                else {
                    "type": message.get("error_type"),
                    "message": message.get("message", "worker failure"),
                },
            }
            _atomic_json(path, record)
            if status == "completed":
                completed += 1
                break
            if classification == "transient_infrastructure" and attempts <= 1:
                continue
            failed += 1
            break
    summary = {
        "completion_status": "completed"
        if completed + skipped == 60 and failed == 0
        else "incomplete",
        "expected": 60,
        "completed": completed + skipped,
        "skipped": skipped,
        "failed": failed,
        "pending": 60 - completed - skipped - failed,
        "plan_digest": plan["plan_digest"],
    }
    _atomic_json(output_dir / "engineering_summary.json", summary)
    return summary


def validate_artifacts(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_execution_plan(plan)
    expected = {fit["fit_id"]: fit for fit in plan["fits"]}
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    completed = failed = 0
    if not output_dir.exists():
        return {
            "valid": False,
            "resumable": False,
            "expected": 60,
            "completed": 0,
            "missing": 60,
            "errors": [{"code": "missing_output"}],
        }
    try:
        if (
            json.loads((output_dir / "execution_plan.json").read_text(encoding="utf-8"))
            != plan
        ):
            errors.append({"code": "plan_mismatch"})
    except (OSError, json.JSONDecodeError):
        errors.append({"code": "missing_or_corrupt_plan"})
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
            errors.append({"code": "corrupt_json", "path": str(path)})
            continue
        fit_id = payload.get("fit_id")
        if fit_id not in expected:
            errors.append({"code": "unexpected_fit", "path": str(path)})
            continue
        if fit_id in seen:
            errors.append({"code": "duplicate_fit", "path": str(path)})
            continue
        seen.add(fit_id)
        if payload.get("plan_digest") != plan["plan_digest"] or not isinstance(
            payload.get("provenance"), dict
        ):
            errors.append({"code": "invalid_payload", "path": str(path)})
            continue
        if payload.get("status") == "completed":
            completed += 1
        elif payload.get("status") == "failed":
            failed += 1
        else:
            errors.append({"code": "invalid_status", "path": str(path)})
    missing = 60 - len(seen)
    blocking = bool(errors)
    return {
        "valid": not blocking,
        "resumable": not blocking and (missing > 0 or failed > 0),
        "expected": 60,
        "completed": completed,
        "failed": failed,
        "missing": missing,
        "errors": errors,
        "completion_status": "completed"
        if completed == 60 and failed == 0 and not errors
        else "incomplete",
    }
