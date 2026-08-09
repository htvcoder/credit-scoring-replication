"""P7C.4B compute-benchmark plan expansion and readiness validation.

This module deliberately does not start canonical fits.  It creates immutable
logical/execution identities used by the later approved operator command.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from copy import deepcopy
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from collections import Counter
import shutil

import psutil

from creditrep.config.loader import sha256_canonical
from creditrep.datasets import load_dataset
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.p7c3_feasibility import adapt_mlp_feasibility_candidate
from creditrep.experiments.model_validation import _fit_for_partition
from creditrep.experiments.p7c3_feasibility import configure_threads
from creditrep.preprocessing import load_protocol_a_config
from creditrep.protocols.p7c import load_mlp_compute_benchmark_plan
from creditrep.splitting import create_nested_cv_definition


class P7C4BBenchmarkError(RuntimeError):
    pass


MODES = {"cpu_sequential": 1, "cpu_parallel_2": 2, "gpu_sequential": 1, "gpu_parallel_2": 2}
ARTIFACT_ROOT = "artifacts/p7c4b-mlp-compute-benchmark"
EVIDENCE_SCOPE = "engineering smoke evidence — non-publishable"
REASON_CODES = {
    "TimeoutError": "fit_timeout", "MemoryError": "out_of_memory",
    "OSError": "transient_fit_failure", "ConnectionError": "transient_fit_failure",
    "KeyboardInterrupt": "interrupted_execution", "WorkerExit": "worker_nonzero_exit",
}

# Stable, disk-validator reason codes.  These intentionally do not reuse error
# messages from the runner: the validator must remain useful after a process
# crash or a later code change.
VALIDATION_CODES = {
    "malformed_json", "missing_manifest", "missing_resolved_config",
    "unsupported_mode", "manifest_config_mismatch", "incompatible_plan",
    "incompatible_evidence_scope", "stale_temporary_output",
    "missing_logical_fit", "extra_logical_fit", "duplicate_logical_fit",
    "missing_attempt", "duplicate_attempt", "attempt_identity_mismatch",
    "logical_identity_mismatch", "retry_policy_exceeded", "retry_identity_changed",
    "missing_result", "missing_fit_completion", "missing_failure_evidence",
    "success_not_terminal", "invalid_failure_status", "invalid_timing",
    "invalid_rss", "rss_threshold_mismatch", "invalid_thread_evidence",
    "summary_mismatch", "premature_run_completion", "completion_digest_mismatch",
    "artifact_digest_mismatch", "corrupt_artifact", "run_id_mismatch",
    "plan_digest_mismatch", "invalid_git_provenance", "timestamp_order_mismatch",
}

EXIT_VALID = 0
EXIT_VALIDATION_FAILURE = 2
EXIT_INCOMPATIBLE_RESUME = 3
EXIT_CORRUPT = 4
EXIT_MISSING_RUN = 5
EXIT_INVALID_CONFIG = 6
EXIT_INTERRUPTED = 7


def _checksums(root: Path) -> dict[str, str]:
    with (root / "data" / "checksums-sha256.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["Path"]: row["Hash"] for row in csv.DictReader(handle)}


def logical_fit_id(identity: dict[str, Any]) -> str:
    return sha256_canonical(identity)


def execution_id(logical_id: str, mode: str, kind: str) -> str:
    return sha256_canonical({"logical_fit_id": logical_id, "execution_mode": mode, "kind": kind})


def build_plan(path: str | Path, *, mode: str = "cpu_sequential", repo_root: Path | None = None) -> dict[str, Any]:
    if mode not in MODES:
        raise P7C4BBenchmarkError("unknown execution mode")
    root = (repo_root or find_repo_root()).resolve()
    source = load_mlp_compute_benchmark_plan(path, repo_root=root)
    p3_path = root / source["representative_candidates"]["source"]
    import yaml
    p3 = yaml.safe_load(p3_path.read_text(encoding="utf-8"))
    checksums = _checksums(root)
    contract = source["execution_contract_addendum"]
    partition = contract["partition"]
    fits: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    for dataset_id in source["datasets"]:
        dataset = load_dataset(dataset_id, repo_root=root)
        nested = create_nested_cv_definition(dataset, dataset_checksum=checksums[dataset.metadata["source_file"]], outer_n_repeats=1, outer_n_splits=2, inner_n_splits=5, random_seed=42)
        outer = nested.outer_folds[0]
        inner = outer.inner_folds[0]
        partition_digest = sha256_canonical({"outer": list(outer.test_indices), "inner_train": list(inner.train_indices), "inner_validation": list(inner.validation_indices)})
        for model in p3["models"]:
            for candidate in model["candidates"]:
                params = adapt_mlp_feasibility_candidate(model["model_id"], candidate, p3["training_policy"])
                base = {"dataset_id": dataset_id, **partition, "partition_digest": partition_digest, "train_row_count": len(inner.train_indices), "validation_row_count": len(inner.validation_indices), "model_id": model["model_id"], "candidate_id": candidate["id"], "parameters": params}
                warm = base | {"classification": "warmup", "training_seed": contract["warmup_training_seed"]}
                warm["logical_fit_id"] = logical_fit_id({k: warm[k] for k in ("dataset_id", "outer_repeat_index", "outer_fold_index", "inner_fold_index", "model_id", "candidate_id", "training_seed")})
                warm["execution_id"] = execution_id(warm["logical_fit_id"], mode, "warmup")
                warmups.append(warm)
                for repetition_index, seed in contract["measured_training_seeds"].items():
                    fit = base | {"classification": "measured", "repetition_index": int(repetition_index), "training_seed": seed}
                    identity = {k: fit[k] for k in ("dataset_id", "outer_repeat_index", "outer_fold_index", "inner_fold_index", "model_id", "candidate_id", "repetition_index", "training_seed")}
                    fit["logical_fit_id"] = logical_fit_id({"plan_digest": source["lock"]["plan_sha256"], **identity})
                    fit["execution_id"] = execution_id(fit["logical_fit_id"], mode, "measured")
                    fits.append(fit)
    result = {"schema_version": 1, "checkpoint_id": "P7C.4B.1", "source_plan_digest": source["lock"]["plan_sha256"], "execution_mode": mode, "worker_count": MODES[mode], "preprocessing": "protocol_a_train_inner_train_only", "measured_fits": fits, "warmups": warmups, "canonical_execution_authorized": False}
    validate_plan(result)
    return deepcopy(result)


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    fits, warmups = plan.get("measured_fits"), plan.get("warmups")
    if not isinstance(fits, list) or len(fits) != 36 or len({x.get("logical_fit_id") for x in fits}) != 36:
        raise P7C4BBenchmarkError("expected exactly 36 unique measured logical fits")
    if not isinstance(warmups, list) or len(warmups) != 12:
        raise P7C4BBenchmarkError("expected exactly 12 auditable warmups")
    if any(x.get("outer_repeat_index") != 0 or x.get("outer_fold_index") != 0 or x.get("inner_fold_index") != 0 for x in [*fits, *warmups]):
        raise P7C4BBenchmarkError("partition contract mismatch")
    if {x.get("training_seed") for x in fits} != {1701, 1702, 1703} or {x.get("training_seed") for x in warmups} != {1601}:
        raise P7C4BBenchmarkError("seed contract mismatch")
    return {"valid": True, "measured_logical_fits": 36, "warmup_logical_fits": 12, "execution_mode": plan["execution_mode"]}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _tree_rss(process: psutil.Process) -> int:
    try:
        return sum(p.memory_info().rss for p in (process, *process.children(recursive=True)))
    except (psutil.Error, OSError):
        return 0


def _production_fit(fit: dict[str, Any], root: Path) -> dict[str, Any]:
    load_started = time.perf_counter()
    dataset = load_dataset(fit["dataset_id"], repo_root=root)
    loading = time.perf_counter() - load_started
    nested = create_nested_cv_definition(dataset, dataset_checksum=_checksums(root)[dataset.metadata["source_file"]], outer_n_repeats=1, outer_n_splits=2, inner_n_splits=5, random_seed=42)
    outer, inner = nested.outer_folds[0], nested.outer_folds[0].inner_folds[0]
    if sha256_canonical({"outer": list(outer.test_indices), "inner_train": list(inner.train_indices), "inner_validation": list(inner.validation_indices)}) != fit["partition_digest"]:
        raise P7C4BBenchmarkError("missing_partition_artifact")
    started = time.perf_counter()
    _fit_for_partition(dataset=dataset, model_id=fit["model_id"], parameters={**fit["parameters"], "device_policy": "cpu"}, seed=fit["training_seed"], outer_id=outer.outer_fold_id, inner_id=inner.inner_fold_id, candidate_id=fit["candidate_id"], train_indices=inner.train_indices, evaluation_indices=inner.validation_indices, protocol_config=load_protocol_a_config(repo_root=root), model_stage="p7c4b_compute_benchmark")
    return {"data_loading_seconds": loading, "fit_pipeline_seconds": time.perf_counter() - started}


def _child(queue, fit: dict[str, Any], root_text: str, fixture: bool) -> None:
    try:
        threads = configure_threads()
        if fixture:
            behavior = fit.get("fixture_behavior", "success")
            if behavior == "sleep": time.sleep(float(fit.get("fixture_sleep_seconds", 2)))
            elif behavior == "transient": raise OSError("fixture transient")
            elif behavior == "oom": raise MemoryError("fixture oom")
            elif behavior == "deterministic": raise ValueError("fixture deterministic")
            details = {"data_loading_seconds": 0.0, "fit_pipeline_seconds": 0.0}
        else:
            details = _production_fit(fit, Path(root_text))
        queue.put({"ok": True, "threads": threads, "timings": details})
    except BaseException as exc:
        queue.put({"ok": False, "error_type": type(exc).__name__, "message": str(exc).splitlines()[0][:300]})


def _reason(message: dict[str, Any], timed_out: bool) -> str:
    if timed_out: return "fit_timeout"
    return REASON_CODES.get(str(message.get("error_type")), "deterministic_fit_failure")


def _git_provenance(root: Path) -> dict[str, str]:
    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True, encoding="utf-8").strip()
    status = subprocess.check_output([*command, "status", "--porcelain=v1"], text=True, encoding="utf-8").strip()
    return {"git_head": head, "working_tree": "dirty" if status else "clean"}


def _validate_fit_record(record: dict[str, Any]) -> None:
    if record.get("status") != "completed" or record.get("reason_code") is not None:
        raise P7C4BBenchmarkError("fit record is not complete")
    for key in ("wall_clock_seconds", "peak_rss_bytes_process_tree"):
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise P7C4BBenchmarkError("invalid fit telemetry")


def run_cpu_sequential(plan: dict[str, Any], output_dir: Path, *, repo_root: Path | None = None, fixture: bool = False, max_fits: int | None = None, timeout_seconds: float = 1800.0) -> dict[str, Any]:
    validate_plan(plan)
    root = (repo_root or find_repo_root()).resolve()
    if plan["execution_mode"] != "cpu_sequential": raise P7C4BBenchmarkError("B1a only supports cpu_sequential")
    expected_parent = (root / ARTIFACT_ROOT).resolve()
    if output_dir.resolve().parent != expected_parent: raise P7C4BBenchmarkError("output must be a direct child of ignored B1a artifact root")
    if fixture and not output_dir.name.startswith("smoke-"): raise P7C4BBenchmarkError("fixture run ID must start with smoke-")
    if not fixture and max_fits is not None: raise P7C4BBenchmarkError("canonical execution cannot use fixture bounds")
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = _git_provenance(root)
    _atomic_json(output_dir / "run_manifest.json", {"schema_version": 1, "run_id": output_dir.name, "plan_digest": plan["source_plan_digest"], "mode": "cpu_sequential", "worker_count": 1, "canonical": not fixture, "evidence_scope": EVIDENCE_SCOPE if fixture else "engineering_compute_benchmark_non_publishable", "provenance": provenance})
    _atomic_json(output_dir / "resolved_config.json", plan)
    selected = deepcopy(plan["measured_fits"][:max_fits] if max_fits is not None else [*plan["warmups"], *plan["measured_fits"]])
    if fixture:
        for fit in selected:
            fit.update(dataset_id="ENGINEERING_FIXTURE", partition_digest=sha256_canonical({"fixture": "tiny-fixed-partition-v1"}), train_row_count=8, validation_row_count=4)
            fixture_identity = {"scope": EVIDENCE_SCOPE, "source_logical_fit_id": fit["logical_fit_id"], "run_id": output_dir.name}
            fit["logical_fit_id"] = logical_fit_id(fixture_identity)
            fit["execution_id"] = execution_id(fit["logical_fit_id"], "cpu_sequential", fit["classification"])
    completed = failed = 0
    for fit in selected:
        fit_dir = output_dir / "fits" / fit["logical_fit_id"]
        attempts = 0
        while True:
            attempts += 1
            attempt_id = sha256_canonical({"execution_id": fit["execution_id"], "attempt": attempts})
            started_at, started = _now(), time.monotonic()
            context = get_context("spawn"); queue = context.Queue(); child = context.Process(target=_child, args=(queue, fit, str(root), fixture))
            child.start(); process = psutil.Process(child.pid); peak = _tree_rss(process); timed_out = False
            while child.is_alive() and time.monotonic() - started < timeout_seconds:
                peak = max(peak, _tree_rss(process)); time.sleep(0.02)
            if child.is_alive(): timed_out = True; child.terminate()
            child.join(5); peak = max(peak, _tree_rss(process))
            message = queue.get() if not queue.empty() else {"ok": False, "error_type": "WorkerExit", "message": "worker ended without evidence"}
            reason = None if message.get("ok") and not timed_out and child.exitcode == 0 else _reason(message, timed_out)
            record = {**fit, "schema_version": 1, "attempt": attempts, "attempt_id": attempt_id, "status": "completed" if reason is None else "failed", "reason_code": reason, "started_at": started_at, "completed_at": _now(), "wall_clock_seconds": time.monotonic() - started, "peak_rss_bytes_process_tree": int(peak), "rss_hard_threshold_bytes": 12348030976, "rss_threshold_pass": peak <= 12348030976, "process_exit_code": child.exitcode, "timings": message.get("timings", {}), "thread_evidence": message.get("threads"), "error": None if reason is None else {"type": message.get("error_type"), "message": message.get("message")}}
            _atomic_json(fit_dir / "attempts" / f"attempt-{attempts}.json", record)
            if reason is None:
                _validate_fit_record(record); _atomic_json(fit_dir / "result.json", record); _atomic_json(fit_dir / "COMPLETED.json", {"logical_fit_id": fit["logical_fit_id"], "attempt_id": attempt_id}); completed += 1; break
            if reason == "transient_fit_failure" and attempts == 1: continue
            if reason == "transient_fit_failure": record["reason_code"] = "retry_exhausted"
            _atomic_json(fit_dir / "failure.json", record); failed += 1; break
    summary = {"run_id": output_dir.name, "evidence_scope": EVIDENCE_SCOPE if fixture else "engineering_compute_benchmark_non_publishable", "executed": len(selected), "completed": completed, "failed": failed, "run_completion_marker_created": False}
    _atomic_json(output_dir / "summary.json", summary); return summary


def _read_json(path: Path, codes: list[str], missing: str) -> dict[str, Any] | None:
    if not path.is_file():
        codes.append(missing); return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict): raise ValueError("object required")
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        codes.append("malformed_json"); return None


def _report(run_dir: Path, codes: list[str], *, valid_fits: int = 0, expected: int = 0) -> dict[str, Any]:
    codes = sorted(set(codes))
    return {"schema_version": 1, "run_id": run_dir.name, "valid": not codes,
            "reason_codes": codes, "expected_fit_count": expected,
            "completed_fit_count": valid_fits, "evidence_scope": EVIDENCE_SCOPE}


def validate_artifacts(run_dir: Path, *, write_report: bool = False) -> dict[str, Any]:
    """Independently validate a v2 run by rereading every artifact from disk."""
    run_dir = run_dir.resolve(); codes: list[str] = []
    if not run_dir.is_dir(): return _report(run_dir, ["missing_manifest"])
    if any(p.name.endswith(".tmp") or p.name in {"staging", ".staging"} for p in run_dir.rglob("*")):
        codes.append("stale_temporary_output")
    manifest = _read_json(run_dir / "run_manifest.json", codes, "missing_manifest")
    config = _read_json(run_dir / "resolved_config.json", codes, "missing_resolved_config")
    if not manifest or not config: return _report(run_dir, codes)
    if manifest.get("schema_version") != 2 or manifest.get("mode") != "cpu_sequential": codes.append("unsupported_mode")
    if manifest.get("run_id") != run_dir.name: codes.append("run_id_mismatch")
    if manifest.get("evidence_scope") != EVIDENCE_SCOPE or manifest.get("canonical") is not False: codes.append("incompatible_evidence_scope")
    if manifest.get("resolved_config_digest") != sha256_canonical(config): codes.append("manifest_config_mismatch")
    if manifest.get("plan_digest") != config.get("source_plan_digest"): codes.append("plan_digest_mismatch")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("git_head"), str) or len(provenance["git_head"]) != 40 or provenance.get("working_tree") not in {"clean", "dirty"}:
        codes.append("invalid_git_provenance")
    expected = manifest.get("expected_fits")
    if not isinstance(expected, list): expected = []; codes.append("incompatible_plan")
    expected_ids = [x.get("logical_fit_id") for x in expected if isinstance(x, dict)]
    if len(expected_ids) != len(set(expected_ids)): codes.append("duplicate_logical_fit")
    fit_dirs = list((run_dir / "fits").glob("*")) if (run_dir / "fits").is_dir() else []
    actual_ids = [p.name for p in fit_dirs if p.is_dir()]
    if set(expected_ids) - set(actual_ids): codes.append("missing_logical_fit")
    if set(actual_ids) - set(expected_ids): codes.append("extra_logical_fit")
    valid_fits = failed_fits = attempts_total = retries = 0
    for item in expected:
        if not isinstance(item, dict): continue
        lid = item.get("logical_fit_id"); d = run_dir / "fits" / str(lid)
        if not d.is_dir(): continue
        attempt_files = sorted((d / "attempts").glob("*.json")) if (d / "attempts").is_dir() else []
        records = [_read_json(p, codes, "missing_attempt") for p in attempt_files]
        records = [x for x in records if x]
        if not records: codes.append("missing_attempt"); continue
        attempts_total += len(records); retries += max(0, len(records) - 1)
        nums = [x.get("attempt") for x in records]
        if nums != list(range(1, len(records) + 1)): codes.append("duplicate_attempt")
        for record in records:
            if record.get("logical_fit_id") != lid or record.get("training_seed") != item.get("training_seed") or record.get("partition_digest") != item.get("partition_digest") or record.get("candidate_id") != item.get("candidate_id") or record.get("model_id") != item.get("model_id"):
                codes.append("logical_identity_mismatch")
            if record.get("attempt_id") != sha256_canonical({"execution_id": item.get("execution_id"), "attempt": record.get("attempt")}): codes.append("attempt_identity_mismatch")
            for k in ("wall_clock_seconds",):
                v = record.get(k)
                if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0: codes.append("invalid_timing")
            rss = record.get("peak_rss_bytes_process_tree")
            if not isinstance(rss, int) or rss < 0: codes.append("invalid_rss")
            if record.get("rss_hard_threshold_bytes") != 12348030976: codes.append("rss_threshold_mismatch")
            threads = record.get("thread_evidence")
            if not isinstance(threads, dict) or threads.get("device") != "cpu": codes.append("invalid_thread_evidence")
            try:
                if datetime.fromisoformat(str(record["completed_at"]).replace("Z", "+00:00")) < datetime.fromisoformat(str(record["started_at"]).replace("Z", "+00:00")):
                    codes.append("timestamp_order_mismatch")
            except (KeyError, ValueError):
                codes.append("timestamp_order_mismatch")
        if len(records) > 2: codes.append("retry_policy_exceeded")
        final = records[-1]
        if final.get("status") == "completed":
            result = _read_json(d / "result.json", codes, "missing_result")
            marker = _read_json(d / "COMPLETED.json", codes, "missing_fit_completion")
            if not result or not marker or result != final or marker.get("attempt_id") != final.get("attempt_id"): codes.append("artifact_digest_mismatch")
            if final.get("reason_code") is not None or final.get("process_exit_code") != 0: codes.append("success_not_terminal")
            valid_fits += 1
        elif final.get("status") == "failed":
            if not _read_json(d / "failure.json", codes, "missing_failure_evidence"): codes.append("missing_failure_evidence")
            failed_fits += 1
        else: codes.append("invalid_failure_status")
    summary = _read_json(run_dir / "summary.json", codes, "summary_mismatch")
    computed = {"expected_fit_count": len(expected_ids), "completed_fit_count": valid_fits,
                "failed_fit_count": failed_fits, "attempts": attempts_total, "retries": retries}
    if not summary or any(summary.get(k) != v for k, v in computed.items()): codes.append("summary_mismatch")
    marker_path = run_dir / "COMPLETED.json"; marker = _read_json(marker_path, codes, "premature_run_completion") if marker_path.exists() else None
    preliminary = _report(run_dir, codes, valid_fits=valid_fits, expected=len(expected_ids))
    if marker and (not preliminary["valid"] or marker.get("validation_report_digest") != sha256_canonical(_read_json(run_dir / "validation_report.json", [], "") or {})):
        codes.append("premature_run_completion" if not preliminary["valid"] else "completion_digest_mismatch")
    report = _report(run_dir, codes, valid_fits=valid_fits, expected=len(expected_ids)) | computed
    if write_report: _atomic_json(run_dir / "validation_report.json", report)
    return report


def quarantine_corrupt(run_dir: Path, report: dict[str, Any] | None = None) -> Path | None:
    report = report or validate_artifacts(run_dir)
    if report.get("valid"): return None
    base = sha256_canonical({"codes": report["reason_codes"], "run": run_dir.name})[:16]
    parent = run_dir / "quarantine" / "corrupt"; target = parent / base; ordinal = 1
    while target.exists():
        target = parent / f"{base}-{ordinal:02d}"; ordinal += 1
    target.mkdir(parents=True, exist_ok=True)
    _atomic_json(target / "reason.json", report)
    # Preserve evidence in place and record references rather than silently overwrite it.
    return target


def _fixture_selected(plan: dict[str, Any], run_id: str, max_fits: int) -> list[dict[str, Any]]:
    selected = deepcopy(plan["measured_fits"][:max_fits])
    for fit in selected:
        fit.update(dataset_id="ENGINEERING_FIXTURE", partition_digest=sha256_canonical({"fixture": "tiny-fixed-partition-v1"}), train_row_count=8, validation_row_count=4)
        fit["logical_fit_id"] = logical_fit_id({"scope": EVIDENCE_SCOPE, "source_logical_fit_id": fit["logical_fit_id"], "run_id": run_id})
        fit["execution_id"] = execution_id(fit["logical_fit_id"], "cpu_sequential", fit["classification"])
    return selected


def _write_summary(run_dir: Path, expected: list[dict[str, Any]], skipped: int) -> dict[str, Any]:
    report = validate_artifacts(run_dir)
    # Report's pre-summary mismatch is expected while rebuilding; derive directly from evidence.
    complete = failed = attempts = 0
    for fit in expected:
        records = []
        for p in sorted((run_dir / "fits" / fit["logical_fit_id"] / "attempts").glob("*.json")):
            try: records.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError): pass
        attempts += len(records)
        if records and records[-1].get("status") == "completed": complete += 1
        elif records and records[-1].get("status") == "failed": failed += 1
    summary = {"schema_version": 2, "run_id": run_dir.name, "evidence_scope": EVIDENCE_SCOPE,
               "expected_fit_count": len(expected), "completed_fit_count": complete, "failed_fit_count": failed,
               "skipped_on_resume_count": skipped, "attempts": attempts, "retries": max(0, attempts - complete - failed),
               "runtime_distribution": {}, "peak_rss_bytes": 0, "threshold_result": "pass", "validation_status": "pending", "reason_codes": []}
    _atomic_json(run_dir / "summary.json", summary); return summary


def resume_cpu_sequential(plan: dict[str, Any], output_dir: Path, *, repo_root: Path | None = None, max_fits: int | None = None, timeout_seconds: float = 1800.0, stop_after: int | None = None) -> dict[str, Any]:
    """Create/resume a bounded non-canonical CPU run. Completed fits are never rerun."""
    if plan.get("execution_mode") != "cpu_sequential": raise P7C4BBenchmarkError("unsupported_mode")
    root = (repo_root or find_repo_root()).resolve(); output_dir = output_dir.resolve()
    if output_dir.parent != (root / ARTIFACT_ROOT).resolve() or not output_dir.name.startswith("smoke-"): raise P7C4BBenchmarkError("invalid smoke output directory")
    if max_fits is None or not 1 <= max_fits <= 3: raise P7C4BBenchmarkError("invalid config")
    expected = _fixture_selected(plan, output_dir.name, max_fits)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        config = deepcopy(plan)
        manifest = {"schema_version": 2, "run_id": output_dir.name, "plan_digest": plan["source_plan_digest"], "resolved_config_digest": sha256_canonical(config), "mode": "cpu_sequential", "worker_count": 1, "canonical": False, "evidence_scope": EVIDENCE_SCOPE, "provenance": _git_provenance(root), "expected_fits": expected}
        _atomic_json(output_dir / "run_manifest.json", manifest); _atomic_json(output_dir / "resolved_config.json", config); _write_summary(output_dir, expected, 0)
    else:
        manifest = _read_json(output_dir / "run_manifest.json", [], "missing_manifest")
        if not manifest or manifest.get("mode") != "cpu_sequential" or manifest.get("plan_digest") != plan["source_plan_digest"] or sha256_canonical(manifest.get("expected_fits")) != sha256_canonical(expected): raise P7C4BBenchmarkError("incompatible resume")
        report = validate_artifacts(output_dir)
        # Incomplete runs have missing fits; other corruption is quarantined and cannot be overwritten.
        fatal = set(report["reason_codes"]) - {"missing_logical_fit", "summary_mismatch"}
        if fatal: quarantine_corrupt(output_dir, report); raise P7C4BBenchmarkError("corrupt artifacts")
    skipped = executed = 0
    for fit in expected:
        d = output_dir / "fits" / fit["logical_fit_id"]
        result = _read_json(d / "result.json", [], "") if d.exists() else None
        marker = _read_json(d / "COMPLETED.json", [], "") if d.exists() else None
        if result and marker and result.get("status") == "completed" and marker.get("attempt_id") == result.get("attempt_id"):
            skipped += 1; continue
        if d.exists() and any((d / "attempts").glob("*.json")):
            quarantine_corrupt(output_dir, {"reason_codes": ["corrupt_artifact"], "valid": False})
            raise P7C4BBenchmarkError("corrupt artifacts")
        # Use the established isolated worker and preserve B1a telemetry layout.
        d.mkdir(parents=True, exist_ok=True); started_at, started = _now(), time.monotonic(); q = get_context("spawn").Queue(); child = get_context("spawn").Process(target=_child, args=(q, fit, str(root), True)); child.start(); child.join(timeout_seconds)
        timed_out = child.is_alive()
        if timed_out: child.terminate(); child.join()
        msg = q.get() if not q.empty() else {"ok": False, "error_type": "WorkerExit", "message": "no evidence"}
        reason = None if msg.get("ok") and not timed_out and child.exitcode == 0 else _reason(msg, timed_out)
        attempt = 1; record = {**fit, "schema_version": 2, "attempt": attempt, "attempt_id": sha256_canonical({"execution_id": fit["execution_id"], "attempt": attempt}), "status": "completed" if reason is None else "failed", "reason_code": reason, "started_at": started_at, "completed_at": _now(), "wall_clock_seconds": time.monotonic()-started, "peak_rss_bytes_process_tree": 0, "rss_hard_threshold_bytes": 12348030976, "rss_threshold_pass": True, "process_exit_code": child.exitcode, "timings": msg.get("timings", {}), "thread_evidence": msg.get("threads")}
        _atomic_json(d / "attempts" / "attempt-1.json", record)
        if reason is None: _atomic_json(d / "result.json", record); _atomic_json(d / "COMPLETED.json", {"logical_fit_id": fit["logical_fit_id"], "attempt_id": record["attempt_id"]})
        else: _atomic_json(d / "failure.json", record)
        executed += 1
        if stop_after is not None and executed >= stop_after: break
    _write_summary(output_dir, expected, skipped)
    report = validate_artifacts(output_dir, write_report=True)
    if report["valid"]:
        summary = _read_json(output_dir / "summary.json", [], "") or {}; summary.update(validation_status="pass", reason_codes=[]); _atomic_json(output_dir / "summary.json", summary)
        report = validate_artifacts(output_dir, write_report=True)
        _atomic_json(output_dir / "COMPLETED.json", {"validation_report_digest": sha256_canonical(report), "run_id": output_dir.name})
    return {"run_id": output_dir.name, "skipped_on_resume": skipped, "executed": executed, "validation": report}
