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
