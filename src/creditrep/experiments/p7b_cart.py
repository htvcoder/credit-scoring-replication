"""P7B.1 CART feasibility runner: inner-evaluation-only, never scientific selection."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
import tracemalloc
from threading import Event, Thread
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from creditrep.checksums import get_dataset_checksum
from creditrep.config.loader import sha256_canonical
from creditrep.datasets.loader import load_dataset
from creditrep.datasets.registry import find_repo_root
from creditrep.experiments.nested_cv import _fit_preprocessing
from creditrep.models.factory import create_model
from creditrep.preprocessing import load_protocol_a_config
from creditrep.protocols.p7a import (
    effective_min_samples_leaf,
    load_manifest,
    manifest_hash,
)
from creditrep.splitting import create_nested_cv_definition
import psutil

P7B_FLAGS = {
    "phase": "P7B",
    "purpose": "engineering_feasibility",
    "publishable": False,
    "scientific_model_selection": False,
    "candidate_selection": "none",
    "predictive_ranking": False,
    "outer_selected_model_refit": False,
}
EXPECTED_DATASETS = ("AC", "HMEQ", "GMC")
EXPECTED_OUTER = "repeat_00_fold_00"
RSS_SAMPLE_INTERVAL_SECONDS = 0.05
ARTIFACT_VALIDATOR_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class P7BContractError(ValueError):
    """Raised when an immutable P7B.1 contract is violated."""


class GitProvenanceError(P7BContractError):
    """Raised when a training run cannot be tied to a Git commit."""


class ProcessRssSampler:
    """Process-local RSS sampler; child processes are intentionally excluded."""

    def __init__(self, interval_seconds: float = RSS_SAMPLE_INTERVAL_SECONDS) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.process_id = self.process.pid
        self.start_bytes = 0
        self.peak_bytes = 0
        self._stop = Event()
        self._thread: Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.peak_bytes = max(
                    self.peak_bytes, int(self.process.memory_info().rss)
                )
            except psutil.Error:
                return

    def start(self) -> None:
        self.start_bytes = int(self.process.memory_info().rss)
        self.peak_bytes = self.start_bytes
        self._thread = Thread(target=self._sample, name="p7b-rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 3)
        try:
            self.peak_bytes = max(self.peak_bytes, int(self.process.memory_info().rss))
        except psutil.Error:
            pass
        return {
            "process_rss_start_bytes": int(self.start_bytes),
            "process_rss_peak_bytes": int(self.peak_bytes),
            "process_rss_delta_peak_bytes": int(
                max(0, self.peak_bytes - self.start_bytes)
            ),
            "process_rss_sampling_interval_seconds": self.interval_seconds,
            "process_id": self.process_id,
            "child_processes_included": False,
            "measurement_method": "psutil.Process.memory_info().rss sampled process-locally",
            "measurement_limitations": "RSS is process-local; child processes and system-wide memory are excluded.",
        }


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run Git against an explicit repository root without altering Git config."""
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
        )
    except FileNotFoundError as exc:
        raise GitProvenanceError(
            "Git executable was not found; install Git and retry before starting P7B training."
        ) from exc


def _git_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "unknown Git failure").strip()
    lowered = detail.lower()
    if "dubious ownership" in lowered or "safe.directory" in lowered:
        kind = "Git ownership/safe-directory protection rejected this repository"
    elif "not a git repository" in lowered:
        kind = "resolved repository root is not a valid Git repository"
    else:
        kind = "Git could not resolve repository provenance"
    return f"{kind}: {detail}. Fix the repository/Git environment, then retry; P7B training was not started."


def capture_git_provenance(root: Path, *, required: bool) -> dict[str, Any]:
    """Capture commit and working-tree state; training requires a full SHA."""
    root = root.resolve()
    try:
        head_result = _git(root, ["rev-parse", "HEAD"])
        if head_result.returncode != 0:
            raise GitProvenanceError(_git_failure_message(head_result))
        head = head_result.stdout.strip().lower()
        if not _GIT_SHA_RE.fullmatch(head):
            raise GitProvenanceError(
                "Git HEAD did not resolve to a full 40-character commit SHA; "
                "P7B training was not started."
            )
        status_result = _git(root, ["status", "--porcelain=v1"])
        if status_result.returncode != 0:
            raise GitProvenanceError(_git_failure_message(status_result))
        entries = [line for line in status_result.stdout.splitlines() if line]
        return {
            "git_head": head,
            "working_tree": "dirty" if entries else "clean",
            "working_tree_details": {
                "is_dirty": bool(entries),
                "porcelain_v1": entries,
            },
        }
    except GitProvenanceError as exc:
        if required:
            raise
        return {
            "git_head": None,
            "working_tree": "unknown",
            "working_tree_details": {"is_dirty": None, "porcelain_v1": []},
            "provenance_error": str(exc),
        }


def _safe_exception(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc).replace("\\", "/")[:500]}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_portable_relative(path: object) -> bool:
    if not isinstance(path, str) or not path or "\\" in path:
        return False
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _expected_run_hash(plan: dict[str, Any]) -> str:
    payload = {
        "flags": P7B_FLAGS,
        "manifest_sha256": plan.get("manifest_sha256"),
        "fits": [
            {
                key: value
                for key, value in fit.items()
                if key not in {"artifact_path", "run_config_hash"}
            }
            for fit in plan.get("fits", [])
        ],
    }
    return sha256_canonical(payload)


def build_plan(manifest_path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Render the locked 60-fit plan; loading/splitting only, never fitting a model."""
    root = (repo_root or find_repo_root()).resolve()
    manifest = load_manifest(manifest_path, verify_lock=True)
    bound_hash = manifest_hash(manifest)
    pilot = manifest["pilot_budget"]
    if tuple(pilot["datasets"]) != EXPECTED_DATASETS or pilot["total_inner_fits"] != 60:
        raise P7BContractError(
            "P7A pilot budget is not the immutable P7B.1 60-fit contract."
        )
    if (
        P7B_FLAGS["candidate_selection"] != "none"
        or P7B_FLAGS["outer_selected_model_refit"]
    ):
        raise P7BContractError("P7B control flags are invalid.")
    fits: list[dict[str, Any]] = []
    protocol_config = manifest["preprocessing"]["protocol_config"]
    for dataset_id in EXPECTED_DATASETS:
        dataset = load_dataset(dataset_id, repo_root=root)
        source_file = _relative(dataset.source_path, root)
        checksum = get_dataset_checksum(dataset_id, source_file, repo_root=root)
        nested = create_nested_cv_definition(
            dataset,
            dataset_checksum=checksum.actual_sha256,
            # P7B.1 is explicitly restricted to repeat 0/fold 0.  Generating
            # the remaining paper repeats here would add no identities and is
            # deliberately avoided; P7C retains the dataset-specific counts.
            outer_n_repeats=1,
            outer_n_splits=2,
            inner_n_splits=5,
            random_seed=42,
        )
        outer = next(
            (
                item
                for item in nested.outer_folds
                if item.outer_fold_id == EXPECTED_OUTER
            ),
            None,
        )
        if (
            outer is None
            or outer.repeat_index != 0
            or outer.fold_index != 0
            or len(outer.inner_folds) != 5
        ):
            raise P7BContractError(
                f"{dataset_id}: required outer/inner partition is unavailable."
            )
        for candidate in pilot["candidates"]:
            params = {
                "max_depth": candidate["max_depth"],
                "min_samples_leaf": candidate["min_samples_leaf"],
            }
            for inner_index, inner in enumerate(outer.inner_folds):
                identity = {
                    "dataset_id": dataset_id,
                    "outer_fold_id": outer.outer_fold_id,
                    "candidate_id": candidate["id"],
                    "inner_fold_index": inner_index,
                    "derived_seed": inner.seed,
                    "manifest_version": manifest["schema_version"],
                    "manifest_sha256": bound_hash,
                }
                fit_id = sha256_canonical(identity)
                fits.append(
                    {
                        **identity,
                        "fit_id": fit_id,
                        "parameters": params,
                        "inner_fold_id": inner.inner_fold_id,
                        "inner_split_hash": inner.split_hash,
                        "dataset_checksum": checksum.actual_sha256,
                        "dataset_source": source_file,
                        "inner_training_rows": len(inner.train_indices),
                        "effective_min_samples_leaf": effective_min_samples_leaf(
                            params["min_samples_leaf"], len(inner.train_indices)
                        ),
                        "preprocessing_config": protocol_config,
                        "artifact_path": f"fits/{fit_id}/result.json",
                    }
                )
    config_payload = {
        "flags": P7B_FLAGS,
        "manifest_sha256": bound_hash,
        "fits": [
            {k: v for k, v in fit.items() if k not in {"artifact_path"}} for fit in fits
        ],
    }
    run_hash = sha256_canonical(config_payload)
    for fit in fits:
        fit["run_config_hash"] = run_hash
    plan = {
        "schema_version": 1,
        **P7B_FLAGS,
        "model_id": "decision_tree",
        "implementation": "sklearn.tree.DecisionTreeClassifier",
        "deviation": "c45_to_cart",
        "full_grid_id": "cart_a_grid_2",
        "pilot_subset_id": "cart_a_p7b_feasibility_subset",
        "manifest_path": _relative(manifest_path.resolve(), root),
        "manifest_sha256": bound_hash,
        "run_config_hash": run_hash,
        "fits": fits,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != 1:
        raise P7BContractError("Plan schema_version must be 1.")
    if {key: plan.get(key) for key in P7B_FLAGS} != P7B_FLAGS:
        raise P7BContractError("P7B engineering-only flags are missing or changed.")
    fits = plan.get("fits")
    if not isinstance(fits, list) or len(fits) != 60:
        raise P7BContractError("Plan must contain exactly 60 fits.")
    ids = [fit.get("fit_id") for fit in fits]
    if len(ids) != len(set(ids)):
        raise P7BContractError("Plan contains duplicate fit identity.")
    if {fit.get("dataset_id") for fit in fits} != set(EXPECTED_DATASETS):
        raise P7BContractError("Plan contains an unsupported dataset.")
    if any(fit.get("outer_fold_id") != EXPECTED_OUTER for fit in fits):
        raise P7BContractError("Plan contains an unsupported outer partition.")
    counts = Counter(fit["dataset_id"] for fit in fits)
    pairs = Counter((fit["dataset_id"], fit["candidate_id"]) for fit in fits)
    if set(counts.values()) != {20} or set(pairs.values()) != {5}:
        raise P7BContractError("Plan cardinalities do not match 3×4×5.")
    if any(not _is_portable_relative(fit.get("artifact_path")) for fit in fits):
        raise P7BContractError("Artifact paths must be portable relative paths.")
    if plan.get("run_config_hash") != _expected_run_hash(plan):
        raise P7BContractError(
            "Plan run_config_hash does not match its immutable plan content."
        )
    if not _SHA256_RE.fullmatch(str(plan.get("manifest_sha256", ""))):
        raise P7BContractError("Plan manifest_sha256 must be a SHA-256 digest.")
    return {
        "valid": True,
        "total_fits": 60,
        "unique_fit_ids": len(set(ids)),
        "per_dataset": dict(counts),
        "per_dataset_candidate": {f"{a}/{b}": n for (a, b), n in pairs.items()},
    }


def render_plan(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    provenance: dict[str, Any] | None = None,
    artifact_kind: str = "plan_only_dry_run",
) -> Path:
    """Render a plan-only artifact; missing Git provenance is allowed only here."""
    validate_plan(plan)
    root = (repo_root or find_repo_root()).resolve()
    provenance = provenance or capture_git_provenance(root, required=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "plan.json", plan)
    _write(output_dir / "validator.json", validate_plan(plan))
    _write(
        output_dir / "config_snapshot.json",
        {key: plan[key] for key in plan if key != "fits"},
    )
    _write(
        output_dir / "environment.json",
        {
            "schema_version": 1,
            "artifact_kind": artifact_kind,
            "publishable": False,
            "python": sys.version,
            "platform": platform.platform(),
            **provenance,
            "ram_method": {
                "kind": "process_rss",
                "unit": "bytes",
                "children_included": False,
                "sampling_interval_seconds": 0.05,
                "limitation": "RSS sampling is process-local; no child process accounting.",
            },
        },
    )
    return output_dir


def _read_json(
    path: Path, errors: list[dict[str, str]], label: str
) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append({"code": "missing_file", "path": label})
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append({"code": "invalid_json", "path": label, "detail": str(exc)})
        return None
    if not isinstance(value, dict):
        errors.append(
            {"code": "invalid_schema", "path": label, "detail": "JSON object required"}
        )
        return None
    return value


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
        and value < float("inf")
    )


def _validate_fit_payload(
    payload: dict[str, Any],
    fit: dict[str, Any],
    status: str,
    errors: list[dict[str, str]],
    path: str,
) -> None:
    if payload.get("status") != status:
        errors.append({"code": "fit_status", "path": path})
    for key in (
        "fit_id",
        "dataset_id",
        "candidate_id",
        "inner_fold_index",
        "derived_seed",
        "run_config_hash",
    ):
        if payload.get(key) != fit.get(key):
            errors.append(
                {"code": "fit_identity_mismatch", "path": path, "detail": key}
            )
    if payload.get("attempt_count") not in {1, 2}:
        errors.append({"code": "retry_contract", "path": path})
    if not _finite_nonnegative(payload.get("elapsed_wall_seconds")):
        errors.append({"code": "invalid_elapsed_time", "path": path})
    for key in (
        "process_rss_start_bytes",
        "process_rss_peak_bytes",
        "process_rss_delta_peak_bytes",
    ):
        if not isinstance(payload.get(key), int) or payload[key] < 0:
            errors.append({"code": "invalid_rss", "path": path, "detail": key})
    start, peak, delta = (
        payload.get(key)
        for key in (
            "process_rss_start_bytes",
            "process_rss_peak_bytes",
            "process_rss_delta_peak_bytes",
        )
    )
    if all(isinstance(value, int) for value in (start, peak, delta)) and (
        peak < start or delta != max(0, peak - start)
    ):
        errors.append({"code": "rss_invariant", "path": path})
    if status == "completed":
        for key in (
            "rows",
            "feature_count",
            "python_tracemalloc_peak_bytes",
            "artifact_bytes",
        ):
            if not isinstance(payload.get(key), int) or payload[key] < 0:
                errors.append(
                    {"code": "invalid_completed_telemetry", "path": path, "detail": key}
                )
    else:
        failure = payload.get("failure")
        if (
            not isinstance(failure, dict)
            or not isinstance(failure.get("type"), str)
            or not isinstance(failure.get("message"), str)
        ):
            errors.append({"code": "invalid_failure", "path": path})


def validate_artifacts(
    expected_plan: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate an on-disk plan-only or completed P7B run; never trust validator.json."""
    del repo_root  # The on-disk manifest hash is bound by the supplied current plan.
    errors: list[dict[str, str]] = []
    validate_plan(expected_plan)
    root = output_dir.resolve()
    plan = _read_json(root / "plan.json", errors, "plan.json")
    snapshot = _read_json(root / "config_snapshot.json", errors, "config_snapshot.json")
    environment = _read_json(root / "environment.json", errors, "environment.json")
    if plan is None or snapshot is None or environment is None:
        return {
            "valid": False,
            "validator_version": ARTIFACT_VALIDATOR_VERSION,
            "errors": errors,
        }
    try:
        validate_plan(plan)
    except P7BContractError as exc:
        errors.append({"code": "invalid_plan", "path": "plan.json", "detail": str(exc)})
    if plan != expected_plan:
        errors.append({"code": "plan_mismatch", "path": "plan.json"})
    expected_snapshot = {
        key: expected_plan[key] for key in expected_plan if key != "fits"
    }
    if snapshot != expected_snapshot:
        errors.append(
            {"code": "config_snapshot_mismatch", "path": "config_snapshot.json"}
        )
    if (
        environment.get("artifact_kind") == "plan_only_dry_run"
        or not (root / "engineering_summary.json").exists()
    ):
        # A plan-only artifact has no fits and may intentionally lack provenance in legacy dry-runs.
        fit_files = (
            list((root / "fits").rglob("*.json")) if (root / "fits").exists() else []
        )
        if fit_files:
            errors.append({"code": "plan_only_contains_fit_outputs", "path": "fits"})
        return {
            "valid": not errors,
            "validator_version": ARTIFACT_VALIDATOR_VERSION,
            "artifact_kind": "plan_only_dry_run",
            "training_artifacts_validated": False,
            "errors": errors,
            "validated_at": _now(),
        }
    head = environment.get("git_head")
    if not isinstance(head, str) or not _GIT_SHA_RE.fullmatch(head):
        errors.append({"code": "invalid_git_head", "path": "environment.json"})
    if environment.get("working_tree") not in {"clean", "dirty"} or not isinstance(
        environment.get("working_tree_details"), dict
    ):
        errors.append(
            {"code": "invalid_working_tree_provenance", "path": "environment.json"}
        )
    summary = _read_json(
        root / "engineering_summary.json", errors, "engineering_summary.json"
    )
    seen: set[str] = set()
    completed = failed = 0
    for fit in expected_plan["fits"]:
        fit_id = fit["fit_id"]
        if fit_id in seen:
            errors.append({"code": "duplicate_fit_identity", "path": "plan.json"})
            continue
        seen.add(fit_id)
        result_path = root / fit["artifact_path"]
        failure_path = result_path.with_name("failure.json")
        has_result, has_failure = result_path.is_file(), failure_path.is_file()
        if has_result and has_failure:
            errors.append({"code": "ambiguous_fit_state", "path": f"fits/{fit_id}"})
        elif has_result:
            payload = _read_json(result_path, errors, f"fits/{fit_id}/result.json")
            if payload is not None:
                _validate_fit_payload(
                    payload, fit, "completed", errors, f"fits/{fit_id}/result.json"
                )
            completed += 1
        elif has_failure:
            payload = _read_json(failure_path, errors, f"fits/{fit_id}/failure.json")
            if payload is not None:
                _validate_fit_payload(
                    payload, fit, "failed", errors, f"fits/{fit_id}/failure.json"
                )
            failed += 1
        else:
            errors.append({"code": "missing_fit_output", "path": f"fits/{fit_id}"})
    if summary is not None:
        pending = len(expected_plan["fits"]) - completed - failed
        expected_counts = {
            "planned": 60,
            "completed": completed,
            "failed": failed,
            "pending": pending,
        }
        if {key: summary.get(key) for key in expected_counts} != expected_counts:
            errors.append(
                {"code": "summary_count_mismatch", "path": "engineering_summary.json"}
            )
        expected_state = (
            "completed"
            if pending == 0 and failed == 0
            else "completed_with_failures"
            if pending == 0
            else "incomplete"
        )
        if summary.get("completion_status") != expected_state:
            errors.append(
                {
                    "code": "completion_status_mismatch",
                    "path": "engineering_summary.json",
                }
            )
    return {
        "valid": not errors,
        "validator_version": ARTIFACT_VALIDATOR_VERSION,
        "artifact_kind": "training_run",
        "training_artifacts_validated": True,
        "planned": 60,
        "completed": completed,
        "failed": failed,
        "errors": errors,
        "validated_at": _now(),
    }


def _completed(path: Path, fit: dict[str, Any], run_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return (
            saved.get("status") == "completed"
            and saved.get("fit_id") == fit["fit_id"]
            and saved.get("run_config_hash") == run_hash
        )
    except (OSError, json.JSONDecodeError):
        return False


def run(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    resume: bool = False,
    max_retry_attempts: int = 1,
) -> dict[str, Any]:
    """Execute only planned inner fits. No scoring, ranking, selection or outer refit exists here."""
    validate_plan(plan)
    root = (repo_root or find_repo_root()).resolve()
    # This happens before rendering or loading any dataset, hence before the first fit.
    provenance = capture_git_provenance(root, required=True)
    render_plan(
        plan,
        output_dir,
        repo_root=root,
        provenance=provenance,
        artifact_kind="training_run",
    )
    protocol = load_protocol_a_config(repo_root=root)
    datasets = {key: load_dataset(key, repo_root=root) for key in EXPECTED_DATASETS}
    completed = 0
    skipped = 0
    failed = 0
    for fit in plan["fits"]:
        result_path = output_dir / fit["artifact_path"]
        if resume and _completed(result_path, fit, plan["run_config_hash"]):
            skipped += 1
            continue
        attempts = 0
        while True:
            attempts += 1
            started = time.perf_counter()
            stamp = _now()
            sampler = ProcessRssSampler()
            sampler.start()
            fit_identity = {
                key: fit[key]
                for key in (
                    "fit_id",
                    "dataset_id",
                    "candidate_id",
                    "inner_fold_index",
                    "derived_seed",
                )
            }
            try:
                tracemalloc.start()
                dataset = datasets[fit["dataset_id"]]
                nested = create_nested_cv_definition(
                    dataset,
                    dataset_checksum=fit["dataset_checksum"],
                    outer_n_repeats=1,
                    outer_n_splits=2,
                    inner_n_splits=5,
                    random_seed=42,
                )
                outer = next(
                    item
                    for item in nested.outer_folds
                    if item.outer_fold_id == EXPECTED_OUTER
                )
                inner = outer.inner_folds[fit["inner_fold_index"]]
                _, X_train, _ = _fit_preprocessing(
                    dataset,
                    train_indices=inner.train_indices,
                    transform_indices=inner.validation_indices,
                    protocol_config=protocol,
                )
                create_model(
                    "decision_tree", fit["parameters"], random_seed=fit["derived_seed"]
                ).fit(X_train, dataset.target.iloc[list(inner.train_indices)])
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rss = sampler.stop()
                payload = {
                    **P7B_FLAGS,
                    **fit_identity,
                    "run_config_hash": plan["run_config_hash"],
                    "status": "completed",
                    "attempt_count": attempts,
                    "started_at": stamp,
                    "ended_at": _now(),
                    "elapsed_wall_seconds": time.perf_counter() - started,
                    "rows": len(X_train),
                    "feature_count": X_train.shape[1],
                    "configured_min_samples_leaf": fit["parameters"][
                        "min_samples_leaf"
                    ],
                    "effective_min_samples_leaf": fit["effective_min_samples_leaf"],
                    **rss,
                    "python_tracemalloc_peak_bytes": int(peak),
                    "artifact_bytes": 0,
                }
                _write(result_path, payload)
                completed += 1
                break
            except Exception as exc:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                rss = sampler.stop()
                failure = {
                    **P7B_FLAGS,
                    **fit_identity,
                    "run_config_hash": plan["run_config_hash"],
                    "status": "failed",
                    "attempt_count": attempts,
                    "started_at": stamp,
                    "ended_at": _now(),
                    "elapsed_wall_seconds": time.perf_counter() - started,
                    **rss,
                    "failure": _safe_exception(exc),
                }
                _write(output_dir / "fits" / fit["fit_id"] / "failure.json", failure)
                if (
                    isinstance(exc, (OSError, TimeoutError))
                    and attempts <= max_retry_attempts
                ):
                    continue
                failed += 1
                break
    summary = {
        **P7B_FLAGS,
        "planned": len(plan["fits"]),
        "completed": completed + skipped,
        "skipped_on_resume": skipped,
        "failed": failed,
        "pending": len(plan["fits"]) - completed - skipped - failed,
        "completion_status": (
            "completed" if failed == 0 else "completed_with_failures"
        ),
        "run_config_hash": plan["run_config_hash"],
    }
    _write(output_dir / "engineering_summary.json", summary)
    return summary
