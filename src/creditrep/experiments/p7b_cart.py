"""P7B.1 CART feasibility runner: inner-evaluation-only, never scientific selection."""

from __future__ import annotations

import json
import platform
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


class P7BContractError(ValueError):
    """Raised when an immutable P7B.1 contract is violated."""


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


def _git(root: Path, args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _safe_exception(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc).replace("\\", "/")[:500]}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


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
    if any(
        Path(fit["artifact_path"]).is_absolute() or "\\" in fit["artifact_path"]
        for fit in fits
    ):
        raise P7BContractError("Artifact paths must be portable relative paths.")
    return {
        "valid": True,
        "total_fits": 60,
        "unique_fit_ids": len(set(ids)),
        "per_dataset": dict(counts),
        "per_dataset_candidate": {f"{a}/{b}": n for (a, b), n in pairs.items()},
    }


def render_plan(plan: dict[str, Any], output_dir: Path) -> Path:
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
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": _git(find_repo_root(), ["rev-parse", "HEAD"]),
            "working_tree": "dirty"
            if _git(find_repo_root(), ["status", "--porcelain"])
            else "clean",
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
    render_plan(plan, output_dir)
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
                    "fit_id": fit["fit_id"],
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
                    "fit_id": fit["fit_id"],
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
        "completed": completed,
        "skipped_on_resume": skipped,
        "failed": failed,
        "run_config_hash": plan["run_config_hash"],
    }
    _write(output_dir / "engineering_summary.json", summary)
    return summary
