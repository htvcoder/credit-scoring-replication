"""Read-only telemetry analysis for a completed P7C.2 feasibility run.

This module deliberately has no imports from estimator, preprocessing, or runner
code.  It only reads the finalized ``result.json`` artifacts and emits a stable,
machine-readable summary for the accompanying decision record.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any


PLAN_DIGEST = "1f3a6cd5b9f4d766fe89b34676ba66cf3ec731b49b27ff3769af042d83f08516"
EXPECTED = {
    (model, dataset, candidate, fold)
    for model, candidates in {
        "random_forest": ("rf_low", "rf_medium", "rf_high"),
        "xgboost": ("xgb_low", "xgb_medium", "xgb_high"),
    }.items()
    for dataset in ("AC", "GMC")
    for candidate in candidates
    for fold in range(5)
}
REQUIRED = {
    "fit_id",
    "plan_digest",
    "model_id",
    "dataset_id",
    "candidate_id",
    "inner_fold_index",
    "started_at",
    "completed_at",
    "duration_seconds",
    "process_cpu_seconds",
    "process_rss_start_bytes",
    "process_rss_peak_bytes",
    "process_rss_delta_peak_bytes",
    "configured_thread_count",
    "effective_thread_count",
    "git_provenance",
}


def percentile(values: list[float], q: float) -> float:
    """Return the nearest-rank percentile; q must be in [0, 1]."""
    if not values or not 0 <= q <= 1:
        raise ValueError("percentile needs non-empty values and q in [0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _stat(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "p90": None,
            "sum": 0.0,
            "cv": None,
        }
    mean = fmean(values)
    variance = fmean([(item - mean) ** 2 for item in values])
    return {
        "n": len(values),
        "min": min(values),
        "median": median(values),
        "mean": mean,
        "max": max(values),
        "p90": percentile(values, 0.9),
        "sum": sum(values),
        "cv": math.sqrt(variance) / mean if mean else None,
    }


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Read a finalized run and return its deterministic aggregation."""
    paths = sorted((run_dir / "fits").glob("*/result.json"))
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON: {path}: {exc}")
            continue
        missing = REQUIRED - set(payload)
        if missing:
            errors.append(f"missing required fields: {path}: {sorted(missing)}")
            continue
        payloads.append(payload)

    identities = [
        (
            p.get("model_id"),
            p.get("dataset_id"),
            p.get("candidate_id"),
            p.get("inner_fold_index"),
        )
        for p in payloads
    ]
    identity_set = set(identities)
    duplicate_count = len(identities) - len(identity_set)
    unexpected = sorted(identity_set - EXPECTED)
    missing = sorted(EXPECTED - identity_set)
    if len(payloads) != 60:
        errors.append(f"expected 60 artifacts, found {len(payloads)}")
    if duplicate_count or unexpected or missing:
        errors.append("fit identities do not exactly match immutable plan")

    digests = sorted({p.get("plan_digest") for p in payloads})
    heads = sorted({p.get("git_provenance", {}).get("git_head") for p in payloads})
    clean_states = sorted(
        {p.get("git_provenance", {}).get("working_tree") for p in payloads}
    )
    thread_pairs = sorted(
        {
            (p.get("configured_thread_count"), p.get("effective_thread_count"))
            for p in payloads
        }
    )
    tree_methods = sorted(
        {p.get("tree_method") for p in payloads if p.get("model_id") == "xgboost"}
    )
    if digests != [PLAN_DIGEST]:
        errors.append(f"unexpected plan digests: {digests}")
    if len(heads) != 1 or clean_states != ["clean"]:
        errors.append("inconsistent or non-clean Git provenance")
    if thread_pairs != [(1, 1)] or tree_methods != ["hist"]:
        errors.append("threading or XGBoost tree method policy mismatch")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        groups[
            (payload["model_id"], payload["dataset_id"], payload["candidate_id"])
        ].append(payload)
    table = []
    for key in sorted(groups):
        rows = groups[key]
        duration = [float(row["duration_seconds"]) for row in rows]
        cpu = [float(row["process_cpu_seconds"]) for row in rows]
        rss_start = [float(row["process_rss_start_bytes"]) for row in rows]
        rss_peak = [float(row["process_rss_peak_bytes"]) for row in rows]
        rss_delta = [float(row["process_rss_delta_peak_bytes"]) for row in rows]
        table.append(
            {
                "model_id": key[0],
                "dataset_id": key[1],
                "candidate_id": key[2],
                "folds": sorted(row["inner_fold_index"] for row in rows),
                "duration_seconds": _stat(duration),
                "cpu_seconds": _stat(cpu),
                "cpu_to_wall_mean": fmean(cpu) / fmean(duration),
                "rss_start_bytes": _stat(rss_start),
                "rss_peak_bytes": _stat(rss_peak),
                "rss_delta_bytes": _stat(rss_delta),
                "invalid_rss_count": sum(
                    x < 0 for x in rss_start + rss_peak + rss_delta
                ),
            }
        )

    ordered = sorted(payloads, key=lambda p: _timestamp(p["started_at"]))
    starts = [_timestamp(p["started_at"]) for p in ordered]
    ends = [_timestamp(p["completed_at"]) for p in ordered]
    gaps = [(starts[i] - ends[i - 1]).total_seconds() for i in range(1, len(starts))]
    overlap_count = sum(gap < -0.001 for gap in gaps)
    all_duration = [float(p["duration_seconds"]) for p in payloads]
    observed = (max(ends) - min(starts)).total_seconds() if starts else 0.0
    return {
        "valid": not errors,
        "errors": errors,
        "artifact_integrity": {
            "result_files": len(paths),
            "completed": len(payloads),
            "failed": 0,
            "missing": len(missing),
            "unexpected": len(unexpected),
            "duplicates": duplicate_count,
            "temporary": len(list(run_dir.rglob("*.tmp"))),
            "corrupt": sum(e.startswith("invalid JSON") for e in errors),
            "plan_digests": digests,
            "git_heads": heads,
            "working_tree_states": clean_states,
            "thread_pairs": thread_pairs,
            "xgboost_tree_methods": tree_methods,
        },
        "groups": table,
        "execution": {
            "first_started_at": min((p["started_at"] for p in payloads), default=None),
            "last_completed_at": max(
                (p["completed_at"] for p in payloads), default=None
            ),
            "observed_elapsed_seconds": observed,
            "summed_fit_duration_seconds": sum(all_duration),
            "between_fit_overhead_seconds": sum(gaps),
            "overlap_count": overlap_count,
            "minimum_gap_seconds": min(gaps) if gaps else None,
            "maximum_gap_seconds": max(gaps) if gaps else None,
            "all_fit_duration_seconds": _stat(all_duration),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_run(args.run_dir), ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
