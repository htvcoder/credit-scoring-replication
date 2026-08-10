"""Pure contracts for P7C.4B.2b. No function in this module starts a fit."""

from __future__ import annotations

from copy import deepcopy
from statistics import mean, median, pstdev
from typing import Any

from creditrep.config.loader import sha256_canonical

SCIENTIFIC_DIGEST = "4d8636c3606e07e243efd2bc7be12806e7adf4fc1b19dbe0dc113a35adc57f75"
MODES = {"cpu_parallel_1": 1, "cpu_parallel_2": 2}
CANONICAL_TOTAL_FITS = 54_270
SCHEMA_VERSION = 3
DATASET_FINGERPRINTS = {
    "TC": "30C6BE3ABD8DCFD3E6096C828BAD8C2F011238620F5369220BD60CFC82700933",
    "GMC": "1BD46DA486A5708C58C7B01A034FAE2A13B327F6F7B62EA7BA4FE3B5824B24AC",
}


class PreflightError(ValueError):
    """Stable planning/validation failure."""


def plan_digest(plan: dict[str, Any]) -> str:
    value = deepcopy(plan)
    value.pop("plan_digest", None)
    return sha256_canonical(value)


def machine_profile_digest(profile: dict[str, Any]) -> str:
    value = deepcopy(profile)
    value.pop("profile_digest", None)
    return sha256_canonical(value)


def _complexity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sum(candidate["hidden_units"]),
        max(candidate["hidden_units"]),
        candidate["dropout"],
        candidate["l2"],
        -float(candidate["learning_rate"]),
        candidate["candidate_id"],
    )


def _coverage_candidates(
    candidates: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    ordered = sorted(candidates, key=_complexity)
    return [
        ("light", ordered[0]),
        ("median", ordered[len(ordered) // 2]),
        ("heavy", ordered[-1]),
    ]


def build_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["lock"]["manifest_sha256"] != SCIENTIFIC_DIGEST:
        raise PreflightError("scientific_manifest_digest_mismatch")
    models = {item["id"]: item["candidates"] for item in manifest["models"]}
    units: list[dict[str, Any]] = []
    for dataset_id in ("TC", "GMC"):
        for model_id in ("mlp_1", "mlp_3", "mlp_5"):
            for role, candidate in _coverage_candidates(models[model_id]):
                identity = {
                    "scientific_manifest_digest": SCIENTIFIC_DIGEST,
                    "dataset_id": dataset_id,
                    "dataset_fingerprint": DATASET_FINGERPRINTS[dataset_id],
                    "model_id": model_id,
                    "candidate_id": candidate["candidate_id"],
                    "candidate": {
                        key: candidate[key]
                        for key in (
                            "hidden_units",
                            "dropout",
                            "l2",
                            "learning_rate",
                            "batch_normalization",
                        )
                    },
                    "inner_fold": 0,
                    "seed_identity": 1701,
                    "coverage_role": role,
                }
                units.append(
                    {**identity, "logical_unit_id": sha256_canonical(identity)}
                )
    tasks: list[dict[str, Any]] = []
    for mode in MODES:
        # A separated measured phase makes its wall-clock interval identifiable.
        # Legacy schema-v2 plans interleaved these classifications and remain
        # readable, but cannot support an orchestration-overhead estimate.
        for classification, repetitions in (("warmup", 1), ("measured", 2)):
            for unit in units:
                for repetition in range(repetitions):
                    identity = {
                        "logical_unit_id": unit["logical_unit_id"],
                        "mode": mode,
                        "classification": classification,
                        "repetition": repetition,
                        "seed_identity": 1601
                        if classification == "warmup"
                        else 1701 + repetition,
                    }
                    tasks.append(
                        {**unit, **identity, "task_id": sha256_canonical(identity)}
                    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": "P7C.4B.2b",
        "evidence_scope": "engineering_compute_preflight_non_publishable",
        "scientific_manifest_digest": SCIENTIFIC_DIGEST,
        "machine_role_required": "intended_single_vm_target",
        "modes": MODES,
        "selection": "deterministic_workload_driver_no_predictive_metric",
        "timing_scope_policy": {
            "execution_order": "all_warmups_before_measured",
            "projection_input": "successful_measured_fit_intervals_only",
            "warmups": "reported_separately_never_projected",
        },
        "thread_policy": {
            "max_workers": 2,
            "threads_per_worker": 2,
            "nested_parallelism": "forbidden",
        },
        "limits": {
            "per_fit_timeout_seconds": {
                "value": 1800,
                "class": "invariant_hard_safety_limit",
            },
            "global_wall_clock_seconds": {
                "value": 43200,
                "class": "configurable_target_parameter",
            },
            "minimum_free_disk_bytes": {
                "value": 17179869184,
                "class": "configurable_target_parameter",
            },
            "aggregate_process_tree_rss_bytes": {
                "value": 12348030976,
                "class": "configurable_target_parameter",
            },
            "minimum_system_available_ram_bytes": {
                "value": 2147483648,
                "class": "derived_from_target_machine_profile",
            },
            "artifact_size_bytes": {
                "value": 2147483648,
                "class": "invariant_hard_safety_limit",
            },
            "retry_maximum": {"value": 1, "class": "invariant_hard_safety_limit"},
        },
        "units": units,
        "tasks": tasks,
        "fit_budget": {
            "warmups_per_mode": 18,
            "measured_per_mode": 36,
            "no_retry_total": 108,
            "retry_maximum_per_task": 1,
            "worst_case_attempts": 216,
        },
        "runtime_cost_status": "pending_target_measurement_no_b1_fixture_input",
    }
    plan["plan_digest"] = plan_digest(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    codes: list[str] = []
    if plan.get("scientific_manifest_digest") != SCIENTIFIC_DIGEST:
        codes.append("scientific_manifest_digest_mismatch")
    if plan.get("plan_digest") != plan_digest(plan):
        codes.append("plan_digest_mismatch")
    if plan.get("modes") != MODES or plan.get("thread_policy") != {
        "max_workers": 2,
        "threads_per_worker": 2,
        "nested_parallelism": "forbidden",
    }:
        codes.append("worker_limit_mismatch")
    if plan.get("timing_scope_policy") != {
        "execution_order": "all_warmups_before_measured",
        "projection_input": "successful_measured_fit_intervals_only",
        "warmups": "reported_separately_never_projected",
    }:
        codes.append("timing_scope_policy_mismatch")
    units, tasks = plan.get("units", []), plan.get("tasks", [])
    if len(units) != 18 or len({x.get("logical_unit_id") for x in units}) != 18:
        codes.append("preflight_unit_coverage_mismatch")
    if len(tasks) != 108 or len({x.get("task_id") for x in tasks}) != 108:
        codes.append("task_coverage_mismatch")
    for task in tasks:
        identity = {
            key: task[key]
            for key in (
                "logical_unit_id",
                "mode",
                "classification",
                "repetition",
                "seed_identity",
            )
        }
        if task.get("task_id") != sha256_canonical(identity):
            codes.append("task_identity_mismatch")
            break
    measured = {
        mode: [
            x
            for x in tasks
            if x.get("mode") == mode and x.get("classification") == "measured"
        ]
        for mode in MODES
    }
    if any(len(value) != 36 for value in measured.values()) or {
        x["logical_unit_id"] for x in measured["cpu_parallel_1"]
    } != {x["logical_unit_id"] for x in measured["cpu_parallel_2"]}:
        codes.append("measured_workload_mismatch")
    if codes:
        raise PreflightError(",".join(sorted(set(codes))))
    return {
        "valid": True,
        "units": 18,
        "measured_fits_per_mode": 36,
        "warmups_per_mode": 18,
        **plan["fit_budget"],
    }


def validate_machine(profile: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    required = {
        "machine_role",
        "machine_id",
        "cloud_provider",
        "instance_type",
        "os",
        "cpu_model",
        "physical_cores",
        "logical_cores",
        "ram_total_bytes",
        "disk_free_bytes",
        "python_executable",
        "python_version",
        "dependency_fingerprint",
        "git_commit",
        "scientific_manifest_digest",
        "dataset_fingerprints",
        "worker_limit",
        "threads_per_worker",
        "utc_captured",
        "profile_digest",
    }
    codes = []
    if not required <= set(profile):
        codes.append("machine_provenance_missing")
    if profile.get("profile_digest") != machine_profile_digest(profile):
        codes.append("machine_profile_digest_mismatch")
    if profile.get("machine_role") != "intended_single_vm_target":
        codes.append("target_machine_not_confirmed")
    if profile.get("machine_role") == "intended_single_vm_target" and (
        profile.get("cloud_provider") == "operator_unspecified"
        or profile.get("instance_type") == "operator_unspecified"
    ):
        codes.append("target_provider_identity_missing")
    if profile.get("scientific_manifest_digest") != SCIENTIFIC_DIGEST:
        codes.append("scientific_manifest_digest_mismatch")
    if profile.get("dataset_fingerprints") != DATASET_FINGERPRINTS:
        codes.append("dataset_fingerprint_mismatch")
    if profile.get("worker_limit") != 2 or profile.get("threads_per_worker") != 2:
        codes.append("worker_limit_mismatch")
    limits = plan["limits"]
    if profile.get("disk_free_bytes", 0) < limits["minimum_free_disk_bytes"]["value"]:
        codes.append("insufficient_free_disk")
    aggregate = limits["aggregate_process_tree_rss_bytes"]["value"]
    reserve = limits["minimum_system_available_ram_bytes"]["value"]
    if profile.get("ram_total_bytes", 0) < aggregate + reserve:
        codes.append("target_ram_incompatible")
    if codes:
        raise PreflightError(",".join(sorted(set(codes))))
    return {"valid": True, "profile_digest": profile["profile_digest"]}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _normalized_records(
    records: list[dict[str, Any]], mode: str
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "mode": record.get("mode", mode),
            "status": record.get("status", "completed"),
            "aggregate_process_tree_peak_rss_bytes": record.get(
                "aggregate_process_tree_peak_rss_bytes", 0
            ),
            "system_available_ram_min_bytes": record.get(
                "system_available_ram_min_bytes", 0
            ),
        }
        for record in records
    ]


def _interval(record: dict[str, Any]) -> tuple[float, float] | None:
    start = record.get("started_monotonic")
    end = record.get("completed_monotonic")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    if end < start:
        raise PreflightError("invalid_or_missing_measured_telemetry")
    return float(start), float(end)


def _interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def summarize_legacy_v1(
    records: list[dict[str, Any]], *, mode: str = "cpu_parallel_1"
) -> dict[str, Any]:
    """Rebuild schema-v1 summaries so immutable legacy artifacts stay readable."""
    records = [
        (
            {
                **x,
                "mode": x.get("mode", mode),
                "status": x.get("status", "completed"),
                "started_monotonic": x.get("started_monotonic", 0),
                "completed_monotonic": x.get(
                    "completed_monotonic", x.get("wall_clock_seconds", 0)
                ),
                "aggregate_process_tree_peak_rss_bytes": x.get(
                    "aggregate_process_tree_peak_rss_bytes", 0
                ),
                "system_available_ram_min_bytes": x.get(
                    "system_available_ram_min_bytes", 0
                ),
            }
        )
        for x in records
    ]
    measured = [
        x
        for x in records
        if x.get("classification") == "measured" and x.get("mode") == mode
    ]
    times = [
        float(x["wall_clock_seconds"])
        for x in measured
        if x.get("status") == "completed"
    ]
    if not measured or any(value < 0 for value in times):
        raise PreflightError("invalid_or_missing_measured_telemetry")
    elapsed = max((x["completed_monotonic"] for x in measured), default=0) - min(
        (x["started_monotonic"] for x in measured), default=0
    )
    sufficient = len(times) >= 20
    cpu_seconds = sum(
        float(x.get("cpu_time_seconds", 0))
        for x in measured
        if x.get("status") == "completed"
    )
    workers = MODES.get(mode, 1)
    parallel_efficiency = (sum(times) / (workers * elapsed)) if elapsed > 0 else None
    return {
        "schema_version": 1,
        "mode": mode,
        "measured_count": len(measured),
        "successful": len(times),
        "failed": sum(x.get("status") == "failed" for x in measured),
        "timed_out": sum(x.get("reason_code") == "fit_timeout" for x in measured),
        "mean_seconds": mean(times) if times else None,
        "median_seconds": median(times) if times else None,
        "stddev_seconds": pstdev(times) if len(times) > 1 else None,
        "min_seconds": min(times) if times else None,
        "max_seconds": max(times) if times else None,
        "p50_seconds": percentile(times, 0.5) if sufficient else "insufficient_sample",
        "p90_seconds": percentile(times, 0.9) if sufficient else "insufficient_sample",
        "p95_seconds": percentile(times, 0.95) if sufficient else "insufficient_sample",
        "total_compute_seconds": sum(times),
        "elapsed_wall_clock_seconds": max(elapsed, 0),
        "throughput_per_hour": (len(times) * 3600 / elapsed) if elapsed > 0 else None,
        "total_cpu_seconds": cpu_seconds,
        "mean_cpu_utilization_percent": (cpu_seconds / sum(times) * 100)
        if sum(times) > 0
        else None,
        "observed_parallel_efficiency": min(1.0, parallel_efficiency)
        if parallel_efficiency is not None
        else None,
        "scheduler_overhead_seconds": max(0, elapsed - sum(times) / workers),
        "peak_aggregate_rss_bytes": max(
            (x.get("aggregate_process_tree_peak_rss_bytes", 0) for x in measured),
            default=0,
        ),
        "minimum_system_available_ram_bytes": min(
            (x.get("system_available_ram_min_bytes", 0) for x in measured), default=0
        ),
        "warmups_excluded": sum(
            x.get("classification") == "warmup" and x.get("mode") == mode
            for x in records
        ),
    }


def summarize(
    records: list[dict[str, Any]], *, mode: str = "cpu_parallel_1"
) -> dict[str, Any]:
    """Summarize warmup and measured scopes without treating warmup as overhead."""
    records = _normalized_records(records, mode)
    measured = [
        record
        for record in records
        if record.get("classification") == "measured" and record.get("mode") == mode
    ]
    successful = [record for record in measured if record.get("status") == "completed"]
    times = [float(record["wall_clock_seconds"]) for record in successful]
    if (
        not measured
        or len(times) != len(successful)
        or any(value < 0 for value in times)
    ):
        raise PreflightError("invalid_or_missing_measured_telemetry")

    warmups = [
        record
        for record in records
        if record.get("classification") == "warmup" and record.get("mode") == mode
    ]
    measured_intervals = [_interval(record) for record in successful]
    warmup_intervals = [_interval(record) for record in warmups]
    timestamps_available = bool(successful) and all(
        interval is not None for interval in measured_intervals
    )
    measured_phase_elapsed = None
    measured_active_union = None
    measured_idle_gap = None
    occupancy = None
    capacity_loss = None
    backcheck = {"status": "unavailable_missing_timestamps"}
    phase_status = "timestamps_unavailable"
    if timestamps_available:
        concrete_measured = [interval for interval in measured_intervals if interval]
        first_measured = min(start for start, _ in concrete_measured)
        last_measured = max(end for _, end in concrete_measured)
        measured_phase_elapsed = last_measured - first_measured
        measured_active_union = _interval_union_seconds(concrete_measured)
        measured_idle_gap = measured_phase_elapsed - measured_active_union
        if warmups and all(interval is not None for interval in warmup_intervals):
            last_warmup = max(
                end
                for interval in warmup_intervals
                if interval
                for _, end in [interval]
            )
            phase_status = (
                "separated_before_measured"
                if last_warmup <= first_measured
                else "interleaved_with_measured"
            )
        elif not warmups:
            phase_status = "no_warmups_present"
        workers = MODES.get(mode, 1)
        aggregate = sum(times)
        if measured_phase_elapsed > 0:
            occupancy = min(1.0, aggregate / (workers * measured_phase_elapsed))
            capacity_loss = max(0.0, measured_phase_elapsed - aggregate / workers)
            backcheck = {
                "status": (
                    "reproduced_separated_measured_scope"
                    if phase_status
                    in {"separated_before_measured", "no_warmups_present"}
                    else "descriptive_only_interleaved_warmup_scope"
                ),
                "work_conserving_seconds": aggregate / workers,
                "observed_elapsed_over_ideal_capacity_seconds": capacity_loss,
                "reconstructed_elapsed_seconds": aggregate / workers + capacity_loss,
                "observed_measured_phase_elapsed_seconds": measured_phase_elapsed,
            }

    sufficient = len(times) >= 20
    cpu_seconds = sum(float(record.get("cpu_time_seconds", 0)) for record in successful)
    warmup_times = [
        float(record["wall_clock_seconds"])
        for record in warmups
        if record.get("status") == "completed"
    ]
    overhead_status = (
        "bounded_measured_scope_observed_not_extrapolated"
        if phase_status in {"separated_before_measured", "no_warmups_present"}
        else "unknown_interleaved_warmup_measurement_scope"
        if phase_status == "interleaved_with_measured"
        else "unknown_missing_timestamps"
    )
    return {
        "schema_version": 2,
        "mode": mode,
        "timing_scope": "successful_measured_fits_only_warmups_reported_separately",
        "measurement_phase_status": phase_status,
        "measured_count": len(measured),
        "successful": len(times),
        "failed": sum(record.get("status") == "failed" for record in measured),
        "timed_out": sum(
            record.get("reason_code") == "fit_timeout" for record in measured
        ),
        "mean_seconds": mean(times) if times else None,
        "median_seconds": median(times) if times else None,
        "stddev_seconds": pstdev(times) if len(times) > 1 else None,
        "min_seconds": min(times) if times else None,
        "max_seconds": max(times) if times else None,
        "p50_seconds": percentile(times, 0.5) if sufficient else "insufficient_sample",
        "p90_seconds": percentile(times, 0.9) if sufficient else "insufficient_sample",
        "p95_seconds": percentile(times, 0.95) if sufficient else "insufficient_sample",
        "aggregate_measured_fit_runtime_seconds": sum(times),
        "measured_phase_elapsed_seconds": measured_phase_elapsed,
        "measured_active_union_seconds": measured_active_union,
        "measured_idle_gap_seconds": measured_idle_gap,
        "observed_worker_occupancy": occupancy,
        "observed_elapsed_over_ideal_capacity_seconds": capacity_loss,
        "orchestration_overhead_status": overhead_status,
        "bounded_workload_backcheck": backcheck,
        "throughput_per_hour": (
            len(times) * 3600 / measured_phase_elapsed
            if measured_phase_elapsed and phase_status != "interleaved_with_measured"
            else None
        ),
        "total_cpu_seconds": cpu_seconds,
        "mean_cpu_utilization_percent": (cpu_seconds / sum(times) * 100)
        if sum(times) > 0
        else None,
        "peak_aggregate_rss_bytes": max(
            (
                record.get("aggregate_process_tree_peak_rss_bytes", 0)
                for record in measured
            ),
            default=0,
        ),
        "minimum_system_available_ram_bytes": min(
            (record.get("system_available_ram_min_bytes", 0) for record in measured),
            default=0,
        ),
        "warmup": {
            "count": len(warmups),
            "successful": len(warmup_times),
            "aggregate_fit_runtime_seconds": sum(warmup_times),
            "excluded_from_projection": True,
        },
    }


def project(
    records: list[dict[str, Any]] | dict[str, Any],
    *,
    evidence_scope: str = "development_fixture_non_benchmark",
    price: dict[str, Any] | None = None,
    price_per_hour: float | None = None,
    two_vm_efficiency: float | None = None,
) -> dict[str, Any]:
    if evidence_scope != "target_single_vm_measured":
        return {
            "status": "pending_target_measurement",
            "single_vm_parallel_1": {"status": "pending"},
            "single_vm_parallel_2": {"status": "pending"},
            "two_vm_cpu": {
                "status": "pending_multi_vm_overhead_evidence_not_authorized"
            },
            "gpu": {"status": "pending_gpu_preflight"},
            "cost": {"status": "pending_operator_price_input"},
        }
    required = {
        (d, m, c, mode)
        for d in ("TC", "GMC")
        for m in ("mlp_1", "mlp_3", "mlp_5")
        for c in ("light", "median", "heavy")
        for mode in MODES
    }
    observed = {
        (x["dataset_id"], x["model_id"], x["coverage_role"], x["mode"])
        for x in records
        if x.get("classification") == "measured" and x.get("status") == "completed"
    }
    if not required <= observed:
        return {
            "status": "insufficient_stratified_coverage",
            "missing_strata": len(required - observed),
            "gpu": {"status": "pending_gpu_preflight"},
            "cost": {"status": "pending_operator_price_input"},
        }
    repetitions = {
        key: sum(
            x.get("classification") == "measured"
            and x.get("status") == "completed"
            and (x["dataset_id"], x["model_id"], x["coverage_role"], x["mode"]) == key
            for x in records
        )
        for key in required
    }
    if any(count < 2 for count in repetitions.values()):
        return {
            "status": "insufficient_stratified_repetitions",
            "strata_below_two_repetitions": sum(
                count < 2 for count in repetitions.values()
            ),
            "execution_plan_eligible": False,
            "gpu": {"status": "pending_gpu_preflight"},
            "cost": {"status": "pending_operator_price_input"},
        }
    model_inner = {"mlp_1": 10800, "mlp_3": 21600, "mlp_5": 21600}
    estimates = {}
    for mode, workers in MODES.items():
        cells = {}
        for dataset in ("TC", "GMC"):
            for model in model_inner:
                role_means = []
                for role in ("light", "median", "heavy"):
                    values = [
                        float(x["wall_clock_seconds"])
                        for x in records
                        if x.get("dataset_id") == dataset
                        and x.get("model_id") == model
                        and x.get("coverage_role") == role
                        and x.get("mode") == mode
                        and x.get("classification") == "measured"
                        and x.get("status") == "completed"
                    ]
                    role_means.append(mean(values))
                cells[(dataset, model)] = mean(role_means)
        low = sum(model_inner[m] * cells[("TC", m)] for m in model_inner)
        high = sum(model_inner[m] * cells[("GMC", m)] for m in model_inner)
        point = (low + high) / 2
        mode_records = [x for x in records if x.get("mode") == mode]
        summary = summarize(mode_records, mode=mode)
        estimates[mode] = {
            "status": "derived_inner_fit_only_incomplete_total_elapsed",
            "measured_input": {
                "strata": 18,
                "repetitions_per_stratum": 2,
                "workers": workers,
                "aggregate_measured_fit_runtime_seconds": summary[
                    "aggregate_measured_fit_runtime_seconds"
                ],
                "measured_phase_elapsed_seconds": summary[
                    "measured_phase_elapsed_seconds"
                ],
                "measurement_phase_status": summary["measurement_phase_status"],
                "observed_worker_occupancy": summary["observed_worker_occupancy"],
            },
            "inner_fit_projection": {
                "aggregate_fit_runtime_hours": {
                    "point": point / 3600,
                    "tc_gmc_range": [low / 3600, high / 3600],
                },
                "conditional_work_conserving_elapsed_hours": {
                    "point": point / (3600 * workers),
                    "tc_gmc_range": [
                        low / (3600 * workers),
                        high / (3600 * workers),
                    ],
                    "formula": "aggregate_fit_runtime_hours / workers",
                    "contention_handling": (
                        "already_observed_in_mode_specific_per_fit_duration_"
                        "no_second_efficiency_penalty"
                    ),
                },
            },
            "bounded_workload_backcheck": summary["bounded_workload_backcheck"],
            "orchestration_overhead": {
                "status": summary["orchestration_overhead_status"],
                "bounded_observed_elapsed_over_ideal_capacity_seconds": summary[
                    "observed_elapsed_over_ideal_capacity_seconds"
                ],
                "canonical_extrapolation": "unknown_not_extrapolated",
            },
            "outer_refits": {
                "count": 270,
                "status": "unknown_unmeasured",
                "elapsed_hours": None,
            },
            "total_canonical_elapsed": {
                "status": "unknown_incomplete_outer_refits_and_overhead",
                "projected_elapsed_hours": None,
            },
            "assumption": [
                "TC/GMC bound four unmeasured datasets",
                "candidate complexity terciles equally represent each model grid",
                "canonical inner-fit scheduler is work-conserving",
            ],
            "unknown": [
                "270 outer refit runtime",
                "canonical orchestration, process-launch, and I/O overhead",
            ],
            "extrapolation_ratio": sum(model_inner.values()) / 36,
            "warning": "high_extrapolation_ratio_non_guaranteed",
        }
    two_vm = {"status": "pending_efficiency_input_not_authorized"}
    if two_vm_efficiency is not None:
        if not 0 < two_vm_efficiency < 1:
            raise PreflightError("two_vm_efficiency_must_be_measured_and_below_one")
        two_vm = {
            "status": "pending_multi_vm_overhead_evidence_not_authorized",
            "efficiency": two_vm_efficiency,
            "assumption": {"measured_or_operator_efficiency": two_vm_efficiency},
            "projected_elapsed_hours": None,
            "reason": "single_vm_total_elapsed_and_distributed_overhead_incomplete",
        }
    result = {
        "schema_version": 2,
        "status": "incomplete_total_elapsed_outer_refits_and_overhead_unknown",
        "execution_plan_eligible": False,
        "execution_plan_blockers": [
            "outer_refit_runtime_unmeasured",
            "canonical_orchestration_overhead_not_supported",
            "high_extrapolation_ratio_non_guaranteed",
        ],
        "coverage": {
            "observed_strata": len(required),
            "canonical_inner_fits": 54000,
            "outer_refits_unmeasured": 270,
        },
        "single_vm_parallel_1": estimates["cpu_parallel_1"],
        "single_vm_parallel_2": estimates["cpu_parallel_2"],
        "two_vm_cpu": two_vm,
        "gpu": {"status": "pending_gpu_preflight"},
        "cost": {
            "status": "pending_operator_price_input"
            if price is None and price_per_hour is None
            else "pending_complete_billable_elapsed_projection"
        },
    }
    return result


def ram_feasibility(
    records: list[dict[str, Any]], profile: dict[str, Any], *, evidence_scope: str
) -> dict[str, Any]:
    if evidence_scope != "target_single_vm_measured":
        return {
            "one_worker": "memory_feasibility_uncertain",
            "two_workers": "memory_feasibility_uncertain",
            "reason": "target_worst_case_concurrency_not_measured",
        }
    peak = max(
        (x.get("aggregate_process_tree_peak_rss_bytes", 0) for x in records), default=0
    )
    reserve = min(
        (x.get("system_available_ram_min_bytes", 0) for x in records), default=0
    )
    base = profile.get("system_used_ram_bytes")
    return {
        "one_worker": "observed_only",
        "two_workers": "pass"
        if peak and reserve >= 2147483648
        else "memory_feasibility_uncertain",
        "observed_peak_aggregate_bytes": peak,
        "base_system_used_ram_bytes": base,
        "expected_peak_total_used_bytes": (base + peak)
        if isinstance(base, (int, float))
        else "unknown",
        "remaining_margin_bytes": reserve,
        "required_reserve_bytes": 2147483648,
        "swap_total_bytes": profile.get("swap_total_bytes", "unknown"),
        "swap_used_bytes": profile.get("swap_used_bytes", "unknown"),
    }


def cost_estimate(
    projected_hours: tuple[float, float] | None, price: dict[str, Any] | None
) -> dict[str, Any]:
    if price is None:
        return {"status": "pending_operator_price_input"}
    required = {
        "price_per_hour",
        "currency",
        "billing_unit",
        "pricing_timestamp",
        "source",
    }
    if not required <= set(price) or projected_hours is None:
        return {"status": "insufficient_input"}
    return {
        "status": "derived_from_operator_input",
        "currency": price["currency"],
        "range": [
            projected_hours[0] * price["price_per_hour"],
            projected_hours[1] * price["price_per_hour"],
        ],
        "assumptions": {
            "tax": "unknown",
            "credits": "unknown",
            "storage_egress": price.get("storage_egress", "unknown"),
        },
    }


def proposed_execution_plan(
    *,
    git_commit: str,
    preflight_plan_digest: str,
    evidence_digest: str,
    mode: str,
    runtime_range: Any,
    ram: Any,
    cost: Any,
) -> dict[str, Any]:
    if (
        not isinstance(runtime_range, dict)
        or runtime_range.get("status") != "complete_total_elapsed_supported"
    ):
        raise PreflightError("projection_not_execution_plan_eligible")
    value = {
        "schema_version": 1,
        "git_commit": git_commit,
        "scientific_manifest_digest": SCIENTIFIC_DIGEST,
        "preflight_plan_digest": preflight_plan_digest,
        "preflight_evidence_digest": evidence_digest,
        "proposed_compute_mode": mode,
        "machine_requirements": {"role": "intended_single_vm_target"},
        "worker_thread_caps": {"workers": MODES.get(mode), "threads_per_worker": 2},
        "runtime_range": runtime_range,
        "ram_requirements": ram,
        "cost_range": cost,
        "artifact_namespace": "artifacts/p7c4b2b-compute-preflight",
        "retry_resume_policy": {"retry_maximum": 1, "resume_valid_completed": True},
        "canonical_execution_command": "pending_human_approval",
        "approval": {
            "status": "pending_human_approval",
            "approver": None,
            "timestamp": None,
            "signature": None,
        },
    }
    value["execution_plan_digest"] = sha256_canonical(value)
    return value


def execution_plan_digest(plan: dict[str, Any]) -> str:
    value = deepcopy(plan)
    value.pop("execution_plan_digest", None)
    return sha256_canonical(value)


def validate_execution_plan(plan: dict[str, Any]) -> dict[str, Any]:
    codes = []
    if plan.get("execution_plan_digest") != execution_plan_digest(plan):
        codes.append("execution_plan_digest_mismatch")
    if plan.get("scientific_manifest_digest") != SCIENTIFIC_DIGEST:
        codes.append("scientific_manifest_digest_mismatch")
    if plan.get("proposed_compute_mode") not in MODES:
        codes.append("canonical_compute_mode_missing")
    if plan.get("approval", {}).get("status") != "pending_human_approval":
        codes.append("proposed_plan_must_remain_unapproved")
    if codes:
        raise PreflightError(",".join(sorted(set(codes))))
    return {
        "valid": True,
        "execution_plan_digest": plan["execution_plan_digest"],
        "approval_status": "pending_human_approval",
    }


def execution_approval_guard(
    plan: dict[str, Any], approval: dict[str, Any] | None
) -> dict[str, Any]:
    codes = []
    try:
        validate_execution_plan(plan)
    except PreflightError as exc:
        codes.extend(str(exc).split(","))
    if not approval:
        codes.append("execution_cost_approval_missing")
    else:
        if approval.get("status") != "approved":
            codes.append("execution_cost_approval_missing")
        if approval.get("execution_plan_digest") != plan.get("execution_plan_digest"):
            codes.append("execution_plan_approval_digest_mismatch")
        if not approval.get("approver") or not approval.get("approved_utc"):
            codes.append("execution_cost_approval_provenance_missing")
    return {"authorized": not codes, "reason_codes": sorted(set(codes))}
