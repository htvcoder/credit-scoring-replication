"""Pure P7C.4B.2c plan, telemetry, projection, and eligibility contracts."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

from creditrep.config.loader import sha256_canonical
from creditrep.protocols.p7c4b2a import P7C4B2AError, load_manifest

SCHEMA_VERSION = 1
EXECUTION_CLASSES = {"synthetic_validation", "target_preflight"}
MODES = {"cpu_parallel_1": 1, "cpu_parallel_2": 2}
DATASET_OUTER_REPEATS = {"AC": 10, "GC": 10, "TH02": 10, "HMEQ": 5, "TC": 5, "GMC": 5}
PROXIES = ("low_cost_proxy", "typical_proxy", "high_cost_proxy")
ADDITIVE_COMPONENTS = (
    "preprocessing_elapsed_seconds",
    "model_fit_elapsed_seconds",
    "prediction_elapsed_seconds",
    "metric_elapsed_seconds",
    "artifact_write_elapsed_seconds",
    "other_measured_orchestration_elapsed_seconds",
)
NON_ADDITIVE_TIMINGS = (
    "warmup_elapsed_seconds",
    "measured_phase_elapsed_seconds",
    "aggregate_measured_fit_runtime_seconds",
    "aggregate_outer_refit_runtime_seconds",
    "worker_process_startup_elapsed_seconds",
    "task_dispatch_or_queue_elapsed_seconds",
)
TIMING_FIELDS = ADDITIVE_COMPONENTS + NON_ADDITIVE_TIMINGS
LOCKED_SCIENTIFIC_MANIFEST = Path(
    "configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml"
)
PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint",
        "scientific_manifest_digest",
        "runtime_input_binding",
        "population",
        "sampling",
        "timing_contract",
        "execution",
        "tasks",
        "plan_digest",
    }
)


def _runtime_input_binding() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "strategy": "semantic_projection",
        "digest_field": "locked_runtime_inputs_digest",
        "target_datasets": ["AC", "GMC"],
        "worker_recheck": "required_before_dataset_consumption",
    }


POPULATION_FIELDS = frozenset(
    {"count", "outer_partitions", "models", "dataset_partition_counts", "derivation"}
)
SAMPLING_FIELDS = frozenset(
    {
        "seed",
        "strata",
        "minimum_repetitions",
        "warmup_per_stratum",
        "measured_per_stratum",
        "proxy_limitation",
        "maximum_tasks",
        "maximum_measured_tasks",
        "stop_conditions",
    }
)
TIMING_CONTRACT_FIELDS = frozenset(
    {
        "clock",
        "additive_components",
        "non_additive_timings",
        "unknown_representation",
        "warmup_projection",
        "aggregate_outer_refit",
        "measured_phase",
        "telemetry_collection",
    }
)
EXECUTION_FIELDS = frozenset(
    {"classes", "target_authorization", "synthetic_scope", "output_collision"}
)
TASK_FIELDS = frozenset(
    {
        "dataset_id",
        "model_id",
        "candidate_proxy",
        "candidate_id",
        "candidate",
        "outer_repeat",
        "outer_fold",
        "dataset_size_class",
        "feature_count_class",
        "mode",
        "worker_count",
        "classification",
        "repetition",
        "seed",
        "sample_id",
    }
)
CANDIDATE_FIELDS = frozenset(
    {"hidden_units", "dropout", "l2", "learning_rate", "batch_normalization"}
)
MODEL_DEPTHS = {"mlp_1": 1, "mlp_3": 3, "mlp_5": 5}
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


class P7C4B2CError(ValueError):
    """Stable planning, validation, or execution-boundary failure."""


def canonical_digest(value: dict[str, Any], field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return sha256_canonical(payload)


def _plain_int(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _plain_float(
    value: Any, *, minimum: float = 0.0, maximum: float | None = None
) -> bool:
    return (
        type(value) is float
        and math.isfinite(value)
        and value >= minimum
        and (maximum is None or value < maximum)
    )


def _nonempty(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _digest(value: Any) -> bool:
    return type(value) is str and HEX_DIGEST.fullmatch(value) is not None


def _exact_keys(value: Any, fields: frozenset[str]) -> bool:
    return type(value) is dict and set(value) == fields


def _task_sort_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        list(MODES).index(task["mode"]),
        ("warmup", "measured").index(task["classification"]),
        list(DATASET_OUTER_REPEATS).index(task["dataset_id"]),
        list(MODEL_DEPTHS).index(task["model_id"]),
        PROXIES.index(task["candidate_proxy"]),
        task["repetition"],
    )


def _finite_nonnegative(value: Any, *, nullable: bool = False) -> bool:
    if value is None:
        return nullable
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _complexity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sum(candidate["hidden_units"]),
        max(candidate["hidden_units"]),
        candidate["dropout"],
        candidate["l2"],
        -float(candidate["learning_rate"]),
        candidate["candidate_id"],
    )


def _representatives(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ordered = sorted(candidates, key=_complexity)
    return {
        "low_cost_proxy": ordered[0],
        "typical_proxy": ordered[len(ordered) // 2],
        "high_cost_proxy": ordered[-1],
    }


def outer_population(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize 6 datasets × 90 partitions × 3 MLP families = 270 refits."""
    partitions = [
        {
            "dataset_id": dataset_id,
            "outer_repeat": repeat,
            "outer_fold": fold,
        }
        for dataset_id, repeats in DATASET_OUTER_REPEATS.items()
        for repeat in range(repeats)
        for fold in range(2)
    ]
    result = []
    for model in manifest["models"]:
        for partition in partitions:
            identity = {"model_id": model["id"], **partition}
            result.append({**identity, "population_id": sha256_canonical(identity)})
    expected = manifest["workload"]["total_outer_refits"]
    if (
        len(partitions) != manifest["workload"]["outer_partitions"]
        or len(result) != expected
    ):
        raise P7C4B2CError("outer_refit_population_mismatch")
    return result


def build_plan(manifest: dict[str, Any], *, seed: int = 4202) -> dict[str, Any]:
    """Build a deterministic, bounded, stratified target plan."""
    population = outer_population(manifest)
    model_candidates = {
        item["id"]: _representatives(item["candidates"]) for item in manifest["models"]
    }
    strata = []
    for dataset_index, dataset_id in enumerate(DATASET_OUTER_REPEATS):
        for model_id, representatives in model_candidates.items():
            for proxy, candidate in representatives.items():
                partition_count = DATASET_OUTER_REPEATS[dataset_id] * 2
                outer_ordinal = (
                    int(
                        sha256_canonical(
                            {
                                "seed": seed,
                                "dataset_id": dataset_id,
                                "model_id": model_id,
                                "proxy": proxy,
                            }
                        ),
                        16,
                    )
                    % partition_count
                )
                strata.append(
                    {
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "candidate_proxy": proxy,
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
                        "outer_repeat": outer_ordinal // 2,
                        "outer_fold": outer_ordinal % 2,
                        "dataset_size_class": (
                            "large" if dataset_index >= 4 else "small_or_medium"
                        ),
                        "feature_count_class": "materialize_at_execution",
                    }
                )
    tasks = []
    for mode, workers in MODES.items():
        for classification, repetitions in (("warmup", 1), ("measured", 2)):
            for stratum in strata:
                for repetition in range(repetitions):
                    identity = {
                        **stratum,
                        "mode": mode,
                        "worker_count": workers,
                        "classification": classification,
                        "repetition": repetition,
                        "seed": seed + repetition,
                    }
                    tasks.append({**identity, "sample_id": sha256_canonical(identity)})
    value = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": "P7C.4B.2c",
        "scientific_manifest_digest": manifest["lock"]["manifest_sha256"],
        "runtime_input_binding": _runtime_input_binding(),
        "population": {
            "count": len(population),
            "outer_partitions": len(population) // len(manifest["models"]),
            "models": [item["id"] for item in manifest["models"]],
            "dataset_partition_counts": {
                key: repeats * 2 for key, repeats in DATASET_OUTER_REPEATS.items()
            },
            "derivation": "sum(dataset outer repeats * two folds) * model families",
        },
        "sampling": {
            "seed": seed,
            "strata": [
                "dataset_id",
                "model_id",
                "candidate_proxy",
                "outer_repeat",
                "outer_fold",
                "mode",
            ],
            "minimum_repetitions": 2,
            "warmup_per_stratum": 1,
            "measured_per_stratum": 2,
            "proxy_limitation": "proxy_not_observed_canonical_selection",
            "maximum_tasks": len(tasks),
            "maximum_measured_tasks": sum(
                x["classification"] == "measured" for x in tasks
            ),
            "stop_conditions": [
                "per_sample_failure",
                "wall_clock_budget",
                "artifact_budget",
                "required_coverage_incomplete",
            ],
        },
        "timing_contract": {
            "clock": "time.perf_counter_monotonic",
            "additive_components": list(ADDITIVE_COMPONENTS),
            "non_additive_timings": list(NON_ADDITIVE_TIMINGS),
            "unknown_representation": None,
            "warmup_projection": "forbidden",
            "aggregate_outer_refit": "sum(additive_components)",
            "measured_phase": "dispatch_to_validated_artifact_promotion_wall_clock",
            "telemetry_collection": "included_in_other_measured_orchestration",
        },
        "execution": {
            "classes": sorted(EXECUTION_CLASSES),
            "target_authorization": "effective_authorization_and_locked_input_rebuild",
            "synthetic_scope": "pipeline_validation_only_non_scientific",
            "output_collision": "forbidden",
        },
        "tasks": tasks,
    }
    value["plan_digest"] = canonical_digest(value, "plan_digest")
    validate_plan(value)
    return value


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete plan shape without treating its digests as authority."""
    codes: list[str] = []
    if not _exact_keys(plan, PLAN_FIELDS):
        raise P7C4B2CError("plan_schema_invalid")
    if (
        type(plan["schema_version"]) is not int
        or plan["schema_version"] != SCHEMA_VERSION
    ):
        codes.extend(["plan_schema_invalid", "unsupported_schema_version"])
    if plan["checkpoint"] != "P7C.4B.2c":
        codes.extend(["plan_schema_invalid", "checkpoint_mismatch"])
    if not _digest(plan["scientific_manifest_digest"]):
        codes.append("plan_schema_invalid")
    if plan.get("runtime_input_binding") != _runtime_input_binding():
        codes.append("plan_schema_invalid")
    if not _digest(plan["plan_digest"]):
        codes.append("plan_schema_invalid")
    else:
        try:
            if plan["plan_digest"] != canonical_digest(plan, "plan_digest"):
                codes.append("plan_hash_mismatch")
        except (TypeError, ValueError):
            codes.extend(["plan_schema_invalid", "plan_hash_mismatch"])

    population = plan["population"]
    expected_partition_counts = {
        dataset: repeats * 2 for dataset, repeats in DATASET_OUTER_REPEATS.items()
    }
    if (
        not _exact_keys(population, POPULATION_FIELDS)
        or type(population.get("count")) is not int
        or type(population.get("outer_partitions")) is not int
        or population.get("count") != 270
        or population.get("outer_partitions") != 90
        or population.get("models") != list(MODEL_DEPTHS)
        or population.get("dataset_partition_counts") != expected_partition_counts
        or not _nonempty(population.get("derivation"))
    ):
        codes.extend(["plan_schema_invalid", "outer_refit_population_mismatch"])

    sampling = plan["sampling"]
    if (
        not _exact_keys(sampling, SAMPLING_FIELDS)
        or type(sampling.get("seed")) is not int
        or sampling.get("strata")
        != [
            "dataset_id",
            "model_id",
            "candidate_proxy",
            "outer_repeat",
            "outer_fold",
            "mode",
        ]
        or type(sampling.get("minimum_repetitions")) is not int
        or type(sampling.get("warmup_per_stratum")) is not int
        or type(sampling.get("measured_per_stratum")) is not int
        or type(sampling.get("maximum_tasks")) is not int
        or type(sampling.get("maximum_measured_tasks")) is not int
        or sampling.get("minimum_repetitions") != 2
        or sampling.get("warmup_per_stratum") != 1
        or sampling.get("measured_per_stratum") != 2
        or sampling.get("maximum_tasks") != 324
        or sampling.get("maximum_measured_tasks") != 216
        or not _nonempty(sampling.get("proxy_limitation"))
        or sampling.get("stop_conditions")
        != [
            "per_sample_failure",
            "wall_clock_budget",
            "artifact_budget",
            "required_coverage_incomplete",
        ]
    ):
        codes.append("plan_schema_invalid")

    timing = plan["timing_contract"]
    if (
        not _exact_keys(timing, TIMING_CONTRACT_FIELDS)
        or timing.get("clock") != "time.perf_counter_monotonic"
        or timing.get("additive_components") != list(ADDITIVE_COMPONENTS)
        or timing.get("non_additive_timings") != list(NON_ADDITIVE_TIMINGS)
        or timing.get("unknown_representation") is not None
        or timing.get("warmup_projection") != "forbidden"
        or timing.get("aggregate_outer_refit") != "sum(additive_components)"
        or timing.get("measured_phase")
        != "dispatch_to_validated_artifact_promotion_wall_clock"
        or timing.get("telemetry_collection")
        != "included_in_other_measured_orchestration"
    ):
        codes.append("plan_schema_invalid")

    execution = plan["execution"]
    if (
        not _exact_keys(execution, EXECUTION_FIELDS)
        or execution.get("classes") != sorted(EXECUTION_CLASSES)
        or execution.get("target_authorization")
        != "effective_authorization_and_locked_input_rebuild"
        or execution.get("synthetic_scope") != "pipeline_validation_only_non_scientific"
        or execution.get("output_collision") != "forbidden"
    ):
        codes.append("plan_schema_invalid")

    tasks = plan["tasks"]
    if type(tasks) is not list or len(tasks) != 324:
        codes.extend(["plan_task_schema_invalid", "plan_task_set_mismatch"])
        tasks = []
    ids: list[str] = []
    candidate_bindings: dict[str, tuple[str, str]] = {}
    for task in tasks:
        if not _exact_keys(task, TASK_FIELDS):
            codes.append("plan_task_schema_invalid")
            continue
        candidate = task["candidate"]
        if not _exact_keys(candidate, CANDIDATE_FIELDS):
            codes.append("plan_candidate_schema_invalid")
            continue
        model_id = task["model_id"]
        hidden = candidate["hidden_units"]
        candidate_valid = (
            type(hidden) is list
            and type(model_id) is str
            and model_id in MODEL_DEPTHS
            and len(hidden) == MODEL_DEPTHS.get(model_id)
            and all(_plain_int(unit, minimum=1) for unit in hidden)
            and hidden == sorted(hidden, reverse=True)
            and _plain_float(candidate["dropout"], maximum=1.0)
            and _plain_float(candidate["l2"])
            and _plain_float(candidate["learning_rate"], minimum=1e-300)
            and type(candidate["batch_normalization"]) is bool
        )
        if not candidate_valid:
            codes.append("plan_candidate_schema_invalid")
        dataset_id = task["dataset_id"]
        classification = task["classification"]
        repetition = task["repetition"]
        task_valid = (
            type(dataset_id) is str
            and dataset_id in DATASET_OUTER_REPEATS
            and type(model_id) is str
            and model_id in MODEL_DEPTHS
            and type(task["candidate_proxy"]) is str
            and task["candidate_proxy"] in PROXIES
            and _nonempty(task["candidate_id"])
            and _plain_int(task["outer_repeat"])
            and task["outer_repeat"] < DATASET_OUTER_REPEATS.get(dataset_id, 0)
            and type(task["outer_fold"]) is int
            and task["outer_fold"] in (0, 1)
            and type(task["dataset_size_class"]) is str
            and task["dataset_size_class"] in ("small_or_medium", "large")
            and task["feature_count_class"] == "materialize_at_execution"
            and type(task["mode"]) is str
            and task["mode"] in MODES
            and type(task["worker_count"]) is int
            and task["worker_count"] == MODES.get(task["mode"])
            and type(classification) is str
            and classification in ("warmup", "measured")
            and type(repetition) is int
            and repetition in ((0,) if classification == "warmup" else (0, 1))
            and type(task["seed"]) is int
            and task["seed"] == sampling.get("seed", 0) + repetition
            and _digest(task["sample_id"])
        )
        if not task_valid:
            codes.append("plan_task_schema_invalid")
        if (
            type(task["mode"]) is str
            and task["mode"] in MODES
            and task["worker_count"] != MODES[task["mode"]]
        ):
            codes.append("execution_mode_worker_mismatch")
        identity = {key: value for key, value in task.items() if key != "sample_id"}
        try:
            identity_matches = task["sample_id"] == sha256_canonical(identity)
        except (TypeError, ValueError):
            identity_matches = False
        if not identity_matches:
            codes.append("sample_identity_mismatch")
        if type(task["sample_id"]) is str:
            ids.append(task["sample_id"])
        else:
            codes.append("plan_task_schema_invalid")
        try:
            binding = (model_id, sha256_canonical(candidate))
        except (TypeError, ValueError):
            binding = (model_id, "invalid")
        candidate_id = task["candidate_id"]
        if _nonempty(candidate_id):
            previous = candidate_bindings.setdefault(candidate_id, binding)
        else:
            previous = None
        if previous != binding or binding[1] == "invalid":
            codes.append("plan_candidate_schema_invalid")

    if len(ids) != len(set(ids)):
        codes.extend(["duplicate_sample_identity", "plan_task_set_mismatch"])
    if tasks:
        try:
            if tasks != sorted(tasks, key=_task_sort_key):
                codes.append("plan_task_set_mismatch")
        except (KeyError, ValueError, TypeError):
            codes.append("plan_task_schema_invalid")
    measured = [x for x in tasks if x.get("classification") == "measured"]
    groups: dict[tuple[Any, ...], int] = defaultdict(int)
    for task in measured:
        try:
            groups[
                (
                    task["dataset_id"],
                    task["model_id"],
                    task["candidate_proxy"],
                    task["mode"],
                )
            ] += 1
        except (KeyError, TypeError):
            codes.append("plan_task_schema_invalid")
    if (
        len(measured) != 216
        or len(groups) != 108
        or any(count != 2 for count in groups.values())
    ):
        codes.extend(["insufficient_repetitions", "plan_task_set_mismatch"])
    if codes:
        raise P7C4B2CError(",".join(sorted(set(codes))))
    return {
        "valid": True,
        "population_count": 270,
        "tasks": len(tasks),
        "measured_tasks": len(measured),
        "strata": len(groups),
    }


def rebuild_canonical_plan(repo_root: Path) -> dict[str, Any]:
    """Rebuild the target authority only from the locked scientific inputs."""
    root = repo_root.resolve()
    try:
        manifest = load_manifest(
            LOCKED_SCIENTIFIC_MANIFEST
            if root == Path.cwd().resolve()
            else root / LOCKED_SCIENTIFIC_MANIFEST,
            repo_root=root,
        )
        expected = build_plan(manifest)
    except (
        OSError,
        P7C4B2AError,
        P7C4B2CError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise P7C4B2CError("plan_locked_input_mismatch") from exc
    return expected


def validate_canonical_plan(plan: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Return a rebuilt canonical plan after strict validation and exact comparison."""
    validate_plan(plan)
    expected = rebuild_canonical_plan(repo_root)
    if plan["scientific_manifest_digest"] != expected["scientific_manifest_digest"]:
        raise P7C4B2CError("plan_locked_input_mismatch")
    if sha256_canonical(plan["tasks"]) != sha256_canonical(expected["tasks"]):
        raise P7C4B2CError("plan_task_set_mismatch")
    if sha256_canonical(plan) != sha256_canonical(expected):
        raise P7C4B2CError("plan_not_canonical")
    return expected


def validate_sample(
    record: dict[str, Any], task: dict[str, Any], manifest: dict[str, Any]
) -> list[str]:
    codes = []
    if record.get("schema_version") != SCHEMA_VERSION:
        codes.append("unsupported_schema_version")
    for key in (
        "sample_id",
        "dataset_id",
        "model_id",
        "candidate_proxy",
        "outer_repeat",
        "outer_fold",
        "mode",
        "worker_count",
        "classification",
        "repetition",
        "seed",
    ):
        if record.get(key) != task.get(key):
            codes.append("sample_identity_mismatch")
            break
    if record.get("plan_digest") != manifest.get("plan_digest"):
        codes.append("plan_hash_mismatch")
    if not record.get("git_commit"):
        codes.append("git_provenance_missing")
    if record.get("execution_class") != manifest.get("execution_class"):
        codes.append("execution_class_mismatch")
    if record.get("input_hash") != sha256_canonical(record.get("input_identity")):
        codes.append("input_hash_mismatch")
    if record.get("mode") not in MODES or record.get("worker_count") != MODES.get(
        record.get("mode")
    ):
        codes.append("execution_mode_worker_mismatch")
    started, ended = record.get("started_monotonic"), record.get("completed_monotonic")
    if (
        not _finite_nonnegative(started)
        or not _finite_nonnegative(ended)
        or ended < started
    ):
        codes.append("invalid_timestamp_order")
    for field in TIMING_FIELDS:
        if not _finite_nonnegative(record.get(field), nullable=True):
            codes.append("invalid_timing_component")
            break
    components = [record.get(field) for field in ADDITIVE_COMPONENTS]
    aggregate = record.get("aggregate_outer_refit_runtime_seconds")
    if any(value is None for value in components) or aggregate is None:
        codes.append("required_component_unknown")
    elif not math.isclose(sum(components), aggregate, rel_tol=1e-9, abs_tol=1e-6):
        codes.append("invalid_additivity_semantics")
    enclosing = (
        ended - started
        if _finite_nonnegative(started) and _finite_nonnegative(ended)
        else None
    )
    if enclosing is not None and any(
        value is not None and value > enclosing + 1e-6 for value in components
    ):
        codes.append("component_exceeds_enclosing_interval")
    if (
        record.get("classification") == "warmup"
        and record.get("projection_eligible") is not False
    ):
        codes.append("warmup_mixed_into_projection")
    if record.get("status") != "completed":
        codes.append("failed_sample")
    return sorted(set(codes))


def summarize(
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    expected_tasks: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    measured = [
        x
        for x in records
        if x.get("classification") == "measured" and x.get("status") == "completed"
    ]
    expected = [
        x
        for x in (expected_tasks if expected_tasks is not None else plan["tasks"])
        if x["classification"] == "measured"
    ]
    expected_groups = {
        (x["dataset_id"], x["model_id"], x["candidate_proxy"], x["mode"])
        for x in expected
    }
    actual_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in measured:
        actual_groups[
            (
                record["dataset_id"],
                record["model_id"],
                record["candidate_proxy"],
                record["mode"],
            )
        ].append(record["aggregate_outer_refit_runtime_seconds"])
    incomplete = [
        list(key)
        for key in sorted(expected_groups)
        if len(actual_groups.get(key, [])) < 2
    ]
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "expected_measured": len(expected),
        "completed_measured": len(measured),
        "expected_strata": len(expected_groups),
        "covered_strata": sum(
            len(actual_groups.get(key, [])) >= 2 for key in expected_groups
        ),
        "incomplete_strata": incomplete,
        "minimum_repetitions": 2,
    }
    strata = [
        {
            "dataset_id": key[0],
            "model_id": key[1],
            "candidate_proxy": key[2],
            "mode": key[3],
            "count": len(values),
            "median_outer_refit_seconds": median(values),
            "minimum_outer_refit_seconds": min(values),
            "maximum_outer_refit_seconds": max(values),
        }
        for key, values in sorted(actual_groups.items())
    ]
    return coverage, {"schema_version": SCHEMA_VERSION, "strata": strata}


def eligibility(
    *,
    artifact_valid: bool,
    execution_class: str,
    coverage_complete: bool,
    clean_overhead: bool,
    inner_evidence_valid: bool,
    total_elapsed: dict[str, float] | None,
    cost_complete: bool,
    high_severity_warnings: list[str] | None = None,
) -> dict[str, Any]:
    codes = []
    if not artifact_valid:
        codes.append("invalid_artifact_evidence")
    if execution_class != "target_preflight":
        codes.append("synthetic_evidence_not_target_evidence")
    if not coverage_complete:
        codes.append("outer_refit_coverage_incomplete")
    if not clean_overhead:
        codes.append("clean_overhead_measurement_missing")
    if not inner_evidence_valid:
        codes.append("inner_fit_evidence_missing")
    if total_elapsed is None:
        codes.append("required_component_unknown")
    if not cost_complete:
        codes.append("operator_price_input_missing")
    if high_severity_warnings:
        codes.append("high_severity_warning_present")
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_plan_eligible": not codes,
        "reason_codes": sorted(codes),
    }


def project_validated(
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    artifact_validation: dict[str, Any],
    execution_class: str,
    inner_projection: dict[str, Any] | None = None,
    overhead_mapping: dict[str, Any] | None = None,
    price_input: dict[str, Any] | None = None,
    allow_controlled_fixture_eligibility: bool = False,
) -> dict[str, Any]:
    """Create a projection only from a validator PASS bound to these artifacts."""
    valid_source = artifact_validation.get("valid") is True and artifact_validation.get(
        "evidence_digest"
    ) == sha256_canonical(records)
    coverage, summary = summarize(records, plan)
    coverage_complete = coverage["covered_strata"] == coverage["expected_strata"]
    controlled = (
        execution_class == "controlled_target_fixture"
        and allow_controlled_fixture_eligibility
    )
    target = execution_class == "target_preflight" or controlled
    warnings = ["proxy_based_outer_refit_projection"]
    if not valid_source or not target or not coverage_complete:
        gate = eligibility(
            artifact_valid=valid_source,
            execution_class="target_preflight" if controlled else execution_class,
            coverage_complete=coverage_complete,
            clean_overhead=False,
            inner_evidence_valid=False,
            total_elapsed=None,
            cost_complete=False,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "incomplete",
            "source_evidence_digest": artifact_validation.get("evidence_digest"),
            "source_artifact_hashes": artifact_validation.get(
                "source_artifact_hashes", []
            ),
            "coverage": coverage,
            "total_canonical_elapsed_seconds": None,
            "warnings": warnings,
            **gate,
        }
    by_mode: dict[str, dict[str, Any]] = {}
    population_counts = plan["population"]["dataset_partition_counts"]
    for mode, workers in MODES.items():
        rows = [x for x in summary["strata"] if x["mode"] == mode]
        point = lower = upper = 0.0
        for row in rows:
            weight = population_counts[row["dataset_id"]] / 3
            point += weight * row["median_outer_refit_seconds"]
            lower += weight * row["minimum_outer_refit_seconds"]
            upper += weight * row["maximum_outer_refit_seconds"]
        scheduler_parallel = bool(
            (overhead_mapping or {}).get("outer_refits_parallel", False)
        )
        divisor = workers if scheduler_parallel else 1
        by_mode[mode] = {
            "aggregate_work_seconds": point,
            "conditional_elapsed_seconds": point / divisor,
            "lower_seconds": lower / divisor,
            "upper_seconds": upper / divisor,
            "worker_divisor": divisor,
        }
    mapping_complete = bool((overhead_mapping or {}).get("complete"))
    inner_valid = bool(
        inner_projection and inner_projection.get("valid_for_combination")
    )
    selected_mode = (overhead_mapping or {}).get("selected_mode")
    total = None
    if selected_mode in by_mode and mapping_complete and inner_valid:
        overhead = float(overhead_mapping["projected_seconds"])
        inner = inner_projection["conditional_elapsed_seconds"]
        total = {
            "point": inner["point"]
            + by_mode[selected_mode]["conditional_elapsed_seconds"]
            + overhead,
            "lower": inner["lower"]
            + by_mode[selected_mode]["lower_seconds"]
            + overhead,
            "upper": inner["upper"]
            + by_mode[selected_mode]["upper_seconds"]
            + overhead,
        }
    gate = eligibility(
        artifact_valid=valid_source,
        execution_class="target_preflight" if controlled else execution_class,
        coverage_complete=coverage_complete,
        clean_overhead=mapping_complete,
        inner_evidence_valid=inner_valid,
        total_elapsed=total,
        cost_complete=price_input is not None,
        high_severity_warnings=[],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if gate["execution_plan_eligible"] else "incomplete",
        "source_evidence_digest": artifact_validation["evidence_digest"],
        "source_artifact_hashes": artifact_validation.get("source_artifact_hashes", []),
        "coverage": coverage,
        "outer_refit_projection_by_mode": by_mode,
        "total_canonical_elapsed_seconds": total,
        "extrapolation_ratio": 270
        / max(len([x for x in records if x.get("classification") == "measured"]), 1),
        "warnings": warnings,
        **gate,
    }
