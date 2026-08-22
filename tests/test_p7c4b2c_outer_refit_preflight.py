from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from creditrep.config.loader import sha256_canonical
from creditrep.experiments.p7c4b2c_preflight import (
    canonical_outer_refit,
    resume,
    run,
    validate_artifacts,
    validate_resume_state,
)
from creditrep.experiments import p7c4b2c_preflight as outer_runner
from creditrep.experiments.p7c4b2c_cli import main as cli_main
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2b import (
    DATASET_FINGERPRINTS as INNER_DATASET_FINGERPRINTS,
    SCIENTIFIC_DIGEST as INNER_SCIENTIFIC_DIGEST,
    build_plan as build_inner_plan,
    machine_profile_digest,
)
from creditrep.protocols.p7c4b2b_authorization import (
    APPROVAL_PHRASE as INNER_APPROVAL_PHRASE,
    create_effective_authorization as create_inner_authorization,
    render_authorization_proposal as render_inner_proposal,
    render_target_environment as render_inner_environment,
)
from creditrep.protocols.p7c4b2c import (
    ADDITIVE_COMPONENTS,
    OVERHEAD_EVENT_COUNTS,
    OVERHEAD_METHOD_IDENTITY,
    P7C4B2CError,
    build_plan,
    eligibility,
    project_validated,
    overhead_artifact_digest,
    validate_canonical_plan,
    validate_plan,
    validate_combined_projection_sources,
    validate_combined_projection_identity,
)


def _plan():
    return build_plan(
        load_manifest("configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml")
    )


def _synthetic_run(
    tmp_path: Path, name: str = "synthetic-run", *, spawned: bool = False
) -> Path:
    output = tmp_path / name
    result = run(
        _plan(),
        output,
        execution_class="synthetic_validation",
        mode="cpu_parallel_1",
        max_samples=2,
    )
    assert result["validation"]["valid"] is True
    return output


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_population_sampling_and_plan_hash_are_deterministic():
    first, second = _plan(), _plan()
    assert first["population"]["count"] == 270
    assert first["population"]["outer_partitions"] == 90
    assert first["population"]["dataset_partition_counts"] == {
        "AC": 20,
        "GC": 20,
        "TH02": 20,
        "HMEQ": 10,
        "TC": 10,
        "GMC": 10,
    }
    assert first["plan_digest"] == second["plan_digest"]
    report = validate_plan(first)
    assert report == {
        "valid": True,
        "population_count": 270,
        "tasks": 324,
        "measured_tasks": 216,
        "strata": 108,
    }


def test_plan_mutation_and_worker_mismatch_are_rejected():
    plan = _plan()
    plan["tasks"][0]["worker_count"] = 9
    with pytest.raises(P7C4B2CError) as exc:
        validate_plan(plan)
    assert "plan_hash_mismatch" in str(exc.value)
    assert "execution_mode_worker_mismatch" in str(exc.value)


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda plan: plan.__setitem__("unknown", True), "plan_schema_invalid"),
        (
            lambda plan: plan["tasks"][0].__setitem__("unknown", True),
            "plan_task_schema_invalid",
        ),
        (
            lambda plan: plan["tasks"][0]["candidate"].__setitem__("unknown", 1),
            "plan_candidate_schema_invalid",
        ),
        (
            lambda plan: plan["tasks"][0].__setitem__("seed", True),
            "plan_task_schema_invalid",
        ),
        (
            lambda plan: plan["tasks"][0].__setitem__("mode", ["cpu_parallel_1"]),
            "plan_task_schema_invalid",
        ),
        (
            lambda plan: plan["tasks"][0].__setitem__("candidate_id", ["invalid"]),
            "plan_task_schema_invalid",
        ),
        (
            lambda plan: plan["tasks"][0]["candidate"].__setitem__(
                "dropout", float("nan")
            ),
            "plan_candidate_schema_invalid",
        ),
        (
            lambda plan: plan["tasks"][0]["candidate"].__setitem__(
                "hidden_units", (5,)
            ),
            "plan_candidate_schema_invalid",
        ),
    ],
)
def test_plan_schema_is_closed_world_and_never_coerces(mutation, code):
    plan = _plan()
    mutation(plan)
    with pytest.raises(P7C4B2CError, match=code):
        validate_plan(plan)


def test_exact_plan_is_rebuilt_from_locked_inputs():
    plan = _plan()
    rebuilt = validate_canonical_plan(plan, Path.cwd())
    assert rebuilt == plan
    changed = json.loads(json.dumps(plan))
    task = changed["tasks"][0]
    task["candidate"]["hidden_units"] = [999]
    task["sample_id"] = sha256_canonical(
        {key: value for key, value in task.items() if key != "sample_id"}
    )
    changed["plan_digest"] = sha256_canonical(
        {key: value for key, value in changed.items() if key != "plan_digest"}
    )
    with pytest.raises(P7C4B2CError, match="plan_"):
        validate_canonical_plan(changed, Path.cwd())


def test_missing_locked_inputs_fail_closed(tmp_path):
    with pytest.raises(P7C4B2CError, match="plan_locked_input_mismatch"):
        validate_canonical_plan(_plan(), tmp_path)


def test_synthetic_end_to_end_has_real_stages_and_is_ineligible(tmp_path):
    output = _synthetic_run(tmp_path, spawned=True)
    report = validate_artifacts(output)
    assert report["valid"] is True
    assert report["execution_class"] == "synthetic_validation"
    records = [_read(path) for path in output.glob("samples/*/result.json")]
    measured = [x for x in records if x["classification"] == "measured"]
    assert measured
    for record in measured:
        assert record["preprocessing_elapsed_seconds"] >= 0
        assert record["model_fit_elapsed_seconds"] >= 0
        assert record["prediction_elapsed_seconds"] >= 0
        assert record["metric_elapsed_seconds"] >= 0
        assert record["artifact_write_elapsed_seconds"] >= 0
        assert record["aggregate_outer_refit_runtime_seconds"] == pytest.approx(
            sum(record[field] for field in ADDITIVE_COMPONENTS)
        )
    projection = _read(output / "projection.json")
    assert projection["execution_plan_eligible"] is False
    assert "synthetic_evidence_not_target_evidence" in projection["reason_codes"]


def test_target_requires_effective_authorization_and_rejects_legacy_bypass(tmp_path):
    plan = _plan()
    with pytest.raises(P7C4B2CError, match="authorization_missing"):
        run(
            plan,
            tmp_path / "target",
            execution_class="target_preflight",
            mode="cpu_parallel_1",
        )
    with pytest.raises(
        P7C4B2CError, match="legacy_target_authorization_flags_forbidden"
    ):
        run(
            plan,
            tmp_path / "target-2",
            execution_class="target_preflight",
            mode="cpu_parallel_1",
            target_authorized=True,
            authorization_plan_digest="wrong",
        )


def test_run_has_no_overwrite_and_resume_skips_valid_samples(tmp_path):
    output = _synthetic_run(tmp_path)
    with pytest.raises(P7C4B2CError, match="output_collision"):
        run(
            _plan(),
            output,
            execution_class="synthetic_validation",
            mode="cpu_parallel_1",
            max_samples=2,
        )
    result = resume(output)
    assert result["executed"] == 0
    assert result["skipped"] == 4


def test_resume_state_accepts_structurally_valid_incomplete_run(tmp_path):
    output = _synthetic_run(tmp_path)
    sample = next((output / "samples").iterdir())
    for path in (sample, output / "COMPLETED.json"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    state = validate_resume_state(output)
    assert state["valid"] is True
    assert state["completion_state"] == "incomplete"
    assert state["resume_safe"] is True


def test_resume_state_rejects_completed_sample_integrity_failure(tmp_path):
    output = _synthetic_run(tmp_path)
    sample = next((output / "samples").iterdir())
    (sample / "COMPLETED.json").unlink()
    (output / "COMPLETED.json").unlink()
    state = validate_resume_state(output)
    assert state["resume_safe"] is False
    assert "complete_marker_integrity_failure" in state["reason_codes"]


def test_outer_rss_sampler_tolerates_only_verified_child_disappearance(monkeypatch):
    class Process:
        pid = 9

        def children(self, recursive):
            return []

        def memory_info(self):
            raise outer_runner.psutil.NoSuchProcess(9)

    monkeypatch.setattr(outer_runner.psutil, "Process", lambda _pid: Process())
    monkeypatch.setattr(outer_runner.psutil, "pid_exists", lambda _pid: False)
    assert outer_runner._aggregate_process_tree_rss(9) == 0
    monkeypatch.setattr(outer_runner.psutil, "pid_exists", lambda _pid: True)
    with pytest.raises(P7C4B2CError, match="memory_sampler_failure"):
        outer_runner._aggregate_process_tree_rss(9)


def test_resume_quarantines_corrupt_sample_and_stale_temp(tmp_path):
    output = _synthetic_run(tmp_path)
    sample = next((output / "samples").iterdir())
    (sample / "result.json").write_text("{broken", encoding="utf-8")
    stale = output / "tmp" / "stale-attempt"
    stale.mkdir(parents=True)
    (stale / "partial.json").write_text("{}", encoding="utf-8")
    result = resume(output)
    assert result["executed"] == 1
    names = [path.name for path in (output / "quarantine").iterdir()]
    assert any("corrupt_completed_sample" in name for name in names)
    assert any("stale_temporary_attempt" in name for name in names)


def test_public_workload_injection_is_rejected_before_output(tmp_path):
    def fail(_task, _root):
        raise RuntimeError("fixture failure")

    output = tmp_path / "failed-run"
    with pytest.raises(TypeError, match="unexpected keyword argument 'workload'"):
        run(
            _plan(),
            output,
            execution_class="synthetic_validation",
            mode="cpu_parallel_1",
            max_samples=1,
            workload=fail,
        )
    assert not output.exists()


def test_canonical_workload_cannot_be_paired_with_synthetic_class(tmp_path):
    output = tmp_path / "bypass"
    with pytest.raises(TypeError, match="unexpected keyword argument 'workload'"):
        run(
            _plan(),
            output,
            execution_class="synthetic_validation",
            mode="cpu_parallel_1",
            workload=canonical_outer_refit,
        )
    assert not output.exists()


def test_resume_rejects_workload_injection_before_mutation(tmp_path):
    output = _synthetic_run(tmp_path)
    before = sorted(path.relative_to(output) for path in output.rglob("*"))
    with pytest.raises(TypeError, match="unexpected keyword argument 'workload'"):
        resume(output, workload=canonical_outer_refit)
    assert sorted(path.relative_to(output) for path in output.rglob("*")) == before


def test_unknown_execution_class_is_rejected_before_output(tmp_path):
    output = tmp_path / "unknown"
    with pytest.raises(P7C4B2CError, match="unsupported_execution_class"):
        run(
            _plan(),
            output,
            execution_class="foreign",
            mode="cpu_parallel_1",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda record: record.update(git_commit=""), "git_provenance_missing"),
        (lambda record: record.update(input_hash="bad"), "input_hash_mismatch"),
        (
            lambda record: record.update(model_fit_elapsed_seconds=-1),
            "invalid_timing_component",
        ),
        (
            lambda record: record.update(model_fit_elapsed_seconds=float("inf")),
            "invalid_json_number",
        ),
        (
            lambda record: record.update(completed_monotonic=0),
            "invalid_timestamp_order",
        ),
        (lambda record: record.update(worker_count=9), "sample_identity_mismatch"),
        (
            lambda record: record.update(aggregate_outer_refit_runtime_seconds=999),
            "invalid_additivity_semantics",
        ),
    ],
)
def test_validator_detects_sample_corruption(tmp_path, mutation, reason):
    output = _synthetic_run(tmp_path)
    directory = next((output / "samples").iterdir())
    record = _read(directory / "result.json")
    mutation(record)
    _write(directory / "result.json", record)
    marker = _read(directory / "COMPLETED.json")
    if all(
        not isinstance(value, float) or value not in {float("inf"), float("-inf")}
        for value in record.values()
    ):
        marker["record_digest"] = sha256_canonical(record)
    _write(directory / "COMPLETED.json", marker)
    assert reason in validate_artifacts(output)["reason_codes"]


def test_validator_detects_missing_unexpected_and_marker_integrity(tmp_path):
    output = _synthetic_run(tmp_path)
    directory = next((output / "samples").iterdir())
    (directory / "COMPLETED.json").unlink()
    unexpected = output / "samples" / "unexpected"
    unexpected.mkdir()
    _write(unexpected / "result.json", {})
    report = validate_artifacts(output)
    assert "complete_marker_integrity_failure" in report["reason_codes"]
    assert "unexpected_sample" in report["reason_codes"]


def test_warmup_is_excluded_and_unknown_fails_closed(tmp_path):
    output = _synthetic_run(tmp_path)
    warmup = next(
        _read(path)
        for path in output.glob("samples/*/result.json")
        if _read(path)["classification"] == "warmup"
    )
    assert warmup["projection_eligible"] is False
    assert warmup["warmup_elapsed_seconds"] > 0
    gate = eligibility(
        artifact_valid=True,
        execution_class="target_preflight",
        coverage_complete=True,
        clean_overhead=True,
        inner_evidence_valid=True,
        total_elapsed=None,
        cost_complete=True,
    )
    assert gate["execution_plan_eligible"] is False
    assert gate["reason_codes"] == ["required_component_unknown"]


def _controlled_records(plan):
    records = []
    for task in plan["tasks"]:
        if task["classification"] != "measured":
            continue
        timings = {field: 1.0 for field in ADDITIVE_COMPONENTS}
        records.append(
            {
                **task,
                **timings,
                "aggregate_outer_refit_runtime_seconds": float(
                    len(ADDITIVE_COMPONENTS)
                ),
                "status": "completed",
            }
        )
    return records


def test_controlled_complete_evidence_can_pass_eligibility_logic():
    plan = _plan()
    records = _controlled_records(plan)
    validation = {
        "valid": True,
        "evidence_digest": sha256_canonical(records),
        "source_git_commit": "d" * 40,
        "source_artifact_hashes": ["e" * 64, "f" * 64],
    }
    projection = project_validated(
        records,
        plan,
        artifact_validation=validation,
        execution_class="controlled_target_fixture",
        inner_projection={
            "valid_for_combination": True,
            "conditional_elapsed_seconds": {
                "point": 100.0,
                "lower": 90.0,
                "upper": 110.0,
            },
        },
        overhead_mapping={
            "complete": True,
            "selected_mode": "cpu_parallel_2",
            "outer_refits_parallel": True,
            "projected_seconds": 5.0,
        },
        price_input={"fixture_only": True},
        allow_controlled_fixture_eligibility=True,
    )
    assert projection["execution_plan_eligible"] is True
    assert projection["reason_codes"] == []
    assert (
        projection["outer_refit_projection_by_mode"]["cpu_parallel_1"]["worker_divisor"]
        == 1
    )
    assert (
        projection["outer_refit_projection_by_mode"]["cpu_parallel_2"]["worker_divisor"]
        == 2
    )
    assert "proxy_based_outer_refit_projection" in projection["warnings"]


def test_missing_overhead_cost_inner_and_synthetic_all_fail_closed():
    gate = eligibility(
        artifact_valid=False,
        execution_class="synthetic_validation",
        coverage_complete=False,
        clean_overhead=False,
        inner_evidence_valid=False,
        total_elapsed=None,
        cost_complete=False,
        high_severity_warnings=["bad"],
    )
    assert gate["execution_plan_eligible"] is False
    assert gate["reason_codes"] == sorted(
        [
            "invalid_artifact_evidence",
            "synthetic_evidence_not_target_evidence",
            "outer_refit_coverage_incomplete",
            "clean_overhead_measurement_missing",
            "inner_fit_evidence_missing",
            "required_component_unknown",
            "operator_price_input_missing",
            "high_severity_warning_present",
        ]
    )


def test_cli_commands_and_exit_codes(tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    assert cli_main(["create-plan", "--output", str(plan_path)]) == 0
    assert plan_path.exists()
    assert cli_main(["validate-plan"]) == 0
    assert (
        cli_main(
            [
                "run",
                "--output",
                str(tmp_path / "unauthorized"),
                "--execution-class",
                "target_preflight",
            ]
        )
        == 4
    )
    output = tmp_path / "cli-synthetic"
    assert (
        cli_main(
            [
                "run",
                "--output",
                str(output),
                "--execution-class",
                "synthetic_validation",
                "--max-samples",
                "2",
            ]
        )
        == 0
    )
    assert cli_main(["validate-artifacts", "--run-dir", str(output)]) == 0
    assert cli_main(["project", "--run-dir", str(output)]) == 3
    assert cli_main(["inspect-eligibility", "--run-dir", str(output)]) == 3
    assert "synthetic_evidence_not_target_evidence" in capsys.readouterr().out


def test_validator_rejects_unsupported_schema_and_invalid_true_gate(tmp_path):
    output = _synthetic_run(tmp_path)
    manifest = _read(output / "manifest.json")
    manifest["schema_version"] = 999
    _write(output / "manifest.json", manifest)
    gate = _read(output / "eligibility.json")
    gate["execution_plan_eligible"] = True
    gate["reason_codes"] = []
    _write(output / "eligibility.json", gate)
    report = validate_artifacts(output)
    assert "unsupported_schema_version" in report["reason_codes"]
    assert "invalid_true_eligibility" in report["reason_codes"]


def test_validator_rejects_invalid_projection_source(tmp_path):
    output = _synthetic_run(tmp_path)
    projection = _read(output / "projection.json")
    projection["source_evidence_digest"] = "foreign"
    _write(output / "projection.json", projection)
    assert "invalid_projection_source" in validate_artifacts(output)["reason_codes"]


def test_combined_projection_requires_exact_duplicate_free_canonical_scope():
    plan = _plan()
    records = [{"sample_id": task["sample_id"]} for task in plan["tasks"]]
    report = validate_combined_projection_sources(
        records, plan, [{"valid": True, "evidence_digest": "a" * 64}]
    )
    assert report["valid"] is True
    assert report["completed_tasks"] == report["expected_tasks"] == 324

    duplicate = validate_combined_projection_sources(
        [*records, records[0]],
        plan,
        [{"valid": True, "evidence_digest": "a" * 64}],
    )
    assert duplicate["valid"] is False
    assert "duplicate_projection_sample" in duplicate["reason_codes"]

    incomplete = validate_combined_projection_sources(
        records[:-1], plan, [{"valid": True, "evidence_digest": "a" * 64}]
    )
    assert incomplete["valid"] is False
    assert "incomplete_projection_task_scope" in incomplete["reason_codes"]


def _combined_identity_fixture():
    source = "a" * 40
    scientific = INNER_SCIENTIFIC_DIGEST
    outer_plan = {"plan_digest": "c" * 64}
    inner_plan = build_inner_plan(
        load_manifest("configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml")
    )
    outer = []
    inner = []
    for index, mode in enumerate(("cpu_parallel_1", "cpu_parallel_2"), start=1):
        outer_path = str((Path.cwd() / f"outer-p{index}").resolve())
        outer.append(
            {
                "run_directory": outer_path,
                "manifest": {
                    "run_id": Path(outer_path).name,
                    "execution_class": "target_preflight",
                    "mode": mode,
                    "plan_digest": outer_plan["plan_digest"],
                    "scientific_manifest_digest": scientific,
                    "authorization_provenance": {
                        "execution_stage": "target_projection_preflight",
                        "execution_mode": mode,
                        "normalized_output_directory": outer_path,
                        "git_commit": source,
                        "target_environment_digest": f"{index}" * 64,
                        "proposal_digest": f"{index + 2}" * 64,
                        "authorization_digest": f"{index + 4}" * 64,
                    },
                },
                "environment": {"git_commit": source},
                "validation": {"valid": True},
            }
        )
        inner_path = str((Path.cwd() / f"inner-p{index}").resolve())
        profile = {
            "schema_version": 1,
            "machine_role": "intended_single_vm_target",
            "machine_id": f"machine-{index}",
            "cloud_provider": "fixture-provider",
            "instance_type": f"fixture-p{index}",
            "os": "fixture-linux",
            "cpu_model": "fixture-cpu",
            "physical_cores": 8,
            "logical_cores": 16,
            "ram_total_bytes": 32 * 1024**3,
            "system_used_ram_bytes": 1024**3,
            "swap_total_bytes": 0,
            "swap_used_bytes": 0,
            "disk_free_bytes": 64 * 1024**3,
            "python_executable": ".venv/bin/python",
            "python_version": "3.11",
            "dependency_fingerprint": "e" * 64,
            "git_commit": source,
            "scientific_manifest_digest": scientific,
            "dataset_fingerprints": INNER_DATASET_FINGERPRINTS,
            "worker_limit": 2,
            "threads_per_worker": 2,
            "virtualization_power": "unknown",
            "utc_captured": "2026-08-14T00:00:00Z",
        }
        profile["profile_digest"] = machine_profile_digest(profile)
        environment = render_inner_environment(
            inner_plan,
            profile,
            mode=mode,
            output_directory=Path(inner_path),
            captured_at="2026-08-14T00:00:00Z",
        )
        proposal = render_inner_proposal(
            environment,
            inner_plan,
            profile,
            run_id=Path(inner_path).name,
            created_at="2026-08-14T00:00:00Z",
        )
        authorization = create_inner_authorization(
            proposal,
            environment,
            inner_plan,
            profile,
            operator_identity=f"operator-{index}",
            operator_approval=INNER_APPROVAL_PHRASE,
            created_at="2026-08-14T00:00:00Z",
            expires_at="2099-08-14T00:00:00Z",
        )
        profile_digest = profile["profile_digest"]
        inner.append(
            {
                "run_directory": inner_path,
                "manifest": {
                    "run_id": Path(inner_path).name,
                    "normalized_output_directory": inner_path,
                    "mode": mode,
                    "evidence_scope": "target_single_vm_measured",
                    "scientific_manifest_digest": scientific,
                    "plan_digest": inner_plan["plan_digest"],
                    "machine_profile_digest": profile_digest,
                    "execution_stage": "target_inner_fit_projection_preflight",
                    "target_environment_digest": environment["environment_digest"],
                    "proposal_digest": proposal["proposal_digest"],
                    "authorization_digest": authorization["authorization_digest"],
                },
                "profile": profile,
                "plan": inner_plan,
                "environment": environment,
                "proposal": proposal,
                "authorization": authorization,
                "validation": {"valid": True},
            }
        )
    return outer, inner, outer_plan


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda o, _i: o[1]["manifest"]["authorization_provenance"].__setitem__(
                "git_commit", "e" * 40
            ),
            "mixed_source_git_commit",
        ),
        (
            lambda o, _i: o[1]["manifest"].__setitem__("mode", "cpu_parallel_1"),
            "outer_mode_scope_mismatch",
        ),
        (
            lambda o, _i: o[1]["manifest"]["authorization_provenance"].__setitem__(
                "execution_stage", "target_canary"
            ),
            "projection_execution_stage_mismatch",
        ),
        (
            lambda o, _i: o[1].__setitem__("run_directory", o[0]["run_directory"]),
            "projection_output_identity_mismatch",
        ),
        (
            lambda _o, i: i[1]["manifest"].__setitem__("mode", "cpu_parallel_1"),
            "inner_mode_scope_mismatch",
        ),
        (
            lambda _o, i: i[1].__setitem__(
                "run_directory", str((Path.cwd() / "relocated" / "inner-p2").resolve())
            ),
            "inner_run_identity_mismatch",
        ),
        (
            lambda _o, i: i[1].__setitem__(
                "authorization", deepcopy(i[0]["authorization"])
            ),
            "invalid_inner_authorization",
        ),
        (
            lambda _o, i: i[1].__setitem__(
                "environment", deepcopy(i[0]["environment"])
            ),
            "invalid_inner_authorization",
        ),
        (
            lambda _o, i: i[1].__setitem__("profile", deepcopy(i[0]["profile"])),
            "invalid_inner_authorization",
        ),
        (
            lambda o, _i: o[1].__setitem__("validation", {"valid": False}),
            "invalid_artifact_evidence",
        ),
    ],
)
def test_combined_projection_identity_is_closed_world(mutation, reason):
    outer, inner, plan = _combined_identity_fixture()
    assert validate_combined_projection_identity(outer, inner, plan)["valid"]
    changed_outer, changed_inner = deepcopy(outer), deepcopy(inner)
    mutation(changed_outer, changed_inner)
    report = validate_combined_projection_identity(changed_outer, changed_inner, plan)
    assert report["valid"] is False
    assert reason in report["reason_codes"]


def test_target_projection_requires_typed_inner_overhead_and_price_inputs():
    plan = _plan()
    records = _controlled_records(plan)
    validation = {
        "valid": True,
        "evidence_digest": sha256_canonical(records),
        "source_git_commit": "d" * 40,
        "source_artifact_hashes": ["e" * 64, "f" * 64],
    }
    inner = {
        "schema_version": 1,
        "artifact_type": "p7c4b2b_validated_inner_projection",
        "valid_for_combination": True,
        "selected_mode": "cpu_parallel_2",
        "conditional_elapsed_seconds": {
            "point": 100.0,
            "lower": 90.0,
            "upper": 110.0,
        },
        "source_evidence_digest": "a" * 64,
        "source_artifact_hashes": ["b" * 64, "c" * 64],
    }
    references = sorted(["e" * 64, "f" * 64, "b" * 64, "c" * 64])
    overhead = {
        "schema_version": 1,
        "artifact_type": "p7c4b2_outer_projection_overhead",
        "source_git_commit": "d" * 40,
        "locked_plan_digest": plan["plan_digest"],
        "selected_mode": "cpu_parallel_2",
        "method_identity": OVERHEAD_METHOD_IDENTITY,
        "event_counts": OVERHEAD_EVENT_COUNTS,
        "event_overhead_seconds": {"canonical_orchestration_and_io": 5.0},
        "outer_refits_parallel": True,
        "source_evidence_references": references,
        "source_evidence_digest": sha256_canonical(
            {"source_evidence_references": references}
        ),
        "reviewed_creation_timestamp": "2026-08-13T00:00:00Z",
    }
    overhead["artifact_digest"] = overhead_artifact_digest(overhead)
    price = {
        "price_per_hour": 0.26,
        "currency": "USD",
        "billing_unit": "vm_hour",
        "pricing_timestamp": "2026-08-13T00:00:00Z",
        "source": "operator_provider_quote",
        "vm_count": 1,
    }
    projection = project_validated(
        records,
        plan,
        artifact_validation=validation,
        execution_class="target_preflight",
        inner_projection=inner,
        overhead_mapping=overhead,
        price_input=price,
    )
    assert projection["execution_plan_eligible"] is True
    assert projection["cost_projection"]["currency"] == "USD"
    assert projection["scientific_projection_eligible"] is True
    assert projection["canonical_scientific_execution_authorized"] is False

    invalid = project_validated(
        records,
        plan,
        artifact_validation=validation,
        execution_class="target_preflight",
        inner_projection={**inner, "unknown": True},
        overhead_mapping=overhead,
        price_input=price,
    )
    assert invalid["execution_plan_eligible"] is False
    assert "inner_fit_evidence_missing" in invalid["reason_codes"]

    mutations = []
    missing = deepcopy(overhead)
    missing.pop("event_counts")
    mutations.append(missing)
    extra = {**overhead, "unknown": True}
    mutations.append(extra)
    wrong_event = deepcopy(overhead)
    wrong_event["event_counts"] = {"canonical_inner_fits": 54_000}
    wrong_event["artifact_digest"] = overhead_artifact_digest(wrong_event)
    mutations.append(wrong_event)
    wrong_source = deepcopy(overhead)
    wrong_source["source_git_commit"] = "e" * 40
    wrong_source["artifact_digest"] = overhead_artifact_digest(wrong_source)
    mutations.append(wrong_source)
    for numeric in (0.0, -1.0, float("nan"), float("inf")):
        changed = deepcopy(overhead)
        changed["event_overhead_seconds"]["canonical_orchestration_and_io"] = numeric
        mutations.append(changed)
    for changed in mutations:
        result = project_validated(
            records,
            plan,
            artifact_validation=validation,
            execution_class="target_preflight",
            inner_projection=inner,
            overhead_mapping=changed,
            price_input=price,
        )
        assert result["execution_plan_eligible"] is False
        assert "clean_overhead_measurement_missing" in result["reason_codes"]
