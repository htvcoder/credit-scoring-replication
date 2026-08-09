import json

import pytest

from creditrep.experiments import p7c4b_mlp_benchmark as harness

PLAN = "configs/protocols/p7c/p7c4a_mlp_compute_benchmark_plan.yaml"


def plan():
    return harness.build_plan(PLAN, mode="cpu_sequential")


def _root(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir()
    (root / harness.ARTIFACT_ROOT).mkdir(parents=True)
    monkeypatch.setattr(harness, "_git_provenance", lambda _: {"git_head": "a" * 40, "working_tree": "clean"})
    return root


def test_interrupted_resume_is_idempotent_and_creates_marker_after_validation(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch); run = root / harness.ARTIFACT_ROOT / "smoke-resume"
    first = harness.resume_cpu_sequential(plan(), run, repo_root=root, max_fits=2, timeout_seconds=20, stop_after=1)
    assert first["executed"] == 1 and not (run / "COMPLETED.json").exists()
    second = harness.resume_cpu_sequential(plan(), run, repo_root=root, max_fits=2, timeout_seconds=20)
    assert second["skipped_on_resume"] == 1 and second["executed"] == 1
    assert (run / "COMPLETED.json").is_file()
    assert harness.validate_artifacts(run)["valid"]
    third = harness.resume_cpu_sequential(plan(), run, repo_root=root, max_fits=2, timeout_seconds=20)
    assert third["skipped_on_resume"] == 2 and third["executed"] == 0


def test_validator_detects_corruption_and_quarantines_reason(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch); run = root / harness.ARTIFACT_ROOT / "smoke-corrupt"
    harness.resume_cpu_sequential(plan(), run, repo_root=root, max_fits=1, timeout_seconds=20)
    result = next((run / "fits").glob("*/result.json"))
    payload = json.loads(result.read_text(encoding="utf-8")); payload["training_seed"] = -1
    result.write_text(json.dumps(payload), encoding="utf-8")
    report = harness.validate_artifacts(run)
    assert not report["valid"] and "artifact_digest_mismatch" in report["reason_codes"]
    quarantine = harness.quarantine_corrupt(run, report)
    assert quarantine and (quarantine / "reason.json").is_file()


@pytest.mark.parametrize("mutation,code", [
    (lambda run: (run / "staging").mkdir(), "stale_temporary_output"),
    (lambda run: (run / "fits" / "extra").mkdir(), "extra_logical_fit"),
])
def test_validator_reads_disk_not_summary(tmp_path, monkeypatch, mutation, code):
    root = _root(tmp_path, monkeypatch); run = root / harness.ARTIFACT_ROOT / "smoke-validator"
    harness.resume_cpu_sequential(plan(), run, repo_root=root, max_fits=1, timeout_seconds=20)
    mutation(run)
    assert code in harness.validate_artifacts(run)["reason_codes"]


def _valid_run(tmp_path, monkeypatch, name="smoke-mutation"):
    root = _root(tmp_path, monkeypatch); run = root / harness.ARTIFACT_ROOT / name
    harness.resume_cpu_sequential(plan(), run, repo_root=root, max_fits=1, timeout_seconds=20)
    return run


@pytest.mark.parametrize("field,value,code", [
    ("schema_version", 9, "unsupported_mode"),
    ("run_id", "other", "run_id_mismatch"),
    ("mode", "gpu_sequential", "unsupported_mode"),
    ("evidence_scope", "scientific", "incompatible_evidence_scope"),
    ("plan_digest", "0" * 64, "plan_digest_mismatch"),
    ("resolved_config_digest", "0" * 64, "manifest_config_mismatch"),
])
def test_manifest_contract_mutations_have_stable_codes(tmp_path, monkeypatch, field, value, code):
    run = _valid_run(tmp_path, monkeypatch)
    path = run / "run_manifest.json"; payload = json.loads(path.read_text(encoding="utf-8")); payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert code in harness.validate_artifacts(run)["reason_codes"]


@pytest.mark.parametrize("mutate,code", [
    (lambda run, fit: (run / "fits" / fit / "result.json").unlink(), "missing_result"),
    (lambda run, fit: (run / "fits" / fit / "COMPLETED.json").unlink(), "missing_fit_completion"),
    (lambda run, fit: (run / "fits" / fit / "attempts" / "attempt-1.json").write_text("{bad", encoding="utf-8"), "malformed_json"),
])
def test_per_fit_artifact_mutations_have_stable_codes(tmp_path, monkeypatch, mutate, code):
    run = _valid_run(tmp_path, monkeypatch); fit = next((run / "fits").iterdir()).name
    mutate(run, fit)
    assert code in harness.validate_artifacts(run)["reason_codes"]


def test_retry_sequence_timing_rss_and_marker_digest_are_validated(tmp_path, monkeypatch):
    run = _valid_run(tmp_path, monkeypatch); fit = next((run / "fits").iterdir()).name
    attempt = run / "fits" / fit / "attempts" / "attempt-1.json"
    payload = json.loads(attempt.read_text(encoding="utf-8")); payload.update(attempt=3, wall_clock_seconds=-1, peak_rss_bytes_process_tree="bad", completed_at="2000-01-01T00:00:00Z")
    attempt.write_text(json.dumps(payload), encoding="utf-8")
    marker = run / "COMPLETED.json"; marker.write_text(json.dumps({"run_id": run.name, "validation_report_digest": "bad"}), encoding="utf-8")
    codes = harness.validate_artifacts(run)["reason_codes"]
    assert {"duplicate_attempt", "attempt_identity_mismatch", "invalid_timing", "invalid_rss", "timestamp_order_mismatch", "premature_run_completion"} <= set(codes)


def test_quarantine_identity_does_not_collide_and_resume_halts(tmp_path, monkeypatch):
    run = _valid_run(tmp_path, monkeypatch); result = next((run / "fits").glob("*/result.json"))
    result.write_text("{bad", encoding="utf-8")
    report = harness.validate_artifacts(run); first = harness.quarantine_corrupt(run, report); second = harness.quarantine_corrupt(run, report)
    assert first != second and first.exists() and second.exists()
    with pytest.raises(harness.P7C4BBenchmarkError, match="corrupt artifacts"):
        harness.resume_cpu_sequential(plan(), run, repo_root=run.parents[2], max_fits=1, timeout_seconds=20)
