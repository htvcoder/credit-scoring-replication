import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from creditrep.experiments import p7c4b_mlp_benchmark as harness

PLAN = "configs/protocols/p7c/p7c4a_mlp_compute_benchmark_plan.yaml"


def _root(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir(); (root / harness.ARTIFACT_ROOT).mkdir(parents=True)
    monkeypatch.setattr(harness, "_git_provenance", lambda _: {"git_head": "a" * 40, "working_tree": "clean"})
    return root


def test_parallel_uses_same_logical_workload_and_two_worker_cap(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch); seq_plan = harness.build_plan(PLAN, mode="cpu_sequential"); par_plan = harness.build_plan(PLAN, mode="cpu_parallel_2")
    seq = root / harness.ARTIFACT_ROOT / "smoke-seq"; par = root / harness.ARTIFACT_ROOT / "smoke-par"
    harness.resume_cpu_sequential(seq_plan, seq, repo_root=root, max_fits=4, timeout_seconds=20)
    result = harness.resume_cpu_parallel_2(par_plan, par, repo_root=root, max_fits=4, timeout_seconds=20)
    assert result["validation"]["valid"] and result["max_observed_active_workers"] <= 2
    seq_ids = [x["logical_fit_id"] for x in json.loads((seq / "run_manifest.json").read_text())["expected_fits"]]
    par_ids = [x["logical_fit_id"] for x in json.loads((par / "run_manifest.json").read_text())["expected_fits"]]
    assert seq_ids == par_ids
    assert {x["training_seed"] for x in json.loads((seq / "run_manifest.json").read_text())["expected_fits"]} == {x["training_seed"] for x in json.loads((par / "run_manifest.json").read_text())["expected_fits"]}
    records = list((par / "fits").glob("*/result.json")); assert len(records) == 4
    assert all(json.loads(x.read_text())["process_tree_telemetry"]["max_active_workers"] <= 2 for x in records)


def test_parallel_interruption_resume_and_corruption_policy(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch); plan = harness.build_plan(PLAN, mode="cpu_parallel_2"); run = root / harness.ARTIFACT_ROOT / "smoke-par-resume"
    first = harness.resume_cpu_parallel_2(plan, run, repo_root=root, max_fits=3, timeout_seconds=20, stop_after=1)
    assert first["executed"] == 1 and not (run / "COMPLETED.json").exists()
    second = harness.resume_cpu_parallel_2(plan, run, repo_root=root, max_fits=3, timeout_seconds=20)
    assert second["skipped_on_resume"] == 1 and second["validation"]["valid"]
    corrupt = next((run / "fits").glob("*/result.json")); corrupt.write_text("{bad", encoding="utf-8")
    assert "malformed_json" in harness.validate_artifacts(run)["reason_codes"]
    with pytest.raises(harness.P7C4BBenchmarkError, match="corrupt artifacts"):
        harness.resume_cpu_parallel_2(plan, run, repo_root=root, max_fits=3, timeout_seconds=20)


def test_parallel_completion_order_inversion_keeps_canonical_summary_order(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch); plan = harness.build_plan(PLAN, mode="cpu_parallel_2")
    plan["measured_fits"][0].update(fixture_behavior="sleep", fixture_sleep_seconds=.20)
    run = root / harness.ARTIFACT_ROOT / "smoke-inversion"
    result = harness.resume_cpu_parallel_2(plan, run, repo_root=root, max_fits=2, timeout_seconds=20)
    assert result["validation"]["valid"] and result["max_observed_active_workers"] == 2
    manifest = json.loads((run / "run_manifest.json").read_text()); summary = json.loads((run / "summary.json").read_text())
    records = [json.loads(p.read_text()) for p in (run / "fits").glob("*/result.json")]
    completion_order = [x["logical_fit_id"] for x in sorted(records, key=lambda x: x["completed_at"])]
    assert completion_order != [x["logical_fit_id"] for x in manifest["expected_fits"]]
    assert summary["terminal_logical_fit_ids"] == [x["logical_fit_id"] for x in manifest["expected_fits"]]


def test_parallel_mixed_failure_preserves_success_and_has_no_run_marker(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch); plan = harness.build_plan(PLAN, mode="cpu_parallel_2")
    plan["measured_fits"][1]["fixture_behavior"] = "deterministic"
    run = root / harness.ARTIFACT_ROOT / "smoke-mixed-failure"
    result = harness.resume_cpu_parallel_2(plan, run, repo_root=root, max_fits=2, timeout_seconds=20)
    assert not result["validation"]["valid"] and not (run / "COMPLETED.json").exists()
    results = list((run / "fits").glob("*/result.json")); failures = list((run / "fits").glob("*/failure.json"))
    assert len(results) == len(failures) == 1
    failure = json.loads(failures[0].read_text()); assert failure["reason_code"] == "deterministic_fit_failure" and failure["process_exit_code"] == 0


def test_cli_parallel_spawn_and_exit_paths():
    repo = Path(__file__).resolve().parents[1]; run_name = f"smoke-cli-{uuid.uuid4().hex[:8]}"; output = f"artifacts/p7c4b-mlp-compute-benchmark/{run_name}"
    base = [sys.executable, "-m", "creditrep.experiments.p7c4b_cli"]
    run = subprocess.run([*base, "resume", "--mode", "cpu_parallel_2", "--non-canonical-smoke", "--output-dir", output, "--max-fits", "2", "--timeout-seconds", "30"], cwd=repo, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stderr + run.stdout
    validated = subprocess.run([*base, "validate-artifacts", "--output-dir", output], cwd=repo, capture_output=True, text=True, timeout=30)
    assert validated.returncode == 0, validated.stderr + validated.stdout
    resumed = subprocess.run([*base, "resume", "--mode", "cpu_parallel_2", "--non-canonical-smoke", "--output-dir", output, "--max-fits", "2"], cwd=repo, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0 and '"executed": 0' in resumed.stdout
    missing = subprocess.run([*base, "validate-artifacts", "--output-dir", f"artifacts/p7c4b-mlp-compute-benchmark/missing-{uuid.uuid4().hex[:8]}"], cwd=repo, capture_output=True, text=True, timeout=30)
    unsupported = subprocess.run([*base, "plan", "--mode", "gpu_sequential"], cwd=repo, capture_output=True, text=True, timeout=30)
    assert missing.returncode != 0 and unsupported.returncode != 0
