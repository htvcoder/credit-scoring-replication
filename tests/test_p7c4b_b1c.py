import json

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
    assert seq_ids != par_ids  # smoke identity includes run ID
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
