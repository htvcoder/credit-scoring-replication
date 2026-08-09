from copy import deepcopy
from pathlib import Path
import json

from creditrep.experiments import p7c4b_mlp_benchmark as harness

PLAN = "configs/protocols/p7c/p7c4a_mlp_compute_benchmark_plan.yaml"

def plan(): return harness.build_plan(PLAN, mode="cpu_sequential")

def test_plan_order_partition_seed_and_identity():
    value = plan(); assert harness.validate_plan(value)["measured_logical_fits"] == 36
    assert [fit["training_seed"] for fit in value["measured_fits"][:3]] == [1701, 1702, 1703]
    assert all((fit["outer_repeat_index"], fit["outer_fold_index"], fit["inner_fold_index"]) == (0, 0, 0) for fit in value["measured_fits"])
    changed = deepcopy(value["measured_fits"][0]); changed["execution_id"] = harness.execution_id(changed["logical_fit_id"], "gpu_sequential", "measured")
    assert changed["logical_fit_id"] == value["measured_fits"][0]["logical_fit_id"]
    assert changed["execution_id"] != value["measured_fits"][0]["execution_id"]

def test_fixture_success_atomic_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir(); artifact_root = root / harness.ARTIFACT_ROOT; artifact_root.mkdir(parents=True)
    monkeypatch.setattr(harness, "_git_provenance", lambda _: {"git_head": "a"*40, "working_tree": "dirty"})
    result = harness.run_cpu_sequential(plan(), artifact_root / "smoke-test", repo_root=root, fixture=True, max_fits=1, timeout_seconds=20)
    assert result["completed"] == 1 and result["failed"] == 0
    fit_dir = next((artifact_root / "smoke-test" / "fits").iterdir())
    record = json.loads((fit_dir / "result.json").read_text(encoding="utf-8"))
    assert (fit_dir / "attempts" / "attempt-1.json").is_file()
    assert (fit_dir / "COMPLETED.json").is_file()
    assert record["peak_rss_bytes_process_tree"] >= 0
    assert not list((artifact_root / "smoke-test").rglob("*.tmp"))

def test_transient_retry_and_deterministic_failure(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir(); (root / harness.ARTIFACT_ROOT).mkdir(parents=True)
    monkeypatch.setattr(harness, "_git_provenance", lambda _: {"git_head": "a"*40, "working_tree": "dirty"})
    value = plan(); value["measured_fits"][0]["fixture_behavior"] = "deterministic"
    result = harness.run_cpu_sequential(value, root / harness.ARTIFACT_ROOT / "smoke-fail", repo_root=root, fixture=True, max_fits=1, timeout_seconds=20)
    assert result["failed"] == 1
    fit_dir = next((root / harness.ARTIFACT_ROOT / "smoke-fail" / "fits").iterdir())
    failure = json.loads((fit_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["reason_code"] == "deterministic_fit_failure" and failure["attempt"] == 1

def test_timeout_and_transient_retry_exhaustion(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir(); (root / harness.ARTIFACT_ROOT).mkdir(parents=True)
    monkeypatch.setattr(harness, "_git_provenance", lambda _: {"git_head": "a"*40, "working_tree": "dirty"})
    timeout_plan = plan(); timeout_plan["measured_fits"][0] |= {"fixture_behavior": "sleep", "fixture_sleep_seconds": 2}
    timed = harness.run_cpu_sequential(timeout_plan, root / harness.ARTIFACT_ROOT / "smoke-timeout", repo_root=root, fixture=True, max_fits=1, timeout_seconds=.1)
    assert timed["failed"] == 1
    transient = plan(); transient["measured_fits"][0]["fixture_behavior"] = "transient"
    result = harness.run_cpu_sequential(transient, root / harness.ARTIFACT_ROOT / "smoke-retry", repo_root=root, fixture=True, max_fits=1, timeout_seconds=20)
    fit_dir = next((root / harness.ARTIFACT_ROOT / "smoke-retry" / "fits").iterdir())
    failure = json.loads((fit_dir / "failure.json").read_text(encoding="utf-8"))
    assert result["failed"] == 1 and failure["reason_code"] == "retry_exhausted" and failure["attempt"] == 2
    assert len(list((fit_dir / "attempts").glob("*.json"))) == 2
