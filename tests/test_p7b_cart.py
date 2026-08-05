from pathlib import Path
import json
import threading
import time
from copy import deepcopy

import pytest

from creditrep.experiments import p7b_cart
from creditrep.experiments import p7b_cli
from creditrep.datasets.registry import find_repo_root
from creditrep.protocols.p7a import effective_min_samples_leaf


def _root() -> Path:
    return find_repo_root()


@pytest.fixture(scope="module")
def plan() -> dict:
    root = _root()
    return p7b_cart.build_plan(
        root / "configs/protocols/p7a/p7a_candidate_manifest.yaml", repo_root=root
    )


def test_p7b_plan_contract_from_manifest(plan: dict):
    report = p7b_cart.validate_plan(plan)
    assert report["total_fits"] == report["unique_fit_ids"] == 60
    assert report["per_dataset"] == {"AC": 20, "HMEQ": 20, "GMC": 20}
    assert all("\\" not in item["artifact_path"] for item in plan["fits"])
    assert all(
        item["effective_min_samples_leaf"]
        == effective_min_samples_leaf(
            item["parameters"]["min_samples_leaf"], item["inner_training_rows"]
        )
        for item in plan["fits"]
    )
    assert (
        plan["candidate_selection"] == "none" and not plan["outer_selected_model_refit"]
    )


def test_p7b_plan_rejects_duplicate_and_outer_change(plan: dict):
    plan = deepcopy(plan)
    plan["fits"][1]["fit_id"] = plan["fits"][0]["fit_id"]
    with pytest.raises(p7b_cart.P7BContractError, match="duplicate"):
        p7b_cart.validate_plan(plan)


def test_render_plan_is_machine_readable(plan: dict, tmp_path: Path):
    p7b_cart.render_plan(plan, tmp_path)
    assert (
        json.loads((tmp_path / "validator.json").read_text(encoding="utf-8"))["valid"]
        is True
    )


def test_sanitized_exception_removes_backslashes():
    saved = p7b_cart._safe_exception(OSError(r"C:\\secret\\token"))
    assert "\\" not in saved["message"]


def test_process_rss_sampler_schema_and_cleanup():
    sampler = p7b_cart.ProcessRssSampler(interval_seconds=0.001)
    sampler.start()
    time.sleep(0.01)
    telemetry = sampler.stop()
    assert telemetry["process_rss_start_bytes"] >= 0
    assert telemetry["process_rss_peak_bytes"] >= telemetry["process_rss_start_bytes"]
    assert telemetry["process_rss_delta_peak_bytes"] == max(
        0,
        telemetry["process_rss_peak_bytes"] - telemetry["process_rss_start_bytes"],
    )
    assert telemetry["process_rss_sampling_interval_seconds"] == 0.001
    assert telemetry["process_id"] > 0
    assert telemetry["child_processes_included"] is False
    assert "psutil.Process.memory_info().rss" in telemetry["measurement_method"]
    assert sampler._thread is not None and not sampler._thread.is_alive()
    assert not any(
        thread.name == "p7b-rss-sampler" and thread.is_alive()
        for thread in threading.enumerate()
    )


def _valid_training_artifacts(plan: dict, root: Path) -> None:
    """Create a synthetic completed run without fitting a model."""
    snapshot = {key: plan[key] for key in plan if key != "fits"}
    p7b_cart._write(root / "plan.json", plan)
    p7b_cart._write(root / "config_snapshot.json", snapshot)
    p7b_cart._write(
        root / "environment.json",
        {
            "schema_version": 1,
            "artifact_kind": "training_run",
            "git_head": "a" * 40,
            "working_tree": "clean",
            "working_tree_details": {"is_dirty": False, "porcelain_v1": []},
        },
    )
    for fit in plan["fits"]:
        p7b_cart._write(
            root / fit["artifact_path"],
            {
                **p7b_cart.P7B_FLAGS,
                **{
                    key: fit[key]
                    for key in (
                        "fit_id",
                        "dataset_id",
                        "candidate_id",
                        "inner_fold_index",
                        "derived_seed",
                    )
                },
                "run_config_hash": plan["run_config_hash"],
                "status": "completed",
                "attempt_count": 1,
                "elapsed_wall_seconds": 0.1,
                "rows": 1,
                "feature_count": 1,
                "python_tracemalloc_peak_bytes": 0,
                "artifact_bytes": 0,
                "process_rss_start_bytes": 1,
                "process_rss_peak_bytes": 2,
                "process_rss_delta_peak_bytes": 1,
            },
        )
    p7b_cart._write(
        root / "engineering_summary.json",
        {
            "planned": 60,
            "completed": 60,
            "failed": 0,
            "pending": 0,
            "completion_status": "completed",
        },
    )


def test_artifact_validator_accepts_synthetic_completed_run(plan: dict, tmp_path: Path):
    _valid_training_artifacts(plan, tmp_path)
    report = p7b_cart.validate_artifacts(plan, tmp_path)
    assert report["valid"] is True
    assert report["training_artifacts_validated"] is True


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_result", "missing_fit_output"),
        ("bad_git", "invalid_git_head"),
        ("bad_rss", "rss_invariant"),
        ("bad_summary", "summary_count_mismatch"),
        ("result_and_failure", "ambiguous_fit_state"),
    ],
)
def test_artifact_validator_rejects_corruption(
    plan: dict, tmp_path: Path, mutation: str, expected: str
):
    _valid_training_artifacts(plan, tmp_path)
    fit = plan["fits"][0]
    result = tmp_path / fit["artifact_path"]
    if mutation == "missing_result":
        result.unlink()
    elif mutation == "bad_git":
        env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
        env["git_head"] = None
        p7b_cart._write(tmp_path / "environment.json", env)
    elif mutation == "bad_rss":
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["process_rss_peak_bytes"] = 0
        p7b_cart._write(result, payload)
    elif mutation == "bad_summary":
        summary = json.loads(
            (tmp_path / "engineering_summary.json").read_text(encoding="utf-8")
        )
        summary["completed"] = 59
        p7b_cart._write(tmp_path / "engineering_summary.json", summary)
    else:
        failure = {"status": "failed", "failure": {"type": "OSError", "message": "x"}}
        p7b_cart._write(result.with_name("failure.json"), failure)
    report = p7b_cart.validate_artifacts(plan, tmp_path)
    assert report["valid"] is False
    assert expected in {error["code"] for error in report["errors"]}


def test_validator_ignores_stale_validator_record_and_plan_only_is_distinct(
    plan: dict, tmp_path: Path
):
    p7b_cart.render_plan(plan, tmp_path, repo_root=_root())
    p7b_cart._write(tmp_path / "validator.json", {"valid": True, "stale": True})
    report = p7b_cart.validate_artifacts(plan, tmp_path)
    assert report["valid"] is True
    assert report["artifact_kind"] == "plan_only_dry_run"
    assert report["training_artifacts_validated"] is False


def test_run_fails_before_training_when_git_provenance_is_unavailable(
    plan: dict, tmp_path: Path, monkeypatch
):
    def fail(_: Path, *, required: bool):
        assert required is True
        raise p7b_cart.GitProvenanceError(
            "Git ownership/safe-directory protection rejected this repository"
        )

    monkeypatch.setattr(p7b_cart, "capture_git_provenance", fail)
    with pytest.raises(p7b_cart.GitProvenanceError, match="safe-directory"):
        p7b_cart.run(plan, tmp_path / "run", repo_root=_root())
    assert not (tmp_path / "run" / "plan.json").exists()


def test_capture_git_provenance_rejects_non_sha(monkeypatch, tmp_path: Path):
    class Result:
        returncode = 0
        stdout = "unknown\n"
        stderr = ""

    monkeypatch.setattr(p7b_cart, "_git", lambda *_: Result())
    with pytest.raises(p7b_cart.GitProvenanceError, match="40-character"):
        p7b_cart.capture_git_provenance(tmp_path, required=True)


def test_validate_artifacts_cli_returns_nonzero_for_invalid_tree(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "sys.argv", ["p7b_cli", "validate-artifacts", "--output-dir", str(tmp_path)]
    )
    assert p7b_cli.main() == 2


def test_partial_training_run_without_summary_is_resumable(plan: dict, tmp_path: Path):
    _valid_training_artifacts(plan, tmp_path)
    (tmp_path / "engineering_summary.json").unlink()
    (tmp_path / plan["fits"][0]["artifact_path"]).unlink()
    report = p7b_cart.validate_artifacts(plan, tmp_path)
    assert report["valid"] is True
    assert report["completion_status"] == "incomplete"
    assert report["resumable"] is True
    assert report["completed"] == 59
    assert report["pending"] == 1


def test_interrupted_training_summary_is_resumable(plan: dict, tmp_path: Path):
    _valid_training_artifacts(plan, tmp_path)
    (tmp_path / plan["fits"][0]["artifact_path"]).unlink()
    summary = json.loads(
        (tmp_path / "engineering_summary.json").read_text(encoding="utf-8")
    )
    summary.update({"completed": 59, "pending": 1, "completion_status": "interrupted"})
    p7b_cart._write(tmp_path / "engineering_summary.json", summary)
    report = p7b_cart.validate_artifacts(plan, tmp_path)
    assert report["valid"] is True
    assert report["resumable"] is True


def test_resume_rejects_changed_provenance_before_loading_data(
    plan: dict, tmp_path: Path, monkeypatch
):
    _valid_training_artifacts(plan, tmp_path)
    (tmp_path / plan["fits"][0]["artifact_path"]).unlink()
    summary = json.loads(
        (tmp_path / "engineering_summary.json").read_text(encoding="utf-8")
    )
    summary.update({"completed": 59, "pending": 1, "completion_status": "interrupted"})
    p7b_cart._write(tmp_path / "engineering_summary.json", summary)
    monkeypatch.setattr(
        p7b_cart,
        "capture_git_provenance",
        lambda *_args, **_kwargs: {
            "git_head": "b" * 40,
            "working_tree": "clean",
            "working_tree_details": {"is_dirty": False, "porcelain_v1": []},
        },
    )
    with pytest.raises(p7b_cart.P7BContractError, match="different Git provenance"):
        p7b_cart.run(plan, tmp_path, repo_root=_root(), resume=True)


def test_run_interrupt_before_first_fit_writes_clear_partial_state(
    plan: dict, tmp_path: Path, monkeypatch
):
    output_dir = tmp_path / "interrupted-run"
    messages: list[str] = []
    monkeypatch.setattr(
        p7b_cart,
        "capture_git_provenance",
        lambda *_args, **_kwargs: {
            "git_head": "a" * 40,
            "working_tree": "clean",
            "working_tree_details": {"is_dirty": False, "porcelain_v1": []},
        },
    )
    monkeypatch.setattr(p7b_cart, "_progress", messages.append)
    monkeypatch.setattr(
        p7b_cart,
        "load_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        p7b_cart.run(plan, output_dir, repo_root=_root())
    summary = json.loads(
        (output_dir / "engineering_summary.json").read_text(encoding="utf-8")
    )
    assert summary["completion_status"] == "interrupted"
    assert summary["completed"] == summary["failed"] == 0
    assert summary["pending"] == 60
    assert not list((output_dir / "fits").rglob("*.json"))
    assert "Git provenance validated" in messages
    assert "Loading dataset AC" in messages
