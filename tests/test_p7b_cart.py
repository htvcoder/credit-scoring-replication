from pathlib import Path
import json
import threading
import time
from copy import deepcopy

import pytest

from creditrep.experiments import p7b_cart
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
