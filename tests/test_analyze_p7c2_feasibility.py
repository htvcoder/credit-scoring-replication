from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "analyze_p7c2_feasibility", Path("scripts/analyze_p7c2_feasibility.py")
)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyzer)


def payload(model: str = "random_forest") -> dict:
    return {
        "fit_id": "a",
        "plan_digest": analyzer.PLAN_DIGEST,
        "model_id": model,
        "dataset_id": "AC",
        "candidate_id": "rf_low" if model == "random_forest" else "xgb_low",
        "inner_fold_index": 0,
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "duration_seconds": 1.0,
        "process_cpu_seconds": 0.9,
        "process_rss_start_bytes": 10,
        "process_rss_peak_bytes": 20,
        "process_rss_delta_peak_bytes": 10,
        "configured_thread_count": 1,
        "effective_thread_count": 1,
        "git_provenance": {"git_head": "b" * 40, "working_tree": "clean"},
        "tree_method": "hist" if model == "xgboost" else None,
    }


def test_percentile_is_nearest_rank_and_rejects_bad_input():
    assert analyzer.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.9) == 5.0
    try:
        analyzer.percentile([], 0.9)
    except ValueError:
        pass
    else:
        raise AssertionError("empty percentile must fail")


def test_analyzer_reports_malformed_and_incomplete_run(tmp_path: Path):
    path = tmp_path / "fits" / "one"
    path.mkdir(parents=True)
    item = payload()
    del item["process_cpu_seconds"]
    (path / "result.json").write_text(json.dumps(item), encoding="utf-8")
    result = analyzer.analyze_run(tmp_path)
    assert result["valid"] is False
    assert any("missing required fields" in error for error in result["errors"])
    assert result["artifact_integrity"]["missing"] == 60
