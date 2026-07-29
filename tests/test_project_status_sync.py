from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sync_project_status.py"
spec = importlib.util.spec_from_file_location("sync_project_status", MODULE_PATH)
sync = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sync
spec.loader.exec_module(sync)


def load_status() -> dict:
    return sync.read_status()


def assert_invalid(status: dict, expected: str) -> None:
    result = sync.validate_status_schema(status, check_tags=False)
    assert any(expected in error for error in result.errors), result.errors


def test_current_repository_state_passes_without_tag_existence() -> None:
    status = load_status()
    assert sync.validate_status_schema(status, check_tags=False).ok
    assert sync.validate_no_local_absolute_paths(status).ok


def test_invalid_status_enum_fails() -> None:
    status = load_status()
    status["phases"][0]["status"] = "done"
    assert_invalid(status, "invalid status")


def test_duplicate_phase_id_fails() -> None:
    status = load_status()
    status["phases"][1]["numeric_id"] = 0
    assert_invalid(status, "Duplicate phase id")


def test_duplicate_checkpoint_id_fails() -> None:
    status = load_status()
    phase3 = status["phases"][3]
    phase3["checkpoints"].append(copy.deepcopy(phase3["checkpoints"][0]))
    assert_invalid(status, "duplicate checkpoint id")


def test_two_in_progress_phases_fail() -> None:
    status = load_status()
    status["phases"][4]["status"] = "in_progress"
    status["phases"][5]["status"] = "in_progress"
    assert_invalid(status, "Only one phase may be in_progress")


def test_two_next_phases_fail() -> None:
    status = load_status()
    status["phases"][4]["status"] = "next"
    status["phases"][5]["status"] = "next"
    assert_invalid(status, "Only one phase may be next")


def test_completed_phase_missing_tag_fails() -> None:
    status = load_status()
    del status["phases"][3]["tag"]
    assert_invalid(status, "must declare a milestone tag")


def test_completed_phase_with_pending_required_checkpoint_fails() -> None:
    status = load_status()
    status["phases"][3]["checkpoints"][2]["status"] = "planned"
    assert_invalid(status, "non-completed checkpoint")


def test_phase3_stale_website_text_fails(tmp_path: Path) -> None:
    stale = tmp_path / "page.tsx"
    stale.write_text("Current status: Phase 3: In Progress\n", encoding="utf-8")
    result = sync.validate_current_status_text([stale])
    assert any("Stale current status" in error for error in result.errors)


def test_phase4_planned_still_fails() -> None:
    status = load_status()
    status["phases"][4]["status"] = "planned"
    assert_invalid(status, "Phase 4 must be next, in_progress, or completed")


def test_phase4_completed_requires_phase5_next_and_project_pointer_updates() -> None:
    status = load_status()
    status["project"]["last_completed_phase"] = 4
    status["project"]["current_phase"] = 5
    status["project"]["next_phase"] = 5
    status["phases"][4]["status"] = "completed"
    status["phases"][4]["tag"] = "p4-metric-validation-complete"
    status["phases"][4]["checkpoints"][2]["status"] = "completed"
    status["phases"][5]["status"] = "next"
    status["phases"][5].pop("tag", None)
    status["phases"][5]["checkpoints"][2]["status"] = "planned"
    status["phases"][6]["status"] = "planned"
    assert sync.validate_status_schema(status, check_tags=False).ok


def test_phase5_completed_allows_phase6_in_progress_with_p6a_complete() -> None:
    status = load_status()
    assert status["project"]["last_completed_phase"] == 5
    assert status["project"]["current_phase"] == status["project"]["next_phase"] == 6
    assert status["phases"][5]["status"] == "completed"
    assert status["phases"][5]["tag"] == "p5-classical-replication-complete"
    assert status["phases"][6]["status"] == "in_progress"
    checkpoints = {item["id"]: item["status"] for item in status["phases"][6]["checkpoints"]}
    assert checkpoints == {"P6A": "completed", "P6B": "next", "P6C": "planned"}


def test_website_derivative_matches_source_policy() -> None:
    status = load_status()
    phase3 = sync.phases_by_number(status)[3]
    assert phase3["status"] == "completed"
    assert phase3["tag"] == "p3-leakage-safe-preprocessing-complete"


def test_readme_docs_managed_section_stale_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    status = load_status()
    doc = tmp_path / "README.md"
    doc.write_text(f"{sync.BEGIN}\nstale\n{sync.END}\n", encoding="utf-8")
    monkeypatch.setattr(sync, "MANAGED_DOCS", [doc])
    expected = sync.expected_managed_outputs(status)
    assert expected[doc] != doc.read_text(encoding="utf-8")


def test_missing_required_phase_fails() -> None:
    status = load_status()
    status["phases"] = [phase for phase in status["phases"] if phase.get("numeric_id") != 4]
    assert_invalid(status, "Missing required Phase 4")


def test_check_mode_does_not_modify_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path
    status_path = repo / "website" / "content" / "progress.yaml"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(yaml.safe_dump(load_status(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    doc = repo / "README.md"
    doc.write_text(f"{sync.BEGIN}\nstale\n{sync.END}\n", encoding="utf-8")
    monkeypatch.setattr(sync, "REPO_ROOT", repo)
    monkeypatch.setattr(sync, "STATUS_PATH", status_path)
    monkeypatch.setattr(sync, "MANAGED_DOCS", [doc])
    monkeypatch.setattr(sync, "WEBSITE_STATUS_FILES", [status_path])
    monkeypatch.setattr(sync, "DOC_STATUS_FILES", [doc])
    before = doc.read_text(encoding="utf-8")
    assert sync.run(check=True, skip_tag_existence=True) == 1
    assert doc.read_text(encoding="utf-8") == before


def test_sync_mode_only_updates_managed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path
    status_path = repo / "website" / "content" / "progress.yaml"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(yaml.safe_dump(load_status(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    doc = repo / "README.md"
    doc.write_text(f"intro\n{sync.BEGIN}\nstale\n{sync.END}\noutro\n", encoding="utf-8")
    untouched = repo / "NOTES.md"
    untouched.write_text("manual prose\n", encoding="utf-8")
    monkeypatch.setattr(sync, "REPO_ROOT", repo)
    monkeypatch.setattr(sync, "STATUS_PATH", status_path)
    monkeypatch.setattr(sync, "MANAGED_DOCS", [doc])
    monkeypatch.setattr(sync, "WEBSITE_STATUS_FILES", [status_path])
    monkeypatch.setattr(sync, "DOC_STATUS_FILES", [doc])
    assert sync.run(check=False, skip_tag_existence=True) == 0
    assert untouched.read_text(encoding="utf-8") == "manual prose\n"
    assert "intro" in doc.read_text(encoding="utf-8")
    assert "outro" in doc.read_text(encoding="utf-8")


def test_generated_output_is_deterministic() -> None:
    status = load_status()
    assert sync.render_managed_section(status) == sync.render_managed_section(copy.deepcopy(status))


def test_does_not_depend_on_raw_data() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "data/raw" not in source
    assert "data\\raw" not in source


def test_rejects_local_absolute_path() -> None:
    status = load_status()
    status["project"]["note"] = r"C:\local\secret"
    result = sync.validate_no_local_absolute_paths(status)
    assert result.errors


def test_phase_status_update_updates_rendered_status() -> None:
    status = load_status()
    status["phases"][4]["status"] = "in_progress"
    rendered = sync.render_managed_section(status)
    assert "| Phase 4 - Metric validation | In Progress | p4-metric-validation-complete |" in rendered


def test_historical_text_is_not_misread_as_stale(tmp_path: Path) -> None:
    doc = tmp_path / "history.md"
    doc.write_text("Historical note: Phase 3: Next was true in an old plan.\n", encoding="utf-8")
    assert sync.validate_current_status_text([doc]).ok


def test_completed_tag_name_is_validated() -> None:
    status = load_status()
    status["phases"][3]["tag"] = "bad tag"
    assert_invalid(status, "invalid format")


def test_shallow_clone_tag_check_policy_skips_existence(monkeypatch: pytest.MonkeyPatch) -> None:
    status = load_status()
    monkeypatch.setattr(sync, "git_tag_exists", lambda tag: False)
    assert not sync.validate_status_schema(status, check_tags=True).ok
    assert sync.validate_status_schema(status, check_tags=False).ok
