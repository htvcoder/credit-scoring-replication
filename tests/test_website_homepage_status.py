"""Regression guards for homepage status data derived from progress.yaml."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "website" / "content" / "progress.yaml"
HOMEPAGE_PATH = REPO_ROOT / "website" / "app" / "page.tsx"
CONTENT_PATH = REPO_ROOT / "website" / "lib" / "content.ts"
ROADMAP_PATH = REPO_ROOT / "website" / "app" / "tien-do" / "page.tsx"


def load_status() -> dict:
    return yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))


def homepage_phases(status: dict) -> list[dict]:
    next_phase = next(
        phase
        for phase in status["phases"]
        if phase["numeric_id"] == status["project"]["next_phase"]
    )
    return [
        phase
        for phase in status["phases"]
        if phase["status"] == "completed" or phase["id"] == next_phase["id"]
    ]


def test_homepage_status_data_contains_completed_phases_and_next_phase() -> None:
    status = load_status()

    assert status["project"]["last_completed_phase"] == 6
    assert status["project"]["current_phase"] == status["project"]["next_phase"] == 7
    assert [(phase["id"], phase["status"]) for phase in homepage_phases(status)] == [
        ("Phase 0", "completed"),
        ("Phase 1", "completed"),
        ("Phase 2", "completed"),
        ("Phase 3", "completed"),
        ("Phase 4", "completed"),
        ("Phase 5", "completed"),
        ("Phase 6", "completed"),
        ("Phase 7", "in_progress"),
    ]


def test_homepage_and_roadmap_use_the_source_status_summary() -> None:
    status = load_status()
    summary = status["project"]["current_status_summary"]
    assert "P7C.4A: Completed" in summary
    assert "P7C.4B blocked awaiting human approval" in summary
    assert "benchmark chưa chạy" in summary
    assert "DR-P7C-03/04 chưa approved" in summary

    phase7 = next(phase for phase in status["phases"] if phase["numeric_id"] == 7)
    checkpoints = {item["id"]: item["status"] for item in phase7["checkpoints"]}
    assert checkpoints["P7C.1"] == "completed"
    assert checkpoints["P7C.2"] == "completed"
    assert checkpoints["P7C.2.1"] == "completed"
    assert checkpoints["P7C.2.2"] == "completed"
    assert checkpoints["P7C.2.3"] == "completed"
    assert checkpoints["P7C.3"] == "completed"
    assert checkpoints["P7C.4A"] == "completed"
    assert checkpoints["P7C.4B"] == "blocked"

    homepage = HOMEPAGE_PATH.read_text(encoding="utf-8")
    content = CONTENT_PATH.read_text(encoding="utf-8")
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")

    assert "getHomepageStatusData" in homepage
    assert "homepageStatus.phases" in homepage
    assert "slice(0, 5)" not in homepage
    assert "current_status_summary" in content
    assert "current_status_summary" in roadmap
    assert "Phase 4 đã hoàn thành metric validation" not in homepage
    assert "Phase 5 - Mô hình truyền thống và ensemble là bước tiếp" not in homepage
