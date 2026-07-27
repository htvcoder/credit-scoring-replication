"""Synchronize and validate project phase status surfaces."""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "website" / "content" / "progress.yaml"
MANAGED_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "EXPERIMENT_IMPLEMENTATION_PLAN.md",
]
GENERATED_NOTE = "Generated from website/content/progress.yaml. Do not edit manually."
BEGIN = "<!-- PROJECT_STATUS:BEGIN -->"
END = "<!-- PROJECT_STATUS:END -->"
ALLOWED_STATUSES = {"planned", "next", "in_progress", "completed", "blocked", "deferred"}
TAG_RE = re.compile(r"^p\d+-[a-z0-9]+(?:-[a-z0-9]+)*$")
STALE_CURRENT_PATTERNS = [
    re.compile(r"Phase\s*3\s*:\s*(Pending|In Progress|Next)", re.IGNORECASE),
    re.compile(r"Phase\s*3\s+(?:is|là)\s+(?:Pending|In Progress|Next)", re.IGNORECASE),
    re.compile(r"P3C\s*:\s*Pending", re.IGNORECASE),
    re.compile(r"Next phase\s*:\s*Phase\s*3", re.IGNORECASE),
    re.compile(r"Phase 3 là bước tiếp theo", re.IGNORECASE),
    re.compile(r"bắt đầu \*\*Phase 3", re.IGNORECASE),
]
WEBSITE_STATUS_FILES = [
    REPO_ROOT / "website" / "content" / "progress.yaml",
    REPO_ROOT / "website" / "content" / "methods.md",
    REPO_ROOT / "website" / "app" / "page.tsx",
    REPO_ROOT / "website" / "app" / "tien-do" / "page.tsx",
    REPO_ROOT / "website" / "app" / "ket-qua" / "page.tsx",
]
DOC_STATUS_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "EXPERIMENT_IMPLEMENTATION_PLAN.md",
    REPO_ROOT / "docs" / "EXPERIMENT_FEASIBILITY_ASSESSMENT.md",
]


@dataclass
class ValidationResult:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, message: str) -> None:
        self.errors.append(message)


def read_status(path: Path = STATUS_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def status_label(status: str) -> str:
    return {
        "planned": "Planned",
        "next": "Next",
        "in_progress": "In Progress",
        "completed": "Completed",
        "blocked": "Blocked",
        "deferred": "Deferred",
    }[status]


def phase_number(phase: dict[str, Any]) -> int:
    if "numeric_id" in phase:
        return int(phase["numeric_id"])
    match = re.fullmatch(r"Phase\s+(\d+)", str(phase.get("id", "")))
    if not match:
        raise ValueError(f"Invalid phase id: {phase.get('id')}")
    return int(match.group(1))


def phases_by_number(status: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {phase_number(phase): phase for phase in status.get("phases", [])}


def validate_status_schema(status: dict[str, Any], check_tags: bool = True) -> ValidationResult:
    result = ValidationResult([])
    if status.get("schema_version") != 1:
        result.add("schema_version must be 1.")

    project = status.get("project")
    if not isinstance(project, dict):
        result.add("project metadata is required.")
        project = {}
    for field in ("last_completed_phase", "current_phase", "next_phase", "updated_at"):
        if field not in project:
            result.add(f"project.{field} is required.")

    declared_enum = set(status.get("status_enum") or [])
    if declared_enum != ALLOWED_STATUSES:
        result.add("status_enum must exactly declare the supported statuses.")

    phases = status.get("phases")
    if not isinstance(phases, list) or not phases:
        result.add("phases must be a non-empty list.")
        return result

    seen_numbers: set[int] = set()
    next_count = 0
    in_progress_count = 0
    completed_numbers: list[int] = []

    for index, phase in enumerate(phases):
      # Keep this loop intentionally explicit; error messages are CI-facing.
        try:
            number = phase_number(phase)
        except ValueError as exc:
            result.add(str(exc))
            continue
        if number in seen_numbers:
            result.add(f"Duplicate phase id: Phase {number}.")
        seen_numbers.add(number)
        if number != index:
            result.add(f"Phase order must be contiguous; expected Phase {index}, found Phase {number}.")

        status_value = phase.get("status")
        if status_value not in ALLOWED_STATUSES:
            result.add(f"Phase {number} has invalid status: {status_value}.")
            continue
        next_count += int(status_value == "next")
        in_progress_count += int(status_value == "in_progress")
        if status_value == "completed":
            completed_numbers.append(number)
            tag = phase.get("tag")
            if not tag:
                result.add(f"Completed Phase {number} must declare a milestone tag.")
            elif not TAG_RE.fullmatch(str(tag)):
                result.add(f"Phase {number} tag has invalid format: {tag}.")

        checkpoints = phase.get("checkpoints") or []
        checkpoint_ids: set[str] = set()
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            checkpoint_id = checkpoint.get("id")
            if not checkpoint_id:
                result.add(f"Phase {number} checkpoint is missing id.")
            elif checkpoint_id in checkpoint_ids:
                result.add(f"Phase {number} has duplicate checkpoint id: {checkpoint_id}.")
            checkpoint_ids.add(str(checkpoint_id))
            checkpoint_status = checkpoint.get("status")
            if checkpoint_status not in ALLOWED_STATUSES:
                result.add(f"Checkpoint {checkpoint_id} has invalid status: {checkpoint_status}.")
            if status_value == "completed" and checkpoint_status != "completed":
                result.add(f"Completed Phase {number} has non-completed checkpoint: {checkpoint_id}.")

    if next_count > 1:
        result.add("Only one phase may be next.")
    if in_progress_count > 1:
        result.add("Only one phase may be in_progress.")
    for required in range(0, 5):
        if required not in seen_numbers:
            result.add(f"Missing required Phase {required}.")

    if completed_numbers:
        last_completed = max(completed_numbers)
        if project.get("last_completed_phase") != last_completed:
            result.add("project.last_completed_phase is inconsistent with completed phases.")
    if in_progress_count == 0 and project.get("current_phase") != project.get("next_phase"):
        result.add("project.current_phase must match project.next_phase while no phase is in progress.")
    by_number = phases_by_number(status)
    if by_number.get(3, {}).get("status") != "completed":
        result.add("Phase 3 must be completed.")
    if by_number.get(3, {}).get("tag") != "p3-leakage-safe-preprocessing-complete":
        result.add("Phase 3 tag must be p3-leakage-safe-preprocessing-complete.")
    phase4_status = by_number.get(4, {}).get("status")
    phase5_status = by_number.get(5, {}).get("status")
    if phase4_status not in {"next", "in_progress", "completed"}:
        result.add("Phase 4 must be next, in_progress, or completed.")
    if phase4_status == "completed":
        if project.get("last_completed_phase") != 4:
            result.add("project.last_completed_phase must be 4 when Phase 4 is completed.")
        if phase5_status not in {"next", "in_progress"}:
            result.add("Phase 5 must be next or in_progress after Phase 4 completion.")
        if project.get("current_phase") != 5 or project.get("next_phase") != 5:
            result.add("project.current_phase and project.next_phase must both be 5 after Phase 4 completion.")
    elif project.get("last_completed_phase") == 3 and project.get("next_phase") != 4:
        result.add("Phase 4 must be the next phase after Phase 3.")

    if check_tags:
        for phase in phases:
            if phase.get("status") == "completed" and phase.get("tag"):
                if not git_tag_exists(str(phase["tag"])):
                    result.add(f"Milestone tag not found locally: {phase['tag']}. Use --skip-tag-existence in shallow clones.")

    return result


def git_tag_exists(tag: str) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPO_ROOT.as_posix()}",
            "rev-parse",
            "-q",
            "--verify",
            f"refs/tags/{tag}",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def render_managed_section(status: dict[str, Any]) -> str:
    project = status["project"]
    by_number = phases_by_number(status)
    lines = [
        BEGIN,
        GENERATED_NOTE,
        "",
        f"- Last completed phase: Phase {project['last_completed_phase']}",
        f"- Current phase: Phase {project['current_phase']}",
        f"- Next phase: Phase {project['next_phase']} - {by_number[project['next_phase']].get('name', by_number[project['next_phase']]['title'])}",
        f"- Updated at: {project['updated_at']}",
        "",
        "| Phase | Status | Milestone tag |",
        "| --- | --- | --- |",
    ]
    for number in sorted(by_number):
        phase = by_number[number]
        tag = phase.get("tag", "")
        lines.append(f"| Phase {number} - {phase.get('name', phase['title'])} | {status_label(phase['status'])} | {tag or '-'} |")

    for number in sorted(by_number):
        structured_checkpoints = [
            checkpoint for checkpoint in by_number[number].get("checkpoints", []) if isinstance(checkpoint, dict)
        ]
        if not structured_checkpoints:
            continue
        lines.extend(["", f"Phase {number} checkpoints:"])
        for checkpoint in structured_checkpoints:
            lines.append(f"- {checkpoint['id']}: {status_label(checkpoint['status'])} - {checkpoint.get('summary', '')}")
    lines.extend(
        [
            "",
            "Current scope limits:",
            "- Core replication has not run.",
            "- Smoke, reduced, fake, preprocessing-validation, and metric-validation artifacts remain non-publishable validation artifacts.",
            "- Website still must not present validation artifacts as scientific results.",
            (
                "- Phase 4 completed metric validation for AUC, Brier Score, Partial Gini, and EMP unsupported handling; "
                "Phase 5 is next."
                if by_number[4]["status"] == "completed"
                else "- Phase 4 - Metric validation is next/in progress until metric validation completes."
            ),
            END,
        ]
    )
    return "\n".join(lines)


def replace_managed_section(text: str, rendered: str, path: Path) -> str:
    pattern = re.compile(rf"{re.escape(BEGIN)}.*?{re.escape(END)}", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{path} is missing managed project status markers.")
    return pattern.sub(rendered, text, count=1)


def expected_managed_outputs(status: dict[str, Any]) -> dict[Path, str]:
    rendered = render_managed_section(status)
    outputs: dict[Path, str] = {}
    for path in MANAGED_DOCS:
        outputs[path] = replace_managed_section(path.read_text(encoding="utf-8"), rendered, path)
    return outputs


def validate_current_status_text(files: list[Path]) -> ValidationResult:
    result = ValidationResult([])
    historical_marker = re.compile(r"(historical|lịch sử|plan cũ|ban đầu|example|acceptance criteria)", re.IGNORECASE)
    for path in files:
        if not path.exists():
            continue
        try:
            display_path = path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = path
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if historical_marker.search(line):
                continue
            for pattern in STALE_CURRENT_PATTERNS:
                if pattern.search(line):
                    result.add(f"Stale current status in {display_path}:{line_number}: {line.strip()}")
    return result


def validate_no_local_absolute_paths(status: dict[str, Any]) -> ValidationResult:
    result = ValidationResult([])
    text = yaml.safe_dump(status, allow_unicode=True, sort_keys=True)
    if re.search(r"[A-Za-z]:\\|/home/|/Users/", text):
        result.add("Project status source must not contain local absolute paths.")
    return result


def build_expected_status(status: dict[str, Any], phase_updates: dict[int, str] | None = None) -> dict[str, Any]:
    updated = copy.deepcopy(status)
    if phase_updates:
        for phase in updated["phases"]:
            number = phase_number(phase)
            if number in phase_updates:
                phase["status"] = phase_updates[number]
    return updated


def run(check: bool, skip_tag_existence: bool) -> int:
    status = read_status()
    errors: list[str] = []
    for validator_result in (
        validate_status_schema(status, check_tags=not skip_tag_existence),
        validate_current_status_text(WEBSITE_STATUS_FILES + DOC_STATUS_FILES),
        validate_no_local_absolute_paths(status),
    ):
        errors.extend(validator_result.errors)

    try:
        expected = expected_managed_outputs(status)
    except ValueError as exc:
        errors.append(str(exc))
        expected = {}

    stale_files: list[Path] = []
    for path, expected_text in expected.items():
        current = path.read_text(encoding="utf-8")
        if current != expected_text:
            stale_files.append(path)

    if check:
        if stale_files:
            errors.extend(f"Managed status section is stale: {path.relative_to(REPO_ROOT)}" for path in stale_files)
        if errors:
            print("Project status synchronization check failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("Project status synchronization check passed")
        return 0

    if errors:
        print("Project status validation failed; no files were updated:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    updated_files: list[str] = []
    for path, expected_text in expected.items():
        if path.read_text(encoding="utf-8") != expected_text:
            path.write_text(expected_text, encoding="utf-8", newline="\n")
            updated_files.append(str(path.relative_to(REPO_ROOT)))
    if updated_files:
        print("Updated project status files:")
        for file_name in updated_files:
            print(f"- {file_name}")
    else:
        print("Project status files already synchronized")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without writing files.")
    parser.add_argument(
        "--skip-tag-existence",
        action="store_true",
        help="Retained for compatibility; local tag existence checks are skipped by default.",
    )
    parser.add_argument(
        "--check-tag-existence",
        action="store_true",
        help="Also verify that completed-phase milestone tags exist locally.",
    )
    args = parser.parse_args()
    return run(check=args.check, skip_tag_existence=not args.check_tag_existence)


if __name__ == "__main__":
    raise SystemExit(main())
