"""Git provenance helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitProvenance:
    git_commit: str | None
    git_dirty: bool | None
    git_available: bool

    def to_dict(self) -> dict:
        return {
            "git_available": self.git_available,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
        }


def get_git_provenance(repo_root: Path) -> GitProvenance:
    safe_directory = repo_root.resolve().as_posix()
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={safe_directory}", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-c", f"safe.directory={safe_directory}", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return GitProvenance(git_commit=None, git_dirty=None, git_available=False)
    if commit.returncode != 0 or dirty.returncode != 0:
        return GitProvenance(git_commit=None, git_dirty=None, git_available=False)
    return GitProvenance(
        git_commit=commit.stdout.strip() or None,
        git_dirty=bool(dirty.stdout.strip()),
        git_available=True,
    )
