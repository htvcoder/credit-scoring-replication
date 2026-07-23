# Repository Agent Instructions

## Project Status Synchronization Rule

Every task that changes a phase, checkpoint, milestone, or next-phase state must update the project status source of truth at `website/content/progress.yaml`.

Codex must audit related documentation and website surfaces before editing status. At minimum, check README/status docs, the implementation plan, website progress and methods content, milestone tag metadata, and CI/status scripts.

After changing status, Codex must run:

```bash
python scripts/sync_project_status.py
python scripts/sync_project_status.py --check
```

Status-changing tasks must update the implementation plan, relevant README/status docs, website progress content, milestone tag metadata, and next phase. Do not mark a phase `completed` merely because code has been written. Mark a phase `completed` only after acceptance criteria pass and required tests pass.

Smoke, reduced, fake, or preprocessing-validation metrics must never be published or described as scientific results. They may only be described as non-publishable validation artifacts when that is true.

Do not end a status-changing task while the status synchronization harness is failing. The final Codex report must include a section named `Project status synchronization` listing the source of truth updated, docs updated, website updated, status check command, and result.

If a task does not change project status, Codex must report:

```text
Project status unchanged; synchronization not required.
```

Do not commit, push, merge, create tags, or edit existing Git tags unless the user explicitly asks for that action.
