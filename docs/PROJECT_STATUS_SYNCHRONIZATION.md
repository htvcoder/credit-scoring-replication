# Project Status Synchronization

## Source Of Truth

`website/content/progress.yaml` is the single source of truth for project phase, checkpoint, next-phase, and milestone tag status. The website reads this file directly during the Next.js build.

## Schema

The status file uses `schema_version: 1` and contains:

- `project.last_completed_phase`
- `project.current_phase`
- `project.next_phase`
- `project.updated_at`
- `status_enum`
- `phases[]`
- optional phase `checkpoints[]`

Each phase has a stable `numeric_id`, display `id`, `title`, optional English `name`, canonical `status`, optional milestone `tag`, narrative fields, and optional checkpoints.

## Status Enum

Allowed statuses are:

- `planned`
- `next`
- `in_progress`
- `completed`
- `blocked`
- `deferred`

Free-form status values are not allowed.

## Transition Rules

Only one phase may be `in_progress`. Only one phase may be `next`. A completed phase must have a milestone tag unless an explicit documented exception is added. A phase with required checkpoints cannot be `completed` unless those checkpoints are `completed`. `last_completed_phase`, `current_phase`, and `next_phase` must match the phase list.

Phase 3 is completed with tag `p3-leakage-safe-preprocessing-complete`. Phase 4 - Metric validation may be `next` before work starts, `in_progress` while P4 checkpoints are active, or `completed` once P4A/P4B/P4C all pass. After Phase 4 completion, Phase 5 must be `next`.

## Commands

Update managed sections:

```bash
python scripts/sync_project_status.py
```

Check without writing:

```bash
python scripts/sync_project_status.py --check
```

Local tag existence checks are skipped by default so synchronization can pass before a reviewed milestone tag is actually created. Use `--check-tag-existence` only when you intentionally want to require local tag existence as well as tag-name validation.

## Generated Files

No separate website derivative is generated because the website imports `website/content/progress.yaml` directly. The sync command updates managed sections in:

- `README.md`
- `docs/EXPERIMENT_IMPLEMENTATION_PLAN.md`

Managed sections are delimited by:

```markdown
<!-- PROJECT_STATUS:BEGIN -->
<!-- PROJECT_STATUS:END -->
```

Do not edit inside those markers manually.

## Human-Written Narrative

The rest of README, implementation docs, feasibility reports, P3A/P3B/P3C docs, website methods copy, and result-page copy remain human-written. They are validated for stale current-status claims, but historical notes and clearly marked old plans can remain.

## CI Guardrail

CI runs:

```bash
python scripts/sync_project_status.py --check
```

This check does not require raw datasets, production secrets, or network access.

## Checkpoint Completion Process

1. Confirm acceptance criteria pass.
2. Run relevant tests.
3. Update checkpoint status in `website/content/progress.yaml`.
4. Run `python scripts/sync_project_status.py`.
5. Run `python scripts/sync_project_status.py --check`.
6. Update human-written narrative only where current public status or method scope changed.

## Phase Completion Process

1. Confirm all required checkpoints are completed.
2. Confirm acceptance criteria and tests pass.
3. Record the milestone tag in `website/content/progress.yaml`.
4. Move the completed phase to `completed`.
5. Set the next phase to `next`.
6. Run the sync and check commands.
7. Run affected website and Python tests.

## Opening The Next Phase

Move at most one phase to `next` or `in_progress`. Keep `project.current_phase` and `project.next_phase` consistent with that state.

## Milestone Tags

Completed phases should record tags such as `p3-leakage-safe-preprocessing-complete`. The harness validates tag format by default and can optionally validate local tag existence when full Git history/tags are available.

## Historical Text

Historical notes, old plans, examples, and acceptance criteria may mention outdated statuses if their context is clear. Do not write stale current-status prose such as `Phase 3: Next` after Phase 3 completion.

## Smoke Metrics Rule

Smoke, reduced, fake, or preprocessing-validation metrics must never be presented as scientific results. They are non-publishable validation artifacts unless Phase 4+ validation explicitly changes their status.

## Example Transition

Phase 3 `in_progress` to `completed`:

- Set Phase 3 `status: completed`.
- Add `tag: p3-leakage-safe-preprocessing-complete`.
- Set P3A, P3B, and P3C checkpoints to `completed`.
- Set `project.last_completed_phase: 3`.

Phase 4 `planned` to `next` or `in_progress`:

- Set Phase 4 `status: next` before implementation starts, or `status: in_progress` once a Phase 4 checkpoint is being worked.
- Set `project.current_phase: 4`.
- Set `project.next_phase: 4`.

Phase 4 `in_progress` to `completed`:

- Set Phase 4 `status: completed`.
- Add `tag: p4-metric-validation-complete`.
- Set P4A, P4B, and P4C checkpoints to `completed`.
- Set `project.last_completed_phase: 4`.
- Set `project.current_phase: 5`.
- Set `project.next_phase: 5`.

## Phase Completion Checklist

[ ] Acceptance criteria pass  
[ ] Tests pass  
[ ] Source of truth updated  
[ ] Milestone tag recorded  
[ ] README/docs synchronized  
[ ] Website synchronized  
[ ] Next phase updated  
[ ] Status sync command run  
[ ] Status harness pass  
[ ] Website build/test pass
