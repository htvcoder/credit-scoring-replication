# P7C.4B.2e-B2 — controlled target-canary operations

This runbook applies only to the locked four-task target canary. It does not
authorize canonical scientific execution. Commands that create authorization or
perform MLP fitting are explicit boundaries and must not be run without the
operator approval required at that gate.

Operational success means that the exact authorized four-task set completes with
valid identity, provenance, telemetry and completion artifacts. The canary has
only one measured repetition in each representative stratum, so scientific
coverage continues to report `incomplete_required_stratum` and
`insufficient_repetitions`; those findings do not by themselves reject the
operational canary. Scientific projection and `execution_plan_eligible` remain
false, and canary success neither creates scientific evidence nor authorizes
canonical execution.

## Preconditions and immutable variables

Run each gate in a fresh Bash shell on the official target server, not the old
validation environment.

```bash
set -euo pipefail
umask 077
REPO=/srv/credit-scoring-replication
PYTHON="$REPO/.venv/bin/python"
CONTROL=/secure/p7c4b2d
EXPECTED_HEAD="${EXPECTED_HEAD:?set to the exact final main commit after merge}"
MODE=cpu_parallel_2
OUT="$REPO/artifacts/p7c4b2d-target"
META="$CONTROL/operator-metadata.json"
ENVIRONMENT="$CONTROL/environment.json"
PROPOSAL="$CONTROL/proposal.json"
AUTHORIZATION="$CONTROL/effective-authorization.json"
UNIT=p7c4b2d-target-canary
LOG_PATH="$CONTROL/$UNIT.log"
LAUNCH_RECORD="$CONTROL/$UNIT.launch.json"
RECEIPT="$CONTROL/$UNIT.submission.json"
OPERATOR_IDENTITY="${OPERATOR_IDENTITY:?set accountable operator identity}"
AUTH_EXPIRES_AT="${AUTH_EXPIRES_AT:?set timezone-aware expiry within 24 hours}"
mkdir -p "$CONTROL"
chmod 700 "$CONTROL"
```

All variables are required. `EXPECTED_HEAD` is an operator-provided immutable
40-character lower-case Git SHA, obtained after this branch is merged to `main`;
it is deliberately not this branch's pre-merge baseline. `$OUT` must not exist;
control files are external to Git and must never contain credentials. The
specific unit name is single-use.

## Gate 1 — source identity (read-only)

```bash
cd "$REPO"
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
CURRENT_HEAD="$(git rev-parse HEAD)"
test "$(git branch --show-current)" = main
test "$CURRENT_HEAD" = "$EXPECTED_HEAD"
test "$(git rev-parse origin/main)" = "$EXPECTED_HEAD"
test -z "$(git status --short --untracked-files=all)"
```

Expected exit is 0. STOP on any failure; do not fetch, pull, reset, clean, or
switch branches.

## Gate 2 — target environment and systemd capability (read-only)

```bash
test -x "$PYTHON"
"$PYTHON" -c 'import psutil, numpy, pandas, sklearn; print("imports_ok")'
"$PYTHON" -m pip check
systemd --version
SYSTEMD_VERSION="$(systemd --version | awk 'NR==1 {print $2}')"
test "$SYSTEMD_VERSION" -ge 240
test "$(loginctl show-user "$USER" -p Linger --value)" = yes
df -h "$REPO/artifacts"
```

Expected exit is 0. `StandardOutput=append:path` and `StandardError=append:path`
require systemd 240 or later. STOP if a dependency, user
systemd, lingering, or append logging is unavailable. Do not silently substitute
tmux or nohup. Lingering keeps a user unit alive across logout/reboot; the unit
still does not survive a reboot as an active process.

## Gate 3 — operator metadata (operator-created input)

`$META` must be strict JSON with exactly these fields: `provider`, `region`,
`instance_id`, `disk_type`, `network_topology`, `vm_count`, `hourly_price`,
`currency`, `price_source`, `price_observed_at`, `maximum_runtime_hours`, and
`maximum_monetary_budget`. Values come from provider/server/pricing evidence;
they are not inferred by this runbook.

```bash
test -f "$META"
"$PYTHON" -m json.tool "$META" >/dev/null
```

STOP on malformed JSON or unknown values.

## Gate 4 — collect target evidence (mutation, no compute)

```bash
test ! -e "$ENVIRONMENT"
status=0
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli collect-target-environment \
  --mode "$MODE" --output-directory "$OUT" --operator-metadata "$META" \
  > "$ENVIRONMENT" || status=$?
test "$status" -eq 3
test -s "$ENVIRONMENT"
"$PYTHON" -m json.tool "$ENVIRONMENT" >/dev/null
```

Exit 3 is expected: collection emits valid JSON evidence but deliberately does
not grant authorization. The gate creates `$ENVIRONMENT`. STOP if it is not
exactly exit 3 or the JSON is absent/malformed.

## Gate 5 — inspect evidence and review plan (read-only)

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli inspect-target-requirements --environment "$ENVIRONMENT"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli review-plan --environment "$ENVIRONMENT"
```

Both commands must exit 0. STOP on any reason code; review must report
`READY_FOR_CANARY_AUTHORIZATION_REVIEW`.

## Gate 6 — render and validate proposal (proposal mutation, then read-only)

```bash
test ! -e "$PROPOSAL"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli render-authorization-proposal \
  --environment "$ENVIRONMENT" > "$PROPOSAL"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli validate-authorization-proposal \
  --environment "$ENVIRONMENT" --proposal "$PROPOSAL"
"$PYTHON" -m json.tool "$PROPOSAL"
```

Expected validation exit is 0. Human review must confirm the four task IDs,
mode, output, expiry/budget envelope, and `authorization_effective: false`.
STOP otherwise.

## Gate 7 — effective authorization

```text
AUTHORIZATION_MUTATION_BOUNDARY
```

```bash
test ! -e "$AUTHORIZATION"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli create-effective-authorization \
  --environment "$ENVIRONMENT" --proposal "$PROPOSAL" \
  --operator-identity "$OPERATOR_IDENTITY" \
  --operator-approval APPROVE_P7C4B2D_TARGET_CANARY \
  --expires-at "$AUTH_EXPIRES_AT" > "$AUTHORIZATION"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli validate-effective-authorization \
  --environment "$ENVIRONMENT" --proposal "$PROPOSAL" --authorization "$AUTHORIZATION"
```

This creates `$AUTHORIZATION`; validation must exit 0. STOP on failure. Do not
replace an authorization artifact.

## Gate 8 — prepare immutable launch evidence (mutation, no compute)

```bash
test ! -e "$OUT"
test ! -e "$LAUNCH_RECORD"
test ! -e "$RECEIPT"
LOAD_STATE="$(systemctl --user show "$UNIT" -p LoadState --value || true)"
test "$LOAD_STATE" = not-found

export PYTHON MODE OUT ENVIRONMENT PROPOSAL AUTHORIZATION
ARGV_JSON="$("$PYTHON" -c 'import json, os; print(json.dumps([os.environ["PYTHON"], "-m", "creditrep.experiments.p7c4b2c_cli", "run", "--execution-class", "target_preflight", "--mode", os.environ["MODE"], "--output", os.environ["OUT"], "--target-environment", os.environ["ENVIRONMENT"], "--authorization-proposal", os.environ["PROPOSAL"], "--effective-authorization", os.environ["AUTHORIZATION"]]))')"
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli create-launch-record \
  --record "$LAUNCH_RECORD" --git-commit "$(git -C "$REPO" rev-parse HEAD)" \
  --operator-identity "$OPERATOR_IDENTITY" --authorization "$AUTHORIZATION" \
  --environment "$ENVIRONMENT" --proposal "$PROPOSAL" --unit "$UNIT" \
  --argv-json "$ARGV_JSON" --working-directory "$REPO" \
  --python-executable "$PYTHON" --output-directory "$OUT" --log-path "$LOG_PATH"
```

This creates an atomic, immutable `$LAUNCH_RECORD` with state
`prepared_not_submitted`; it does not claim the workload was submitted. STOP if
the unit LoadState is anything other than `not-found`, any old control record
exists, or output exists. Never reset/reuse the unit name.

## Gate 9 — one-time systemd submission and receipt

```text
TARGET_CANARY_COMPUTE_BOUNDARY
```

```bash
status=0
systemd-run --user --unit="$UNIT" \
  --property="WorkingDirectory=$REPO" \
  --property="StandardOutput=append:$LOG_PATH" \
  --property="StandardError=append:$LOG_PATH" \
  "$PYTHON" -m creditrep.experiments.p7c4b2c_cli run \
    --execution-class target_preflight --mode "$MODE" --output "$OUT" \
    --target-environment "$ENVIRONMENT" --authorization-proposal "$PROPOSAL" \
    --effective-authorization "$AUTHORIZATION" || status=$?

SNAPSHOT_JSON="$(systemctl --user show "$UNIT" \
  -p LoadState -p ActiveState -p SubState -p MainPID -p ExecMainCode \
  -p ExecMainStatus -p Result -p InvocationID -p ExecMainStartTimestamp \
  -p ExecMainExitTimestamp | "$PYTHON" -c 'import json, sys; pairs=[line.rstrip("\n").split("=", 1) for line in sys.stdin if "=" in line]; print(json.dumps(dict(pairs), sort_keys=True))')"
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli create-submission-receipt \
  --receipt "$RECEIPT" --launch-record "$LAUNCH_RECORD" \
  --unit-snapshot-json "$SNAPSHOT_JSON" --systemd-run-exit-code "$status"
test "$status" -eq 0
```

The receipt is created after `systemd-run` returns regardless of its status and
contains the complete unit snapshot; `MainPID=0` alone is not failure. STOP on a
nonzero launcher exit, a missing/invalid receipt, `LoadState=not-found`, or an
unclassifiable snapshot. No `--collect` is used: the unit remains queryable for
final evidence. The log file is the stdout/stderr source; journal entries are
only lifecycle/launcher diagnostics.

## Gate 10 — monitor without resubmission (read-only)

```bash
systemctl --user show "$UNIT" -p LoadState -p ActiveState -p SubState -p MainPID \
  -p ExecMainCode -p ExecMainStatus -p Result -p InvocationID \
  -p ExecMainStartTimestamp -p ExecMainExitTimestamp
tail -n 100 "$LOG_PATH"
journalctl --user-unit "$UNIT" --no-pager -n 100
```

`active/running` means monitor only. A final `inactive/dead` plus
`ExecMainCode=exited`, `ExecMainStatus=0`, and `Result=success` is the success
candidate for artifact validation. Any failed/unknown state is STOP: do not run
the initial command again.

## Separate interrupted-run procedure

Do this only after the initial unit has a captured final snapshot. First prove it
is not `active`, `activating`, or `deactivating`, and that no related process
remains:

```bash
systemctl --user show "$UNIT" -p LoadState -p ActiveState -p SubState -p MainPID \
  -p ExecMainCode -p ExecMainStatus -p Result -p InvocationID \
  -p ExecMainStartTimestamp -p ExecMainExitTimestamp
if pgrep -af 'creditrep\.experiments\.p7c4b2c_cli|p7c4b2c_preflight'; then
  echo 'STOP: related process still exists' >&2
  exit 1
fi
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli resume-precheck --run-dir "$OUT"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli validate-effective-authorization \
  --environment "$ENVIRONMENT" --proposal "$PROPOSAL" --authorization "$AUTHORIZATION"
```

The resume precheck is read-only and exits 0 only when the run has the required
provenance files, is incomplete, and has no run-level completion marker. It does
not confirm authorization, source integrity, or that no child process remains;
the next command validates authorization and the runner repeats its own
fail-closed checks. STOP if the original unit/process state is uncertain,
authorization validation fails, or the precheck is nonzero. Do not delete,
alter, or replace `$OUT`.

Choose a new unique `RESUME_UNIT`, `RESUME_LOG_PATH`, `RESUME_RECORD`, and
`RESUME_RECEIPT`. Before creating the resume launch record, the new unit must
be unused:

```bash
test ! -e "$RESUME_RECORD"
test ! -e "$RESUME_RECEIPT"
RESUME_LOAD_STATE="$(systemctl --user show "$RESUME_UNIT" -p LoadState --value || true)"
test "$RESUME_LOAD_STATE" = not-found
```

Repeat Gates 8–10 with the `RESUME_*` values and replace only the argv after
`p7c4b2c_cli` with:

```bash
resume --run-dir "$OUT" --target-environment "$ENVIRONMENT" \
  --authorization-proposal "$PROPOSAL" --effective-authorization "$AUTHORIZATION"
```

Never call `run` for an interrupted output.

## Final artifact validation (read-only)

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2c_cli validate-artifacts --run-dir "$OUT"
```

Expected exit is 0 with `valid: true`, `target_canary_acceptance.accepted: true`
and `completed == expected`. STOP on every top-level acceptance reason code,
failure marker, missing completion marker, or incomplete task. Scientific
coverage reason codes remain visible under `scientific_coverage` and in
projection/eligibility while operational acceptance succeeds.

After this remediation is merged, any fresh target rerun must use a new proposal,
new target-environment evidence bound to the new source SHA, new effective
authorization, a new output namespace, and a new single-use systemd unit. Normal
validation always requires the run-level completion marker and never reclassifies
the marker-less historical failure as a successful run. That output remains
immutable diagnostic evidence: its four-task compute and task artifacts may be
interpreted read-only, but it must not be rewritten or presented as successful
evidence under the new source SHA.
