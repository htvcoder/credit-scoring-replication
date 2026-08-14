# P7C.4B.2b — Single-VM CPU preflight runbook

The executable bounded-preflight harness is ready. Two target preflight
artifacts are available for projection-methodology audit; they remain
non-publishable engineering evidence and do not authorize canonical execution.

The scientific manifest remains locked at
`4d8636c3606e07e243efd2bc7be12806e7adf4fc1b19dbe0dc113a35adc57f75`.
The current schema-v3 preflight plan digest is
`9e6927da025fcc810ba6edbbce282a409593640d68fb6a493e0b18e7eec2fa8a`.
Schema-v2 artifacts with plan digest
`8e0f4d2c819ee4b2c89d0282fbaebf0601483e7f25db040c76307057eb3b1d5e`
remain validator/project compatible. They interleave warmup and measured tasks,
so their bounded orchestration overhead is unknown under the corrected method.
It selects TC/GMC × MLP-1/3/5 × light/median/heavy using workload-driver
complexity, never predictive metrics. Each mode has 18 warm-ups and 36 measured
fits; both modes total 108 fits without retry and 216 worst-case attempts.

Limits are: maximum two workers and two threads/worker; 1,800 seconds/fit;
12-hour global cap; one transient retry; 16 GiB free disk; 2 GiB artifact cap;
aggregate process-tree RSS at most 11.5 GiB (not per worker); and a target-profile
derived available-memory abort threshold initially set to 2 GiB. Target profile
validation rejects incompatible RAM/disk and requires provider/instance identity.

The target runner uses append-only generation-chained runtime state mirrored by
the run manifest and a per-task attempt-event chain. Resume validates complete
chains, reconstructs the exact attempt count, deterministically closes an
interrupted attempt, and never resets authorization expiry, runtime or resource
accounting. A transient failure remains in quarantine/attempt history; a later
successful retry owns the sole canonical task result and marker.

Each worker has a unique run/task/attempt temporary namespace. The parent alone
may promote it after rechecking authorization, accumulated runtime, timeout,
deduplicated process-tree RSS, available RAM, disk, projected artifact size,
control-file hashes, task/attempt identity, ledger heads and physical output
identity. Final runtime mutation and private validation precede the no-clobber
run marker; public marker-required validation is last. Process cleanup uses a
bounded suspend/discover/terminate/kill/reap algorithm, while uncertain RSS
sampling fails closed. These digests detect corruption and cross-binding under
the documented artifact-integrity threat model; they do not claim protection
against an attacker able to rewrite every artifact and canonical input.

Projection remains pending for development fixtures. Target evidence is
stratified; two-VM efficiency must be supplied below 100%, GPU remains pending,
and cost requires operator pricing.

## Projection timing methodology

The directly measured quantities are each task's process interval and wall-clock
duration, classification (`warmup` or `measured`), mode, and stratum. The
measured timing scope excludes every warmup. New runs execute all warmups before
the measured phase so that the measured-phase wall-clock interval and its idle
gaps are identifiable. Legacy runs that interleave warmups and measured tasks
remain readable, but their orchestration overhead is explicitly unknown because
the two scopes cannot be separated defensibly.

For each mode, stratified mean measured fit durations are multiplied by the
54,000 canonical inner-fit counts. This yields aggregate inner-fit runtime
`C_mode`, not CPU time and not total canonical elapsed time. Mode-specific fit
durations already include the contention observed with that mode. The only
work-conserving elapsed conversion is therefore `C_mode / N`, where `N` is one
or two workers; no additional parallel-efficiency divisor is applied.

For a separated bounded measured phase, the harness reports the observed
measured elapsed interval `E`, aggregate measured fit runtime `C_b`, worker
occupancy `C_b / (N * E)`, and the bounded capacity-loss equivalent
`E - C_b / N`. Adding that last term to `C_b / N` is an exact bounded-workload
back-check. It is descriptive only: scheduler/process-launch/I/O overhead is not
extrapolated 1,500 times from 36 measured fits. For interleaved legacy runs,
overhead remains unknown instead of being inferred from a warmup-contaminated
interval.

The 270 outer refits have no timing measurement, so total canonical elapsed is
unknown. Projection output separates the conditional inner-fit estimate from
the missing outer-refit and overhead components, retains
`high_extrapolation_ratio_non_guaranteed`, and is not execution-plan eligible.
Cost, GPU, and multi-VM elapsed remain pending until their required evidence is
available. `propose-execution-plan` must fail closed while any of these timing
components prevents a complete supported elapsed range.

## Target operator commands (do not run on development machines)

Fresh target execution requires a B2b-specific closed environment → proposal →
effective-authorization chain. The old `--bounded-preflight-authorized` flag is
not target authorization. B2d outer/canary authorization is also invalid here.
P1 and P2 require separate profile, environment, proposal, authorization,
output, launch and unit identities. The existing B2b limits above are bound
unchanged; outer limits such as USD 5, 20 minutes/task and 12 GiB do not apply.

Block 1 — source, process and namespace precheck (read-only):

```bash
set -euo pipefail
REPO=/srv/credit-scoring-replication
PYTHON="$REPO/.venv/bin/python"
EXPECTED_HEAD="${EXPECTED_HEAD:?reviewed merged SHA}"
cd "$REPO"
test "$(git branch --show-current)" = main
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse origin/main)" = "$EXPECTED_HEAD"
test -z "$(git status --short --untracked-files=all)"
ps -fu "$USER"
systemctl --user list-units 'p7c4b2b-inner-*' --all --no-pager
systemctl --user list-unit-files 'p7c4b2b-inner-*' --no-pager
test -x "$PYTHON"
test -d "$REPO/artifacts"
df -B1 "$REPO/artifacts"
free -b
ulimit -a
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli validate-plan
git status --short --untracked-files=all
echo 'READ_ONLY_PRECHECK_COMPLETE'
```

Block 2 — declare fresh P1 identities and enforce no-clobber:

```bash
set -euo pipefail
REPO=/srv/credit-scoring-replication
PYTHON="$REPO/.venv/bin/python"
CONTROL="${CONTROL:?fresh external control root}"
RUN_P1="${RUN_P1:?new P1 run ID}"
OUT_P1="$REPO/artifacts/p7c4b2b-compute-preflight/$RUN_P1"
PROFILE_P1="$CONTROL/$RUN_P1-profile.json"
ENV_P1="$CONTROL/$RUN_P1-environment.json"
PROPOSAL_P1="$CONTROL/$RUN_P1-proposal.json"
AUTH_P1="$CONTROL/$RUN_P1-authorization.json"
UNIT_P1="p7c4b2b-inner-p1-$RUN_P1"
LAUNCH_P1="$CONTROL/$RUN_P1-launch.json"
CLAIM_P1="$CONTROL/.$RUN_P1-launch.json.submission-claim.json"
RECEIPT_P1="$CONTROL/$RUN_P1-receipt.json"
SNAPSHOT_P1="$CONTROL/$RUN_P1-unit-snapshot.json"
LOG_P1="$CONTROL/$RUN_P1.log"
test ! -e "$OUT_P1"
test ! -e "$PROFILE_P1"
test ! -e "$ENV_P1"
test ! -e "$PROPOSAL_P1"
test ! -e "$AUTH_P1"
test ! -e "$LAUNCH_P1"
test ! -e "$CLAIM_P1"
test ! -e "$RECEIPT_P1"
test ! -e "$SNAPSHOT_P1"
test ! -e "$LOG_P1"
test "$(dirname "$OUT_P1")" = "$REPO/artifacts/p7c4b2b-compute-preflight"
test "$UNIT_P1" = "p7c4b2b-inner-p1-$RUN_P1"
echo 'P1_IDENTITIES_DECLARED_NO_CLOBBER'
```

Block 3 — collect and review P1 environment/profile evidence:

```bash
set -euo pipefail
set -o noclobber
: "${PYTHON:?reuse Block 2 session}"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli profile-machine \
  --machine-role intended_single_vm_target --provider "${PROVIDER:?}" \
  --instance-type "${INSTANCE_TYPE:?}" --profile-output "$PROFILE_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli collect-target-environment \
  --mode cpu_parallel_1 --profile "$PROFILE_P1" --output-dir "$OUT_P1" \
  > "$ENV_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli inspect-target-requirements \
  --profile "$PROFILE_P1" --target-environment "$ENV_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli review-target-plan \
  --profile "$PROFILE_P1" --target-environment "$ENV_P1"
test -s "$PROFILE_P1"
test -s "$ENV_P1"
sha256sum "$PROFILE_P1" "$ENV_P1"
git status --short --untracked-files=all
echo 'P1_ENVIRONMENT_REVIEW_COMPLETE'
```

Block 4 — finalize P1 proposal, then stop for human review:

```bash
set -euo pipefail
set -o noclobber
: "${PYTHON:?reuse Blocks 2-3 session}"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli render-authorization-proposal \
  --profile "$PROFILE_P1" --target-environment "$ENV_P1" \
  --run-id "$RUN_P1" > "$PROPOSAL_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli validate-authorization-proposal \
  --profile "$PROFILE_P1" --target-environment "$ENV_P1" \
  --authorization-proposal "$PROPOSAL_P1"
test -s "$PROPOSAL_P1"
sha256sum "$PROFILE_P1" "$ENV_P1" "$PROPOSAL_P1"
test ! -e "$AUTH_P1"
test ! -e "$OUT_P1"
systemctl --user is-active --quiet "$UNIT_P1" && exit 4 || true
git status --short --untracked-files=all
echo 'STOP: B2B_P1_AUTHORIZATION_MUTATION_BOUNDARY'
```

Repeat Blocks 2–4 for P2 with new names, `cpu_parallel_2`, and a distinct
profile/environment/proposal. Never copy or edit P1 JSON to create P2.

Block 5 — explicit authorization mutation after review (one mode at a time):

```bash
set -euo pipefail
set -o noclobber
: "${PYTHON:?reuse reviewed session}"
: "${OPERATOR_IDENTITY:?reviewed operator identity}"
: "${EXPIRES_AT:?reviewed UTC expiry}"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli create-effective-authorization \
  --profile "$PROFILE_P1" --target-environment "$ENV_P1" \
  --authorization-proposal "$PROPOSAL_P1" \
  --operator-identity "$OPERATOR_IDENTITY" \
  --operator-approval APPROVE_P7C4B2B_TARGET_INNER_PREFLIGHT \
  --expires-at "$EXPIRES_AT" > "$AUTH_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli validate-effective-authorization \
  --profile "$PROFILE_P1" --target-environment "$ENV_P1" \
  --authorization-proposal "$PROPOSAL_P1" \
  --effective-authorization "$AUTH_P1"
test -s "$AUTH_P1"
sha256sum "$PROFILE_P1" "$ENV_P1" "$PROPOSAL_P1" "$AUTH_P1"
test ! -e "$OUT_P1"
test ! -e "$LAUNCH_P1"
test ! -e "$CLAIM_P1"
test ! -e "$RECEIPT_P1"
systemctl --user is-active --quiet "$UNIT_P1" && exit 4 || true
ps -fu "$USER"
echo 'STOP: B2B_P1_COMPUTE_SUBMISSION_BOUNDARY'
```

Block 6 — create the immutable launch record after the second confirmation:

```bash
set -euo pipefail
set -o noclobber
: "${PYTHON:?reuse reviewed session}"
: "${EXPECTED_HEAD:?reuse reviewed source SHA}"
test ! -e "$OUT_P1"
test ! -e "$LAUNCH_P1"
test ! -e "$CLAIM_P1"
test ! -e "$RECEIPT_P1"
systemctl --user is-active --quiet "$UNIT_P1" && exit 4 || true
ARGV_P1=$(jq -cn --args "$PYTHON" -m creditrep.experiments.p7c4b2b_cli run \
  --mode cpu_parallel_1 --profile "$PROFILE_P1" --target-machine-asserted \
  --target-environment "$ENV_P1" --authorization-proposal "$PROPOSAL_P1" \
  --effective-authorization "$AUTH_P1" --output-dir "$OUT_P1" '$ARGS.positional')
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli create-launch-record \
  --execution-stage target-inner-preflight --record "$LAUNCH_P1" \
  --git-commit "$EXPECTED_HEAD" --operator-identity "$OPERATOR_IDENTITY" \
  --machine-profile "$PROFILE_P1" --environment "$ENV_P1" \
  --proposal "$PROPOSAL_P1" --authorization "$AUTH_P1" --unit "$UNIT_P1" \
  --argv-json "$ARGV_P1" --working-directory "$REPO" \
  --python-executable "$PYTHON" --output-directory "$OUT_P1" --log-path "$LOG_P1"
test -s "$LAUNCH_P1"
test ! -e "$CLAIM_P1"
echo 'STOP: B2B_P1_EXPLICIT_COMPUTE_BOUNDARY'
```

Block 7 — submit exactly the recorded command once and persist its receipt:

```bash
set -euo pipefail
test -s "$LAUNCH_P1"
test ! -e "$CLAIM_P1"
test ! -e "$RECEIPT_P1"
test ! -e "$OUT_P1"
systemctl --user is-active --quiet "$UNIT_P1" && exit 4 || true
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli create-submission-claim \
  --launch-record "$LAUNCH_P1" --receipt "$RECEIPT_P1"
test -s "$CLAIM_P1"
set +e
systemd-run --user --unit="$UNIT_P1" --collect --working-directory="$REPO" \
  --property="StandardOutput=append:$LOG_P1" --property="StandardError=append:$LOG_P1" \
  "$PYTHON" -m creditrep.experiments.p7c4b2b_cli run \
  --mode cpu_parallel_1 --profile "$PROFILE_P1" --target-machine-asserted \
  --target-environment "$ENV_P1" --authorization-proposal "$PROPOSAL_P1" \
  --effective-authorization "$AUTH_P1" --output-dir "$OUT_P1"
SUBMIT_RC=$?
set -e
SNAPSHOT_JSON=$(systemctl --user show "$UNIT_P1" --output=json \
  -p LoadState -p ActiveState -p SubState -p MainPID -p ExecMainCode \
  -p ExecMainStatus -p Result -p InvocationID -p ExecMainStartTimestamp \
  -p ExecMainExitTimestamp | jq '.[0] | with_entries(.value |= tostring)' || \
  jq -cn '{LoadState:"",ActiveState:"",SubState:"",MainPID:"",ExecMainCode:"",ExecMainStatus:"",Result:"",InvocationID:"",ExecMainStartTimestamp:"",ExecMainExitTimestamp:""}')
printf '%s\n' "$SNAPSHOT_JSON" > "$SNAPSHOT_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli create-submission-receipt \
  --receipt "$RECEIPT_P1" --launch-record "$LAUNCH_P1" \
  --unit "$UNIT_P1" \
  --unit-snapshot-json "$(cat "$SNAPSHOT_P1")" --systemd-run-exit-code "$SUBMIT_RC"
test -s "$RECEIPT_P1"
test "$SUBMIT_RC" -eq 0
```

Block 8 — read-only monitoring; never submit `run` again:

```bash
set -euo pipefail
test -s "$LAUNCH_P1"
test -s "$CLAIM_P1"
test -s "$RECEIPT_P1"
ps -fu "$USER"
systemctl --user status "$UNIT_P1" --no-pager || true
journalctl --user -u "$UNIT_P1" --no-pager -n 100 || true
find "$OUT_P1" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM %p\n' || true
du -sb "$OUT_P1" || true
df -B1 "$REPO/artifacts"
free -b
test ! -L "$OUT_P1"
test "$(readlink -f "$(dirname "$OUT_P1")")" = \
  "$(readlink -f "$REPO/artifacts/p7c4b2b-compute-preflight")"
echo 'If interrupted, first confirm the unit and every descendant are gone.'
echo 'Then use p7c4b2b_cli resume with the same four canonical control paths.'
echo 'Create a fresh resume unit, launch record, and receipt; never repeat run.'
echo 'Monitoring is read-only; do not edit ledger, manifest, marker, or controls.'
```

Block 9 — closeout and validation after the unit is inactive:

```bash
set -euo pipefail
systemctl --user is-active --quiet "$UNIT_P1" && exit 4 || true
test -d "$OUT_P1"
test ! -L "$OUT_P1"
test -f "$OUT_P1/COMPLETED.json" || exit 3
test -d "$OUT_P1/runtime-ledger"
test -d "$OUT_P1/attempt-ledger"
test -s "$LAUNCH_P1"
test -s "$CLAIM_P1"
test -s "$RECEIPT_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli validate-artifacts \
  --output-dir "$OUT_P1"
find "$OUT_P1" -name '*.tmp' -o -name '*.partial'
test -z "$(find "$OUT_P1" -name '*.tmp' -o -name '*.partial')"
sha256sum "$PROFILE_P1" "$ENV_P1" "$PROPOSAL_P1" "$AUTH_P1"
sha256sum "$LAUNCH_P1" "$CLAIM_P1" "$RECEIPT_P1"
git status --short --untracked-files=all
echo 'B2B_P1_CLOSEOUT_VALIDATED_NON_SCIENTIFIC_PREFLIGHT_ONLY'
```

Blocks 10–13 apply only to an interrupted, incomplete run. Never use them for a
run with a valid run-level `COMPLETED.json`.

Block 10 — prepare and atomically claim one explicit resume submission:

```bash
set -euo pipefail
: "${RESUME_ID:?fresh reviewed resume identity}"
systemctl --user is-active --quiet "$UNIT_P1" && exit 4 || true
test -d "$OUT_P1"
test ! -e "$OUT_P1/COMPLETED.json"
RESUME_UNIT_P1="p7c4b2b-inner-p1-$RUN_P1-resume-$RESUME_ID"
RESUME_LAUNCH_P1="$CONTROL/$RUN_P1-resume-$RESUME_ID-launch.json"
RESUME_CLAIM_P1="$CONTROL/.$RUN_P1-resume-$RESUME_ID-launch.json.submission-claim.json"
RESUME_RECEIPT_P1="$CONTROL/$RUN_P1-resume-$RESUME_ID-receipt.json"
RESUME_SNAPSHOT_P1="$CONTROL/$RUN_P1-resume-$RESUME_ID-unit-snapshot.json"
RESUME_LOG_P1="$CONTROL/$RUN_P1-resume-$RESUME_ID.log"
test ! -e "$RESUME_LAUNCH_P1"
test ! -e "$RESUME_CLAIM_P1"
test ! -e "$RESUME_RECEIPT_P1"
test ! -e "$RESUME_SNAPSHOT_P1"
systemctl --user is-active --quiet "$RESUME_UNIT_P1" && exit 4 || true
RESUME_ARGV_P1=$(jq -cn --args "$PYTHON" -m creditrep.experiments.p7c4b2b_cli resume \
  --mode cpu_parallel_1 --profile "$PROFILE_P1" --target-machine-asserted \
  --target-environment "$ENV_P1" --authorization-proposal "$PROPOSAL_P1" \
  --effective-authorization "$AUTH_P1" --output-dir "$OUT_P1" '$ARGS.positional')
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli create-launch-record \
  --execution-stage target-inner-preflight --record "$RESUME_LAUNCH_P1" \
  --git-commit "$EXPECTED_HEAD" --operator-identity "$OPERATOR_IDENTITY" \
  --machine-profile "$PROFILE_P1" --environment "$ENV_P1" \
  --proposal "$PROPOSAL_P1" --authorization "$AUTH_P1" --unit "$RESUME_UNIT_P1" \
  --argv-json "$RESUME_ARGV_P1" --working-directory "$REPO" \
  --python-executable "$PYTHON" --output-directory "$OUT_P1" --log-path "$RESUME_LOG_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli create-submission-claim \
  --launch-record "$RESUME_LAUNCH_P1" --receipt "$RESUME_RECEIPT_P1"
```

Block 11 — submit exactly the claimed resume argv and persist its receipt:

```bash
set -euo pipefail
test -s "$RESUME_LAUNCH_P1"
test -s "$RESUME_CLAIM_P1"
test ! -e "$RESUME_RECEIPT_P1"
systemctl --user is-active --quiet "$RESUME_UNIT_P1" && exit 4 || true
set +e
systemd-run --user --unit="$RESUME_UNIT_P1" --collect --working-directory="$REPO" \
  --property="StandardOutput=append:$RESUME_LOG_P1" \
  --property="StandardError=append:$RESUME_LOG_P1" \
  "$PYTHON" -m creditrep.experiments.p7c4b2b_cli resume \
  --mode cpu_parallel_1 --profile "$PROFILE_P1" --target-machine-asserted \
  --target-environment "$ENV_P1" --authorization-proposal "$PROPOSAL_P1" \
  --effective-authorization "$AUTH_P1" --output-dir "$OUT_P1"
RESUME_SUBMIT_RC=$?
set -e
RESUME_SNAPSHOT_JSON=$(systemctl --user show "$RESUME_UNIT_P1" --output=json \
  -p LoadState -p ActiveState -p SubState -p MainPID -p ExecMainCode \
  -p ExecMainStatus -p Result -p InvocationID -p ExecMainStartTimestamp \
  -p ExecMainExitTimestamp | jq '.[0] | with_entries(.value |= tostring)' || \
  jq -cn '{LoadState:"",ActiveState:"",SubState:"",MainPID:"",ExecMainCode:"",ExecMainStatus:"",Result:"",InvocationID:"",ExecMainStartTimestamp:"",ExecMainExitTimestamp:""}')
printf '%s\n' "$RESUME_SNAPSHOT_JSON" > "$RESUME_SNAPSHOT_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli create-submission-receipt \
  --receipt "$RESUME_RECEIPT_P1" --launch-record "$RESUME_LAUNCH_P1" \
  --unit "$RESUME_UNIT_P1" --unit-snapshot-json "$(cat "$RESUME_SNAPSHOT_P1")" \
  --systemd-run-exit-code "$RESUME_SUBMIT_RC"
test "$RESUME_SUBMIT_RC" -eq 0
```

Block 12 — read-only resume monitoring:

```bash
set -euo pipefail
test -s "$RESUME_LAUNCH_P1"
test -s "$RESUME_CLAIM_P1"
test -s "$RESUME_RECEIPT_P1"
test -s "$RESUME_SNAPSHOT_P1"
jq -e --arg unit "$RESUME_UNIT_P1" \
  '.systemd_unit == $unit and .runner_command == "resume"' "$RESUME_LAUNCH_P1"
jq -e --arg unit "$RESUME_UNIT_P1" \
  '.systemd_unit == $unit and .submission_state == "claimed_not_submitted"' \
  "$RESUME_CLAIM_P1"
jq -e --arg unit "$RESUME_UNIT_P1" '.systemd_unit == $unit' "$RESUME_RECEIPT_P1"
ps -fu "$USER"
systemctl --user status "$RESUME_UNIT_P1" --no-pager || true
journalctl --user -u "$RESUME_UNIT_P1" --no-pager -n 100 || true
find "$OUT_P1" -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM %p\n'
du -sb "$OUT_P1"
df -B1 "$REPO/artifacts"
free -b
test ! -L "$OUT_P1"
echo 'Monitoring only: do not edit control, ledger, manifest, marker, or claim.'
```

Block 13 — resume closeout uses the same public validator:

```bash
set -euo pipefail
systemctl --user is-active --quiet "$RESUME_UNIT_P1" && exit 4 || true
test -f "$OUT_P1/COMPLETED.json" || exit 3
test -s "$RESUME_SNAPSHOT_P1"
jq -e --arg unit "$RESUME_UNIT_P1" \
  '.systemd_unit == $unit and .runner_command == "resume"' "$RESUME_LAUNCH_P1"
jq -e --arg unit "$RESUME_UNIT_P1" \
  '.systemd_unit == $unit and .submission_state == "claimed_not_submitted"' \
  "$RESUME_CLAIM_P1"
jq -e --arg unit "$RESUME_UNIT_P1" \
  '.systemd_unit == $unit and
   (.submission_state == "submitted" or .submission_state == "submission_failed")' \
  "$RESUME_RECEIPT_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli validate-artifacts \
  --output-dir "$OUT_P1"
test -z "$(find "$OUT_P1" -name '*.tmp' -o -name '*.partial')"
sha256sum "$PROFILE_P1" "$ENV_P1" "$PROPOSAL_P1" "$AUTH_P1"
sha256sum "$RESUME_LAUNCH_P1" "$RESUME_CLAIM_P1" "$RESUME_RECEIPT_P1"
git status --short --untracked-files=all
echo 'B2B_P1_RESUME_CLOSEOUT_VALIDATED_NON_SCIENTIFIC_PREFLIGHT_ONLY'
```

If a process crashes after a claim is created but before `systemd-run`, stop for
explicit operator review. The immutable claim deliberately blocks blind
resubmission; do not delete or replace it. Execute P2 only through its distinct
reviewed chain and identities.

Canonical execution remains blocked while outer-refit runtime and defensible
canonical orchestration overhead are unknown and extrapolation remains 1,500x.
`propose-execution-plan` fails closed for this incomplete projection; do not run
it until a future evidence contract makes total elapsed execution-plan eligible.
After that, a compute-mode decision, exact execution-plan digest and human
execution/cost approval are still required.
The proposed plan remains deliberately unapproved; an approval record must bind
its exact digest and contain a human approver and timestamp before a later
canonical guard can authorize execution.
