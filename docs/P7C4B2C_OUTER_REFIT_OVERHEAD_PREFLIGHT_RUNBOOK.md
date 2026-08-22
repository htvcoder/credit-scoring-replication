# P7C.4B.2c — Outer-refit and orchestration-overhead preflight

P7C.4B.2c closes the implementation gap left by P7C.4B.2b: outer-refit work and
clean orchestration overhead. It does not authorize or report a target preflight,
canonical experiment, selected compute mode, execution plan, cost, or scientific
result. Synthetic artifacts are pipeline-validation evidence only.

## Audit and evidence boundary

The P7C.4B.2a manifest derives 90 outer partitions from six datasets: AC, GC and
TH02 have 10 repeats × two folds; HMEQ, TC and GMC have five repeats × two folds.
Each of MLP-1/3/5 has one selected-candidate refit per partition, therefore the
population is `90 × 3 = 270`. The plan stores this structured derivation rather
than maintaining 270 as an unrelated total.

The canonical nested-CV path fits preprocessing on outer-train data, transforms
outer test, constructs and refits the selected/proxy MLP, predicts, computes the
metric, serializes result/model metadata, and writes artifacts. P7C.4B.2c adds an
optional timing sink to the existing `_fit_for_partition` primitive, preserving
its return contract. The target adapter uses this primitive. The synthetic
adapter uses a tiny deterministic dataset and a real imputer, scaler, estimator,
prediction, ROC AUC and serialization through the same orchestration pipeline;
its Logistic Regression model is an explicit typed limitation, not an MLP timing
proxy.

## Sampling plan

The bounded target plan covers dataset × MLP family × low/typical/high candidate
proxy × CPU mode. Each stratum has one separate warmup and two measured
repetitions. Dataset partition selection is deterministic from seed 4202. The
plan contains 324 tasks: 108 warmups and 216 measured samples. Candidate proxies
are workload representatives and never an observed canonical selection.

Stop conditions are per-sample failure, wall-clock budget, artifact budget and
incomplete required coverage. Missing strata or repetitions fail full-plan
scientific coverage and projection eligibility. For the separately authorized
four-task `target_canary`, operational acceptance instead requires the exact
authorized task set and valid artifacts/telemetry; its single measured repetition
per representative stratum remains explicitly scientifically insufficient. A
target run cannot truncate its authorized task set; synthetic validation may use
`--max-samples` to keep local work tiny.

Operational canary acceptance never implies scientific projection eligibility,
canonical evidence, or authorization for canonical scientific execution.
Normal validation also requires the run-level completion marker and cross-binds
the authorized source commit to the captured environment and every sample record;
stored validation reason codes cannot waive either requirement.

## Timing contract

All clocks use `time.perf_counter`. The additive outer-refit components are:

- `preprocessing_elapsed_seconds`;
- `model_fit_elapsed_seconds` (including model construction);
- `prediction_elapsed_seconds`;
- `metric_elapsed_seconds`;
- `artifact_write_elapsed_seconds`;
- `other_measured_orchestration_elapsed_seconds`.

Their sum is `aggregate_outer_refit_runtime_seconds`. Worker startup,
dispatch/queue, measured-phase wall clock, aggregate fit work and warmup elapsed
are non-additive observations. They must not be added again to outer-refit work,
which prevents double-counting contention/capacity loss. Warmup is always marked
`projection_eligible: false`. Unknown values remain `null`; a required unknown,
negative, non-finite, out-of-order, non-additive or interval-violating duration
fails validation.

## Artifact lifecycle

Every run has `plan.json`, `manifest.json`, `environment.json`, per-sample
`telemetry.json`/`result.json`/`COMPLETED.json`, `coverage.json`,
`stratum_summary.json`, `projection.json`, `eligibility.json`, `validation.json`
and a run-level `COMPLETED.json`. Failures retain attempt identity under
`failures/`. Work is first written under `tmp/`, validated, then atomically
promoted. Stale temporary and corrupt completed samples are moved under
`quarantine/`; valid completed samples are skipped on resume. Existing output
directories are never overwritten.

Sample provenance includes dataset/model/proxy/candidate, outer repeat/fold,
mode/worker count, seed, preprocessing identity, input identity/hash, plan hash,
Git HEAD, execution class, attempt identity, timestamps, timings, outcome and
limitations. Git HEAD is mandatory and cannot silently become null.

The validator reports stable reason codes for missing/malformed artifacts,
unsupported schema, plan/environment/input hash mismatch, missing/duplicate or
unexpected samples, incomplete strata/repetitions, bad timing/additivity,
warmup contamination, mode/worker mismatch, synthetic/target mismatch, failed
attempts, invalid completion markers, invalid projection sources and an invalid
true eligibility gate.

## Projection and eligibility

`project` consumes records only with a validator PASS whose evidence digest
matches those records. Outer-refit work is population-weighted by dataset,
model and proxy. Worker division is used only when an explicit overhead mapping
states that the canonical outer-refit scheduler parallelizes at that scope.
Fixed/per-worker/per-task/per-fold/per-artifact overhead requires a complete
event-count mapping; bounded P7C.4B.2b capacity loss is never multiplied by
1,500. Point, empirical lower/upper range, coverage, extrapolation ratio, source
digest and proxy warning are retained.

Eligibility remains false for missing/invalid inner evidence, coverage,
repetition, clean overhead mapping, total elapsed range, price input, valid
artifact or for high-severity warnings. Controlled tests can drive the gate true
to prove reachability; synthetic run artifacts and repository target state remain
ineligible. The existing P7C.4B.2b `propose-execution-plan` remains fail-closed.

## Commands

Planning and synthetic validation:

```powershell
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2c_cli create-plan --output <new-plan.json>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2c_cli validate-plan
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2c_cli run --execution-class synthetic_validation --mode cpu_parallel_1 --max-samples 2 --output <new-ignored-run-dir>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2c_cli resume --run-dir <run-dir>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2c_cli validate-artifacts --run-dir <run-dir>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2c_cli project --run-dir <run-dir>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2c_cli inspect-eligibility --run-dir <run-dir>
```

A target command requires separate `--target-environment`,
`--authorization-proposal` and `--effective-authorization` artifacts validated
by P7C.4B.2d. `target_canary` remains a four-task operational scope;
`target_projection_preflight` authorizes one complete 162-task mode. Canary
residual reuse is forbidden across source SHAs. The legacy
`--target-preflight-authorized` and
`--authorization-plan-digest` bypass is forbidden. Execution class selects its
workload from a closed mapping; public Python APIs reject callable injection.
Target output must resolve exactly to the authorized namespace, and live disk,
expiry and remaining runtime are checked before dispatch.

The outer projection environment is closed-world and ordered: `AC, GC, TH02,
HMEQ, TC, GMC`. Its versioned artifact binds exactly those six active-file
SHA-256 values, the full locked-runtime-input digest, plan digest, source Git
SHA, mode, output and resource policy. Missing, extra, duplicate, case-variant,
malformed or mismatched dataset evidence stops before proposal/authorization and
again before the runner creates its output directory. The historical AC/GMC
canary environment remains valid only for `target_canary`; it cannot authorize
fresh outer projection work.

Target manifests bind the original authorization digest and full scope.
`authorization_runtime.json` conservatively counts the wall-clock envelope from
the original start, including crash/downtime, so resume cannot replace the
authorization or reset elapsed runtime. The four-task canary retains its original
dispatch boundary. The projection-preflight stage instead uses killable spawned
children, a 20-minute task timeout, 12 GiB aggregate process-tree RSS, zero
tolerated failures, a maximum of two in-flight tasks, and independent per-run
12-hour/USD 5 ceilings. Any violation stops dispatch and terminates/reaps children.
This runbook does not create a real authorization or execute a target command.

On target run and resume, a self-consistent `sample_id`, plan digest, manifest or
task-set digest does not establish canonicality. The runner first applies exact
closed-world schemas at every plan, task and nested-candidate level, including
primitive types, finite numeric values, enums, ranges, counts and ordering. It
then reloads the locked P7C.4B.2a scientific manifest, which validates and
materializes its P7A source manifest and deterministic candidate generator, and
rebuilds the complete P7C.4B.2c plan with seed 4202. The submitted or persisted
plan must be canonical-JSON identical to this independently rebuilt plan.

The runner resolves the four ordered authorized IDs only against task objects
from the rebuilt plan. Submitted `plan.json` and `manifest.expected_tasks` are
persisted representations for exact comparison, never dispatch authorities. The
authorization provenance task-set digest is an additional consistency check, not
a replacement for locked-input rebuild or exact comparison. All mismatches fail
before cleanup, runtime-state mutation, executor construction or submission.

The locked manifests and their digests are unsigned local artifacts. This design
prevents an isolated plan edit followed by recomputation of local plan/sample
digests; it does not provide cryptographic authenticity against an attacker able
to rewrite every locked input, authorization and provenance artifact together.
Digests and checksums are not signatures.

## Typed target outer operations flow

The target outer stage is `target-outer-projection-preflight`; it maps only to
the authorization protocol stage `target_projection_preflight`. Never substitute
`target-inner-preflight`, invoke the runner directly, or hand-write operational
JSON. The following is the executable P1 flow. P2 uses a separately reviewed
`ENV_P2`/`PROPOSAL_P2`/`AUTH_P2`/`OUT_P2` chain, mode `cpu_parallel_2`, a `p2`
unit prefix, and may be submitted only after P1 closeout passes.

```bash
set -euo pipefail
REPO=/srv/credit-scoring-replication
PYTHON="$REPO/.venv/bin/python"
MODE_P1=cpu_parallel_1
OUT_P1="$REPO/artifacts/p7c4b2c-outer/$RUN_ID_P1"
UNIT_P1="p7c4b2c-outer-p1-$RUN_ID_P1.service"
LOG_P1="$REPO/artifacts/p7c4b2c-operations/$UNIT_P1.log"
LAUNCH_P1="$REPO/artifacts/p7c4b2c-operations/$UNIT_P1.launch.json"
CLAIM_P1="${LAUNCH_P1%/*}/.${LAUNCH_P1##*/}.submission-claim.json"
RECEIPT_P1="$REPO/artifacts/p7c4b2c-operations/$UNIT_P1.receipt.json"
SNAPSHOT_P1="$REPO/artifacts/p7c4b2c-operations/$UNIT_P1.unit.json"
SYSTEMD_STDOUT_P1="$REPO/artifacts/p7c4b2c-operations/$UNIT_P1.systemd-run.stdout"
SYSTEMD_STDERR_P1="$REPO/artifacts/p7c4b2c-operations/$UNIT_P1.systemd-run.stderr"
SUBMISSION_RESULT_P1="$REPO/artifacts/p7c4b2c-operations/$UNIT_P1.submission-result.json"
SYSTEMD_EXIT_P1="$REPO/artifacts/p7c4b2c-operations/$UNIT_P1.systemd-run.exit-code"
cd "$REPO"
test ! -e "$OUT_P1"
test ! -e "$LAUNCH_P1" && test ! -e "$CLAIM_P1" && test ! -e "$RECEIPT_P1"
test ! -e "$SYSTEMD_STDOUT_P1" && test ! -e "$SYSTEMD_STDERR_P1"
test ! -e "$SUBMISSION_RESULT_P1" && test ! -e "$SYSTEMD_EXIT_P1"
test ! -e "$SNAPSHOT_P1"
```

Validate the existing controls read-only. This does not create or refresh an
authorization.

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli \
  validate-authorization-proposal --environment "$ENV_P1" \
  --proposal "$PROPOSAL_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli \
  validate-effective-authorization --environment "$ENV_P1" \
  --proposal "$PROPOSAL_P1" --authorization "$AUTH_P1"
ARGV_P1=$(jq -cn --args '$ARGS.positional' -- \
  "$PYTHON" -m creditrep.experiments.p7c4b2c_cli run \
  --execution-class target_preflight --mode "$MODE_P1" --output "$OUT_P1" \
  --target-environment "$ENV_P1" --authorization-proposal "$PROPOSAL_P1" \
  --effective-authorization "$AUTH_P1")
```

Create the immutable launch record, then stop for the explicit compute boundary.

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli \
  create-launch-record --execution-stage target-outer-projection-preflight \
  --record "$LAUNCH_P1" --git-commit "$(git rev-parse HEAD)" \
  --operator-identity "$OPERATOR_ID" --authorization "$AUTH_P1" \
  --environment "$ENV_P1" --proposal "$PROPOSAL_P1" --unit "$UNIT_P1" \
  --argv-json "$ARGV_P1" --working-directory "$REPO" \
  --python-executable "$PYTHON" --output-directory "$OUT_P1" --log-path "$LOG_P1"
```

```text
TARGET_OUTER_PROJECTION_PREFLIGHT_COMPUTE_BOUNDARY
```

Claim exactly once immediately before submission. After this command succeeds,
or after any `Running as unit` response, **never submit again**, including after
an SSH disconnect. Recover the same unit and write its one-time receipt.

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli \
  create-submission-claim --launch-record "$LAUNCH_P1" --receipt "$RECEIPT_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli \
  submit-systemd-run --submission-result "$SUBMISSION_RESULT_P1" \
  --launch-record "$LAUNCH_P1" --systemd-run-stdout "$SYSTEMD_STDOUT_P1" \
  --systemd-run-stderr "$SYSTEMD_STDERR_P1" \
  --systemd-run-exit-code-file "$SYSTEMD_EXIT_P1"
SYSTEMD_RUN_RC=$(jq -r '.systemd_run_exit_code' "$SUBMISSION_RESULT_P1")
```

The typed submission result parses the saved output itself: a successful exit
must contain exactly the recorded unit and one lowercase 32-hex Invocation ID.
Never type or copy an Invocation ID into JSON. Parse `systemctl show` key/value
output; do not request JSON from systemctl. A collected unit is recoverable from
the immutable submission result even if the snapshot attempt is empty.

```bash
{ systemctl --user show "$UNIT_P1" \
  --property=LoadState,ActiveState,SubState,MainPID,ExecMainCode,ExecMainStatus,Result,InvocationID,ExecMainStartTimestamp,ExecMainExitTimestamp || true; } \
  | "$PYTHON" -c 'import json,sys; keys="LoadState ActiveState SubState MainPID ExecMainCode ExecMainStatus Result InvocationID ExecMainStartTimestamp ExecMainExitTimestamp".split(); got=dict(line.rstrip("\n").split("=",1) for line in sys.stdin if "=" in line); print(json.dumps({k:got.get(k,"") for k in keys},sort_keys=True))' \
  > "$SNAPSHOT_P1.tmp"
mv -n "$SNAPSHOT_P1.tmp" "$SNAPSHOT_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli \
  create-submission-receipt --receipt "$RECEIPT_P1" \
  --launch-record "$LAUNCH_P1" --unit "$UNIT_P1" \
  --submission-result "$SUBMISSION_RESULT_P1" \
  --unit-snapshot-file "$SNAPSHOT_P1" \
  --systemd-run-exit-code "$SYSTEMD_RUN_RC"
```

The receipt proves only the submission outcome and InvocationID binding; it does
not prove compute success. Monitor read-only, then close out P1 before preparing
or claiming P2.

```bash
systemctl --user show "$UNIT_P1" \
  --property=LoadState,ActiveState,SubState,MainPID,Result,InvocationID
journalctl --user -u "$UNIT_P1" --no-pager --lines=100
"$PYTHON" -m creditrep.experiments.p7c4b2c_cli \
  validate-artifacts --run-dir "$OUT_P1"
test -f "$OUT_P1/COMPLETED.json"
```

`submit-systemd-run` first atomically creates an immutable submission-attempt
marker that consumes the claim, then ignores SSH hangup while it runs the exact
launch-record command and atomically persists stdout, stderr, exit code and the
typed result before returning. A concurrent call or retry after any crash is
rejected even if capture did not finish. After SSH loss, do not repeat
submission; reuse the completed typed result. If the unit is already collected,
pass a key/value snapshot
whose `LoadState` is `not-found`, or pass the immutable empty failed snapshot
attempt; the receipt records snapshot unavailability without inferring compute
success. Never delete or overwrite the failed snapshot attempt. If the capture
transaction itself is incomplete, recovery is fail-closed: keep its evidence for
review and do not guess the exit code or resubmit.

For runner resume, first rebuild `SNAPSHOT_P1` with the proven key/value parser.
Resume only a B2c structurally safe incomplete run with inactive unit/process and
the original controls. Missing planned samples, final derived artifacts, and the
global completion marker are expected before finalization; they do not by
themselves make a run terminal. Resumability is determined from structural and
integrity invariants, never by matching validator reason-code text. A corrupt
persisted sample, provenance/runtime/control mismatch, completed run, or active
unit is NO-GO.

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli resume-precheck \
  --run-dir "$OUT_P1" --launch-record "$LAUNCH_P1" \
  --unit-snapshot-json "$(cat "$SNAPSHOT_P1")" \
  --environment "$ENV_P1" --proposal "$PROPOSAL_P1" --authorization "$AUTH_P1"
RESUME_UNIT_P1="p7c4b2c-outer-p1-$RUN_ID_P1-resume-$RESUME_ID.service"
RESUME_LAUNCH_P1="$REPO/artifacts/p7c4b2c-operations/$RESUME_UNIT_P1.launch.json"
RESUME_RECEIPT_P1="$REPO/artifacts/p7c4b2c-operations/$RESUME_UNIT_P1.receipt.json"
RESUME_LOG_P1="$REPO/artifacts/p7c4b2c-operations/$RESUME_UNIT_P1.log"
RESUME_SYSTEMD_STDOUT_P1="$REPO/artifacts/p7c4b2c-operations/$RESUME_UNIT_P1.systemd-run.stdout"
RESUME_SYSTEMD_STDERR_P1="$REPO/artifacts/p7c4b2c-operations/$RESUME_UNIT_P1.systemd-run.stderr"
RESUME_SYSTEMD_EXIT_P1="$REPO/artifacts/p7c4b2c-operations/$RESUME_UNIT_P1.systemd-run.exit-code"
RESUME_SUBMISSION_RESULT_P1="$REPO/artifacts/p7c4b2c-operations/$RESUME_UNIT_P1.submission-result.json"
RESUME_SNAPSHOT_P1="$REPO/artifacts/p7c4b2c-operations/$RESUME_UNIT_P1.unit.json"
RESUME_ARGV_P1=$(jq -cn --args '$ARGS.positional' -- \
  "$PYTHON" -m creditrep.experiments.p7c4b2c_cli resume --run-dir "$OUT_P1" \
  --target-environment "$ENV_P1" --authorization-proposal "$PROPOSAL_P1" \
  --effective-authorization "$AUTH_P1")
```

Create the fresh resume launch with `RESUME_ARGV_P1`, fresh unit/log paths, then
record and claim it as follows.

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli \
  create-launch-record --execution-stage target-outer-projection-preflight \
  --record "$RESUME_LAUNCH_P1" --git-commit "$(git rev-parse HEAD)" \
  --launch-record "$LAUNCH_P1" \
  --operator-identity "$OPERATOR_ID" --authorization "$AUTH_P1" \
  --environment "$ENV_P1" --proposal "$PROPOSAL_P1" --unit "$RESUME_UNIT_P1" \
  --argv-json "$RESUME_ARGV_P1" --working-directory "$REPO" \
  --python-executable "$PYTHON" --output-directory "$OUT_P1" \
  --log-path "$RESUME_LOG_P1"
```

```text
TARGET_OUTER_PROJECTION_PREFLIGHT_RESUME_COMPUTE_BOUNDARY
```

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli \
  create-submission-claim --launch-record "$RESUME_LAUNCH_P1" \
  --receipt "$RESUME_RECEIPT_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli \
  submit-systemd-run --submission-result "$RESUME_SUBMISSION_RESULT_P1" \
  --launch-record "$RESUME_LAUNCH_P1" \
  --systemd-run-stdout "$RESUME_SYSTEMD_STDOUT_P1" \
  --systemd-run-stderr "$RESUME_SYSTEMD_STDERR_P1" \
  --systemd-run-exit-code-file "$RESUME_SYSTEMD_EXIT_P1"
RESUME_SYSTEMD_RUN_RC=$(jq -r '.systemd_run_exit_code' \
  "$RESUME_SUBMISSION_RESULT_P1")
```

After the claim or any `Running as unit` response, never submit again. Snapshot
`RESUME_UNIT_P1` with the same key/value Python parser, then create the receipt
with `RESUME_LAUNCH_P1`, `RESUME_RECEIPT_P1`,
`RESUME_SUBMISSION_RESULT_P1`, the fresh unit snapshot and
`RESUME_SYSTEMD_RUN_RC`. Repeat read-only monitoring and public validation.
Never reuse the initial launch, unit, claim or receipt. Resume does not replace
authorization, reset runtime/budget, widen task scope, or turn completed/invalid
evidence into PASS.
For a resume launch, `submit-systemd-run` also queries the original unit with
key/value `systemctl show` immediately before consuming the fresh claim. It
fails closed unless that unit is absent, inactive or failed with no main PID;
the earlier operator precheck cannot be skipped to run two writers concurrently.

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2e_operations_cli \
  create-submission-receipt --receipt "$RESUME_RECEIPT_P1" \
  --launch-record "$RESUME_LAUNCH_P1" --unit "$RESUME_UNIT_P1" \
  --submission-result "$RESUME_SUBMISSION_RESULT_P1" \
  --unit-snapshot-file "$RESUME_SNAPSHOT_P1" \
  --systemd-run-exit-code "$RESUME_SYSTEMD_RUN_RC"
```

Target plans also declare the closed `runtime_input_binding` contract. Its
`locked_runtime_inputs_digest` is a canonical semantic projection of the complete
Protocol A typed config; the AC/GMC active file, target mapping, identifier,
ignored, categorical, numeric and missing-value fields; the reader options that
the loader actually consumes; and the unique selected SHA-256 registry rows plus
verified raw bytes. Descriptive registry metadata and unused datasets are
deliberately excluded, so documentation-only edits do not invalidate execution.
Git HEAD alone is insufficient because it cannot detect uncommitted edits.

Run and resume recompute this projection before output/cleanup/runtime mutation,
executor construction or submit. The digest is carried by target environment,
proposal, effective authorization and persisted provenance. Each worker reloads
and exact-checks it again, then passes the same typed Protocol A config and
dataset registry snapshot into `canonical_outer_refit`. Locked runtime YAML uses
safe duplicate-key rejection at every mapping level and strict primitive types;
numeric fields reject booleans and non-finite values. For each dataset load, the
loader reads the source once into an immutable byte snapshot, computes and checks
SHA-256 over those bytes before parsing, and gives pandas a `BytesIO` over the same
snapshot. It never reopens the source path for parsing or falls back to path-based
reading. This guarantees checked-byte/parsed-byte identity, not an atomic
filesystem read. The workload therefore does not independently reload unchecked
protocol or registry data.

Runtime state is cross-bound to the run ID, authorization/proposal/environment/
plan digests, normalized output, runtime limit and immutable original start.
Its checkpoint is mirrored in the manifest with a generation and state digest.
Missing, malformed, future-dated, clock-rollback, inconsistent or rollback state
fails closed before cleanup or dispatch; remaining runtime cannot increase across
valid resumes. These checks detect accidental corruption and localized tampering,
not an attacker able to rewrite every artifact and recompute all checksums.

Exit code 2 is artifact/config validation failure, 3 is valid but incomplete or
ineligible evidence, and 4 is authorization failure. Target evidence must still
supply validator-bound P7C.4B.2b inner projection, a closed-schema overhead event
mapping and current typed price input before an execution plan can be considered.
Combined projection rejects missing, unexpected or duplicate canonical task
identities and always leaves canonical scientific authorization false.
