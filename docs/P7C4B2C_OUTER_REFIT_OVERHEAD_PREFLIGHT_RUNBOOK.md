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
incomplete required coverage. Missing strata or repetitions fail validation and
eligibility. A target run cannot truncate the plan; synthetic validation may use
`--max-samples` to keep local work tiny.

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

A target-canary command requires separate `--target-environment`,
`--authorization-proposal` and `--effective-authorization` artifacts validated
by P7C.4B.2d. The legacy `--target-preflight-authorized` and
`--authorization-plan-digest` bypass is forbidden. Execution class selects its
workload from a closed mapping; public Python APIs reject callable injection.
Target output must resolve exactly to the authorized namespace, and live disk,
expiry and remaining runtime are checked before dispatch.

Target manifests bind the original authorization digest and full scope.
`authorization_runtime.json` conservatively counts the wall-clock envelope from
the original start, including crash/downtime, so resume cannot replace the
authorization or reset elapsed runtime. The runner does not claim to hard-kill a
task already running when authorization expires; it prevents subsequent dispatch.
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
supply complete inner projection, overhead event mapping and current price input
before an execution plan can be considered.
