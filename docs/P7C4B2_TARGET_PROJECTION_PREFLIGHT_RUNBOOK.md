# P7C.4B.2 — Target projection-preflight runbook

## Purpose and boundary

This is the next technical step inside the existing P7C.4B.2 checkpoint. It
collects target inner-fit and outer-refit timing evidence for a supported
scientific runtime/cost projection. It is engineering/preflight execution, not
canonical scientific execution, and it cannot authorize the latter.

The accepted canary provides operational identity, launch, telemetry and four
completed task records. It does not provide complete projection coverage. The
remaining evidence is target P7C.4B.2b inner-fit timing for both CPU modes,
complete P7C.4B.2c outer-refit strata/repetitions, a reviewed overhead mapping,
and current operator price input.

## Deterministic minimum scope

The locked P7C.4B.2c plan has 108 strata: six datasets × three MLP families ×
three candidate-complexity proxies × two CPU modes. Every stratum requires one
warmup and two measured repetitions, for 324 tasks total.

- Each full mode is 54 strata, 54 warmups and 108 measured tasks: 162 tasks.
- The accepted canary used an older source SHA. It cannot be combined with a
  post-merge projection preflight without violating source identity. Production
  scope therefore uses two self-contained 162-task runs and offers no residual
  or compatibility bypass.
- P7C.4B.2b contributes 18 warmups and 36 measured fits per mode, 108 tasks
  across both modes. These are inner-fit engineering measurements.
- Each outer run independently binds 12 hours, USD 5, a 20-minute hard task
  timeout, 12 GiB aggregate process-tree RSS, zero tolerated task failures and
  at most its authorized worker count in flight (hard cap two). P1 and P2 have
  a combined theoretical ceiling of 24 hours/USD 10, but there is no shared
  campaign ledger and no transfer of unused allowance.
- The two B2b inner runs retain their separate B2b contract. These outer limits
  do not apply to inner measurement or future canonical scientific execution.

## Gate 1 — source and static validation

```bash
set -euo pipefail
REPO=/srv/credit-scoring-replication
PYTHON="$REPO/.venv/bin/python"
EXPECTED_HEAD="${EXPECTED_HEAD:?set reviewed merged main SHA}"
cd "$REPO"
test "$(git branch --show-current)" = main
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse origin/main)" = "$EXPECTED_HEAD"
test -z "$(git status --short --untracked-files=all)"
"$PYTHON" -m creditrep.experiments.p7c4b2b_cli validate-plan
"$PYTHON" -m creditrep.experiments.p7c4b2c_cli validate-plan
```

STOP on any nonzero exit. Do not repair the server from this runbook.

## Gate 2 — target inner-fit preflight

```text
TARGET_INNER_FIT_PREFLIGHT_BOUNDARY
```

Follow `P7C4B2B_SINGLE_VM_PREFLIGHT_RUNBOOK.md` to create two independent typed
chains. Each chain has a distinct validated target profile/environment,
non-effective proposal, effective authorization, output, run, launch and unit.
Only `APPROVE_P7C4B2B_TARGET_INNER_PREFLIGHT` is accepted. The legacy boolean
flag alone cannot authorize target execution, and no B2d artifact may substitute.

Do not invoke either runner in the foreground from this projection runbook.
For each mode, use Blocks 1–9 of the B2b runbook: create the typed
`target-inner-preflight` launch record, cross the explicit compute boundary,
submit that exact recorded argv with its prechecked `systemd-run --user` unit,
and persist the one-time receipt. P1 must close out before the separately
reviewed P2 chain is submitted. A stopped run may use only an explicit `resume`
argv in a fresh unit/launch/receipt; never resubmit `run`.

The two outputs require explicit no-clobber and single-submit checks. Resume
uses the same persisted typed chain and never replaces authorization or a
completed artifact. B2b retains its locked 1,800-second fit timeout, 12-hour
global wall-clock accounting, disk/RSS/RAM/artifact/retry limits and worker/thread
caps. It does not inherit the outer 12-hour/USD 5/20-minute/12-GiB policy.
New B2b manifests bind their normalized output directory; combined projection
requires that binding, so a relocated export or summary cannot substitute for
the original validator-pass run root. Legacy B2b validation remains compatible,
but legacy manifests without this projection binding are not combination input.

## Gate 3 — outer-refit proposals

Create separate target environments because mode and output namespace are
digest-bound. Use `target_projection_preflight` for the self-contained full
scope.

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli \
  render-authorization-proposal --environment "$ENV_P1" \
  --stage target_projection_preflight > "$PROPOSAL_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli \
  render-authorization-proposal --environment "$ENV_P2" \
  --stage target_projection_preflight > "$PROPOSAL_P2"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli \
  validate-authorization-proposal --environment "$ENV_P1" \
  --proposal "$PROPOSAL_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2d_cli \
  validate-authorization-proposal --environment "$ENV_P2" \
  --proposal "$PROPOSAL_P2"
```

Human review must confirm exactly 162 tasks per proposal.

## Gate 4 — authorization boundary

```text
EFFECTIVE_PROJECTION_PREFLIGHT_AUTHORIZATION_BOUNDARY
```

Any later effective authorization must use the exact approval phrase
`APPROVE_P7C4B2_TARGET_PROJECTION_PREFLIGHT`, bind one proposal/environment/output,
and remain within its runtime, expiry, disk and monetary limits. Canary approval
is rejected for this stage. Creating authorization is an operator action after
review; this runbook does not perform it automatically.

## Gate 5 — target outer-refit execution and resume

```text
TARGET_OUTER_REFIT_PREFLIGHT_BOUNDARY
```

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2c_cli run \
  --execution-class target_preflight --mode cpu_parallel_1 --output "$OUT_P1" \
  --target-environment "$ENV_P1" --authorization-proposal "$PROPOSAL_P1" \
  --effective-authorization "$AUTH_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2c_cli run \
  --execution-class target_preflight --mode cpu_parallel_2 --output "$OUT_P2" \
  --target-environment "$ENV_P2" --authorization-proposal "$PROPOSAL_P2" \
  --effective-authorization "$AUTH_P2"
```

Use the reviewed long-running operations wrapper where required. Never repeat
`run` for an existing output. Use `resume --run-dir` with the original bound
environment, proposal and authorization only after process-state checks.
P1 and P2 require distinct environment, proposal, effective authorization,
output, run, launch and unit identities. Treat every output as single-use and
stop on collision, policy mismatch, timeout, expiry, memory, failure, runtime
or monetary violation.

## Gate 6 — validation and projection

`overhead-mapping.json` is a reviewed closed-world
`p7c4b2_outer_projection_overhead` artifact. It binds common source SHA, locked
plan digest, selected mode, method identity, exact event counts, one finite
positive orchestration/I/O overhead, validator evidence references and digest,
review timestamp, and deterministic self-digest. It is never inferred.
`price.json` has exactly
`price_per_hour`, `currency`, `billing_unit`, `pricing_timestamp`, `source`, and
`vm_count`. Unknown fields or invalid values fail closed.

```bash
"$PYTHON" -m creditrep.experiments.p7c4b2c_cli validate-artifacts \
  --run-dir "$OUT_P1"
"$PYTHON" -m creditrep.experiments.p7c4b2c_cli validate-artifacts \
  --run-dir "$OUT_P2"
"$PYTHON" -m creditrep.experiments.p7c4b2c_cli project \
  --run-dir "$OUT_P1" --run-dir "$OUT_P2" \
  --inner-run "$INNER_P1" --inner-run "$INNER_P2" \
  --overhead-mapping "$OVERHEAD" --price-input "$PRICE"
```

Combined projection requires exactly two runs, one per CPU mode, the same source
SHA and plan digest, and the exact 324 canonical task IDs. It rejects duplicates
and validates every source artifact. It also requires exactly two validator-pass
B2b target run directories, one per mode, with the same lowercase source SHA.
Relocated exports, summaries and failed historical evidence are rejected. Price
freshness is not currently a protocol gate; review its provenance without
inventing a local freshness rule.

## Final gate

Projection eligibility means only that runtime/cost planning inputs are complete.
The output always retains
`canonical_scientific_execution_authorized: false`. A separate reviewed compute
mode decision, proposed execution-plan digest, human approval and canonical
authorization contract are required before any scientific execution command may
be considered.
