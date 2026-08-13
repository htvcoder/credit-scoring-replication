# P7C.4B.2e — Fresh Target-Canary Closeout

## Scope and evidence boundary

This closeout audits only the fresh immutable export for run
`p7c4b2d-target-rerun-01`.  It does not re-audit previously reviewed and merged
implementation.  The export remains external to the repository; no raw evidence,
archive, checksum manifest, or local evidence path is committed here.

## Verified operational outcome

- Source identity is `main` at `410cb38375afe29063c4cb80b8d482c7320debbb`.
- The run is `target_preflight` in `cpu_parallel_2`, with one VM and two workers.
- Archive SHA-256 and all 32 entries in its `SHA256SUMS` manifest were verified.
- Environment, proposal, effective authorization, launch record, submission
  receipt, output manifest, four sample records, validation, and run completion
  marker cross-bind the same source SHA, run ID, mode, task set, plan digest,
  environment digest, proposal digest, and authorization digest.
- The exact authorized four-task set completed; no unexpected sample task is
  present. `validation.valid` is `true`, its completed/expected count is `4/4`,
  its evidence digest is
  `a6c2b54441157f11d3640325f42b862bd1a16d5f48570ebfbfaf2197481e5127`,
  and `target_canary_acceptance.accepted` is `true`.

The target canary is therefore operationally accepted. This is an engineering
validation artifact, not a scientific result.

## Scientific boundary retained

`scientific_coverage.valid` remains `false` with exactly these reason codes:

- `incomplete_required_stratum`
- `insufficient_repetitions`

`scientific_projection_eligible` is `false` and
`canonical_scientific_execution_authorized` is `false`. The projection artifact
has `status: incomplete` and `total_canonical_elapsed_seconds: null`; no full
scientific runtime, resource, or cost estimate is inferred from this canary.

## Observed telemetry and resources

The authorization runtime envelope is 461.109 seconds (about 7 minutes 41
seconds). The observed target had 4 vCPU, 15.7 GB RAM, one VM, two workers, no
GPU, and a static price input of USD 0.26/hour. The arithmetic envelope cost is
about USD 0.0333, compared with a 36-hour / USD 20 authorization ceiling. This
is an observed canary resource record only; it is neither billing reconciliation
nor a projection for a scientific run.

## Permitted next step

Keep target compute/outer-refit preflight and canonical scientific execution at
NO-GO unless their separately required evidence and authorization gates are met.
No scientific result is added by this closeout.
