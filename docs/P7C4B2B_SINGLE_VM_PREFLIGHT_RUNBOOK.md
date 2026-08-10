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

The runner uses spawn-safe isolated workers, thread environment caps before its
lazy numerical/model imports, process-tree timeout cleanup, atomic temporary
attempt promotion, stable retry identity, idempotent resume, reconstructed
summaries and a fail-closed artifact validator. Projection remains pending for
development fixtures. Target evidence is stratified; two-VM efficiency must be
supplied below 100%, GPU remains pending, and cost requires operator pricing.

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

```powershell
$env:PYTHONNOUSERSITE='1'
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2b_cli validate-plan
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2b_cli profile-machine --machine-role intended_single_vm_target --provider <provider> --instance-type <instance> --profile-output target-profile.json
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2b_cli run --mode cpu_parallel_1 --profile target-profile.json --target-machine-asserted --bounded-preflight-authorized --output-dir artifacts/p7c4b2b-compute-preflight/<run-id-p1>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2b_cli run --mode cpu_parallel_2 --profile target-profile.json --target-machine-asserted --bounded-preflight-authorized --output-dir artifacts/p7c4b2b-compute-preflight/<run-id-p2>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2b_cli validate-artifacts --output-dir artifacts/p7c4b2b-compute-preflight/<run-id-p1>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2b_cli validate-artifacts --output-dir artifacts/p7c4b2b-compute-preflight/<run-id-p2>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2b_cli project --input-run artifacts/p7c4b2b-compute-preflight/<run-id-p1> --input-run artifacts/p7c4b2b-compute-preflight/<run-id-p2> --price-input <operator-price.json> --two-vm-efficiency <measured-value-below-1>
```

Canonical execution remains blocked while outer-refit runtime and defensible
canonical orchestration overhead are unknown and extrapolation remains 1,500x.
`propose-execution-plan` fails closed for this incomplete projection; do not run
it until a future evidence contract makes total elapsed execution-plan eligible.
After that, a compute-mode decision, exact execution-plan digest and human
execution/cost approval are still required.
The proposed plan remains deliberately unapproved; an approval record must bind
its exact digest and contain a human approver and timestamp before a later
canonical guard can authorize execution.
