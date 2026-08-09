# P7C.4B.2b — Single-VM CPU preflight runbook

The executable bounded-preflight harness is ready; target preflight remains
pending. This development machine is `development_calibration_only`; fixture
tests are non-benchmark evidence and cannot decide feasibility, cost, or mode.

The scientific manifest remains locked at
`4d8636c3606e07e243efd2bc7be12806e7adf4fc1b19dbe0dc113a35adc57f75`.
The immutable preflight plan digest is
`8e0f4d2c819ee4b2c89d0282fbaebf0601483e7f25db040c76307057eb3b1d5e`.
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
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2b_cli propose-execution-plan --mode cpu_parallel_2 --input-run artifacts/p7c4b2b-compute-preflight/<run-id-p1> --input-run artifacts/p7c4b2b-compute-preflight/<run-id-p2> --price-input <operator-price.json>
```

Canonical execution remains blocked pending target preflight, compute-mode
decision, exact execution-plan digest and human execution/cost approval.
The proposed plan remains deliberately unapproved; an approval record must bind
its exact digest and contain a human approver and timestamp before a later
canonical guard can authorize execution.
