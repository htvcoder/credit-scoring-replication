# P7C.4B.2b — Single-VM CPU preflight runbook

The planning contract is complete; executable target-harness hardening and target
preflight are pending. This development machine
has not been identified as `intended_single_vm_target`, so no benchmark result
may be used for feasibility, cost, or canonical-mode decisions.

The immutable plan selects TC/GMC × MLP-1/3/5 × light/median/heavy candidate
from the locked B2a manifest, with two measured repetitions in both
`cpu_parallel_1` and `cpu_parallel_2`. It has 18 logical units, 36 measured
fits per mode and 18 separate warm-ups per mode. Limits: max two workers, two
threads/worker, 1,800 seconds/fit, 12-hour global cap, one transient retry,
16 GiB free disk floor, 11.5 GiB RSS hard cap and 2 GiB artifact guard.

Before target execution, record the required machine profile (role, OS, CPU,
cores, RAM, free disk, interpreter/dependency fingerprint, commit, manifest and
dataset digests, thread policy and UTC boundaries) and validate it. The operator
must run with the project `.venv`; then only an authorized B2b operator command
may create `artifacts/p7c4b2b-compute-preflight/<run-id>/`. This repository
currently supplies deterministic plan/validation/projection contracts, not a
target-execution authorization.

Runtime/cost projections remain pending measured stratified telemetry. Two VM
and GPU are conditional/not authorized; no 100% scaling efficiency, B1 fixture
speedup, or CPU-to-GPU extrapolation is permitted. Canonical execution remains
blocked pending preflight, a canonical mode decision, an execution plan digest,
and human execution/cost approval.
