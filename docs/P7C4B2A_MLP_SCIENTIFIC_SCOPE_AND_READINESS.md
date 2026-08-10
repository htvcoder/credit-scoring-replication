# P7C.4B.2a — MLP scientific scope and execution readiness

## Approved scientific decisions

`DR-P7C-03` is approved as the balanced/recommended reduced deterministic subset:
MLP-1 has 24 candidates (12 mandatory, 12 seeded-stratified), MLP-3 has 48
(18, 30), and MLP-5 has 48 (18, 30). `DR-P7C-04` is approved to retain MLP-5
in the replication baseline. This preserves the MLP-1/3/5 depth comparison; it
does not make the subset equivalent to the paper's exhaustive Table-2 search.

The proposed scientific manifest is digest-locked after deterministic materialization
from seed 42. Candidate generation is pure, validates membership in the P7A grid,
and accepts no pilot or predictive metric input.

## Workload and compute boundary

The workload is 54,000 inner fits plus 270 selected-model outer refits = 54,270
fits: MLP-1 10,800 + 90; MLP-3 21,600 + 90; MLP-5 21,600 + 90. Runtime and
cost are `pending_preflight`; B1's four-fit fixture speedup is not an estimator.

Scientific scope is approved, whereas compute feasibility and canonical compute
mode are pending and canonical execution remains not authorized. The proposed
preflight is CPU parallel-2 with at most two workers per VM. GPU is only a
conditional candidate; hybrid and multi-VM execution are not authorized.

## Approval and scaling readiness

Scientific approver: Hoàng Trọng Vĩnh. Scientific reviewer: Trần Công Phú Khánh
(pending). A later, separate execution/cost approval by Hoàng Trọng Vĩnh must
bind both the exact execution-plan digest and this scientific-manifest digest.

Two-VM scaling is planning only. Static shards are deterministically assigned by
`dataset × outer repeat × outer fold` from the scientific digest. Every shard
requires isolated artifacts, identical commit/dependencies/dataset fingerprint/
preprocessing/seed/thread provenance, resume without identity changes, and a
validated deterministic merge proving non-overlap and completeness. The current
runner has no authorized multi-VM executor or merge implementation; P7C.4B.2b
must collect single-VM preflight evidence first.

## Execution guard

The guard fails closed with stable reason codes until preflight, canonical mode,
execution/cost approval, and execution-plan digest exist. GPU and multi-VM
requests additionally return their authorization/readiness codes. No training,
benchmark, GPU, VM, or canonical execution is authorized by this record.
