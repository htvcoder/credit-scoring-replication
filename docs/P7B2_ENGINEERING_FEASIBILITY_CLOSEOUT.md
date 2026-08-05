# P7B.2 CART engineering-feasibility closeout

## Scope and provenance

This is a read-only audit of `artifacts/p7b-cart-feasibility/p7b2-run-002`; the original artifact was not modified, moved, renamed, or rerun. It is an engineering-feasibility run, not a scientific experiment: `purpose=engineering_feasibility`, `publishable=false`, `candidate_selection=none`, `predictive_ranking=false`, `scientific_model_selection=false`, and `outer_selected_model_refit=false`.

The run used `sklearn.tree.DecisionTreeClassifier` as the documented C4.5-to-CART deviation, manifest SHA-256 `62b5bddc29017cf152607dab029d790e7c6f822bcf840a814b81a90228025f4e`, run-config hash `6b6828231991b972c38d0ffc949a9f96ffa27940020ab805c2c80b2c705f02ee`, Git HEAD `989997a9dd8cb792636d99f5e2b243b5775807ed`, and a clean working tree.

## Artifact validation and coverage

`validate-artifacts` re-read the plan, configuration snapshot, environment, engineering summary, and all 60 fit records. It returned `valid=true`, `training_artifacts_validated=true`, `completion_status=completed`, 60 completed, 0 failed, and 0 pending. The deterministic pilot covers AC, HMEQ, and GMC; one outer partition (`repeat_00_fold_00`); four fixed CART-A candidates; and five inner folds, for 20 fits per dataset. Every fit completed on its first attempt.

No predictive metric, candidate winner, ranking, outer-test evaluation, selected-model refit, or publishable result was produced or inferred.

## Observed runtime and memory

All fits ran sequentially in one process. The span between the first recorded start and last recorded end was 13.041 seconds; the sum of per-fit elapsed wall times was 12.998 seconds. Per-fit elapsed time was 0.217 seconds mean, 0.143 seconds median, 0.419 seconds p95, and 0.422 seconds maximum.

| Dataset | Fits | Training rows per fit | Mean elapsed | p95 elapsed | Max RSS peak | Mean RSS delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AC | 20 | 274–276 | 0.150 s | 0.183 s | 188.1 MB | 0.19 MB |
| HMEQ | 20 | 2,383–2,384 | 0.116 s | 0.125 s | 192.9 MB | 0.26 MB |
| GMC | 20 | 59,999–60,001 | 0.383 s | 0.421 s | 230.6 MB | 29.82 MB |
| All | 60 | 274–60,001 | 0.217 s | 0.419 s | 230.6 MB | 10.09 MB |

The largest elapsed fit was GMC / `cart_a_high_complexity` at 0.422 s. The IQR upper outlier threshold was 0.713 s, so no elapsed-time outlier was observed. Chronological RSS endpoints rose by 1.19 MB (AC), 2.76 MB (HMEQ), and 5.19 MB (GMC); this small one-run endpoint change is not enough to establish a memory leak.

RSS was sampled every 0.05 seconds with `psutil.Process.memory_info().rss`; it is process-local, excludes child processes and system-wide memory, and can miss a short peak between samples. `tracemalloc` is Python-allocation telemetry only and is not RAM capacity evidence. CPU utilisation, disk I/O, GPU telemetry, and parallel-worker memory were not recorded.

## Engineering feasibility and scaling estimate

The pilot provides a positive technical signal for sequential CART fitting on the observed machine: no fit failure, no retry, and a bounded observed process RSS peak. It does **not** demonstrate production capacity, full-workload completion, or scientific validity.

The locked CART-A grid has 12 candidates and the manifest states 5,400 inner candidate-evaluation fits across the six-dataset outer-CV protocol. The pilot has four candidates and one outer partition, so its measured inner-fit work is a 60-fit feasibility sample, not a direct whole-protocol benchmark. Linear sequential estimates using the observed mean time per pilot dataset are:

| Dataset | Full inner fits | Pilot basis | Linear serial estimate |
| --- | ---: | --- | ---: |
| AC | 1,200 | 20 pilot fits × 60 | 180.5 s (3.0 min) |
| HMEQ | 600 | 20 pilot fits × 30 | 69.6 s (1.2 min) |
| GMC | 600 | 20 pilot fits × 30 | 230.1 s (3.8 min) |
| GC | 1,200 | no pilot measurement | unvalidated |
| TH02 | 1,200 | no pilot measurement | unvalidated |
| TC | 600 | no pilot measurement | unvalidated |

For all 5,400 inner fits, applying the all-pilot mean mechanically gives 1,169.8 seconds (19.5 minutes) of fit-only serial time. This is an illustrative lower-bound-style extrapolation, not a scheduling commitment: the remaining datasets are unmeasured, actual preprocessing/artifact overhead and machine contention vary, and retries are excluded. The manifest's 90 outer selected-model refits are also excluded because P7B did not measure them.

## Decision boundary and next action

P7B.2 is complete as an engineering checkpoint. P7B closeout accepted feasibility and locked the complete CART-A grid in `configs/protocols/p7c/p7c_cart_final_manifest.yaml`; see `docs/P7B_CART_FEASIBILITY_DECISION.md`. This is not P7C execution: GC/TH02/TC remain unbenchmarked and the search spaces/budgets for other models remain unlocked. Any P7C run must retain the C4.5-to-CART deviation, leakage-safe nested-CV contract, and non-publication boundary until scientific acceptance criteria are met.
