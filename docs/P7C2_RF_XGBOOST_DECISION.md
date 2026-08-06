# P7C.2.3 — RF/XGBoost final scientific protocol decision

## Decision metadata

- Decision IDs: `DR-P7C-01` (Random Forest) and `DR-P7C-02` (XGBoost).
- Version: `1.0.0`.
- Decision date: 2026-08-06.
- Approval provenance: the user instruction for task P7C.2.3, recorded as `user_task_instruction` in the final manifest. No approval timestamp is asserted beyond that instruction.
- Final manifest: `configs/protocols/p7c/p7c_rf_xgboost_final_manifest.yaml`.

## Evidence inputs and boundary

The immutable P7C.2 pilot plan remains `configs/protocols/p7c/p7c2_rf_xgboost_pilot_plan.yaml`, with digest `1f3a6cd5b9f4d766fe89b34676ba66cf3ec731b49b27ff3769af042d83f08516`. Its completed artifact is `artifacts/p7c2-rf-xgboost-feasibility/run-001`; the read-only evidence analysis is `docs/P7C2_RF_XGBOOST_FEASIBILITY_RESULTS.md`.

`validate-plan` and `validate-artifacts` established 60 expected and completed fits, with 0 failed, 0 missing, 0 unexpected, and 0 duplicate artifacts. The pilot ran sequentially on CPU with one estimator thread, no outer refit, no final nested-CV execution, and no predictive metric/ranking/selection. It is non-publishable engineering evidence only.

The pilot covered AC and GMC, one predeclared outer partition, five inner folds, and three predeclared candidates per model. It does not establish predictive quality, general runtime guarantees, whole-system RAM capacity, or final nested-CV completion.

## Approved final decision

The project prioritizes replication fidelity. The approved final scientific search spaces are the complete P7A/Table-2 reference grids:

| Model | Final selection | Candidate count | Exact parameter grid |
| --- | --- | ---: | --- |
| Random Forest | Full reference grid | 30 | `n_estimators={100,250,500,750,1000}` × `max_features_multiplier_of_sqrt_m={0.1,0.25,0.5,1,2,4}` |
| XGBoost | Full reference grid | 108 | `n_estimators={50,100,150}` × `max_depth={1,2,3}` × `learning_rate={0.3,0.4}` × `colsample_bytree={0.6,0.8}` × `subsample={0.5,0.75,1.0}` |

No reduced grid is selected. The completed pilot contained no preregistered rule that could validly map its engineering telemetry to a reduced scientific grid; no predictive result is used post hoc to remove candidates. No additional feasibility pilot is required by this decision.

The planning worksheet estimates approximately 41–43 sequential CPU-hours for the combined full RF/XGBoost workload with scheduling buffer. This is a resource-planning estimate, not a runtime guarantee or a basis for extrapolating pilot performance linearly to all datasets.

## Remaining deviations and execution authorization

The project still differs from the paper where recorded in P7A, including the six-public-dataset partial-replication scope, project seed because the paper seed is not reported, contemporary library implementations, and the separate C4.5-to-CART deviation. This RF/XGBoost decision does not create a new reduced-grid deviation.

P7C.2 is completed as a protocol-decision checkpoint after the manifest and validation suite pass. Full scientific execution remains **not authorized** by this decision: it requires resource scheduling plus the unified concurrency, retry/failure, retention, and readiness policies of P7C.7, after P7C.3–P7C.7 have been resolved as required. No training, pilot, or full nested-CV execution is performed by this decision record.
