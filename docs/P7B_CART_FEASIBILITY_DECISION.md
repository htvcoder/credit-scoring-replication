# P7B CART feasibility decision record

## Status

**Accepted — CART engineering feasibility; final CART-A 12-candidate search space locked for P7C.** Decision date: 2026-08-05.

This decision closes P7B. It is a feasibility and protocol decision only; it is not a scientific result, a candidate ranking, or a claim that any CART configuration is better than another.

## Scope and evidence

The decision covers only `decision_tree` implemented as `sklearn.tree.DecisionTreeClassifier`, with the recorded `c45_to_cart` deviation. It relies on the read-only P7B.2 artifact `artifacts/p7b-cart-feasibility/p7b2-run-002`, created at Git HEAD `989997a9dd8cb792636d99f5e2b243b5775807ed` with a clean working tree, P7A manifest SHA-256 `62b5bddc29017cf152607dab029d790e7c6f822bcf840a814b81a90228025f4e`, and run-config hash `6b6828231991b972c38d0ffc949a9f96ffa27940020ab805c2c80b2c705f02ee`.

The validator re-read the artifact and confirmed 60 unique planned fit identities: 60 completed, 0 failed, 0 pending, and `training_artifacts_validated=true`. All fits completed on the first attempt. Sequential fit time totals 12.998 seconds (median 0.143 seconds/fit; p95 0.419 seconds/fit); the largest observed process-local RSS peak is 230.6 MB. There is no elapsed-time IQR outlier.

The pilot did not assess predictive performance, candidate selection, ranking, scientific model selection, outer-test evaluation, selected-model refit, full-system memory, CPU/I/O, GPU, or direct runtime on GC, TH02, and TC. Its metrics and telemetry are non-publishable engineering evidence.

## Options considered

1. **Keep the complete 12-candidate CART-A grid.** Preserves the predeclared four-depth by three-relative-leaf-fraction design.
2. **Reduce the grid.** Rejected because the pilot did not supply a scientific basis to remove candidates and the estimated resource reduction is not needed for the observed engineering budget.
3. **Run an additional feasibility pilot.** Deferred: useful only if the remaining unmeasured datasets or a different execution environment must be profiled before scheduling.

## Decision and consequences

The final CART search space is the 12 configurations in `configs/protocols/p7c/p7c_cart_final_manifest.yaml`: depths 3, 5, 7, and 9 crossed with `min_samples_leaf` fractions 0.005, 0.01, and 0.02. Dataset order, preprocessing boundary, repeated stratified two-fold outer CV, five-fold inner CV, and seed 42 are carried unchanged from the P7A protocol. The final manifest is independently SHA-256 locked so the historical P7A manifest and completed P7B.2 artifact remain immutable and re-validatable.

The locked CART workload is 5,400 inner candidate-evaluation fits and 90 outer selected-model refits: AC 1,200/20, GC 1,200/20, HMEQ 600/10, TH02 1,200/20, TC 600/10, and GMC 600/10. Applying the pilot-wide mean (12.9976799003 / 60 = 0.2166279983 seconds/fit) mechanically gives 5,400 × 0.2166279983 = 1,169.79 seconds, or 19.50 minutes, of serial fit-only time. This excludes outer refits, retries, preprocessing/artifact overhead, contention, and all unmeasured-dataset variation; it is not a runtime commitment.

## Limits, reconsideration, and next state

The C4.5-to-CART deviation remains material. GC, TH02, and TC have not been benchmarked directly; process-local RSS is not whole-system capacity; and a changed machine, parallel execution plan, data/protocol change, repeated failure, or an operational budget breach requires reconsidering this decision before execution.

At the time of this P7B decision, it unlocked only the CART final search-space prerequisite and did not lock the other model decisions. RF/XGBoost were subsequently locked by `docs/P7C2_RF_XGBOOST_DECISION.md`; MLP-1/3/5, CatBoost, TabNet and FT-Transformer remain governed by their P7C checkpoints. No P7C training is authorized or performed by this P7B record.
