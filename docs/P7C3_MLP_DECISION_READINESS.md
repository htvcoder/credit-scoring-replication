# P7C.3 — MLP-1/3/5 decision-readiness and preregistered feasibility plan

## Status

**In progress — plan prepared, not run.** P7C.3 does not lock the final MLP scientific grids or compute budget yet. It prepares the evidence needed to make that decision without using predictive performance post hoc.

## Fact: reference protocol and implementation readiness

The paper's Table 2 includes MLP with one, three and five hidden layers. Its shared tuning values are hidden units `{5,10,15,20}`, dropout `{0,0.25,0.5}`, L2 `{0.1,0.01,0.001,0}`, batch normalization `{Yes,No}`, and depth-specific learning rates: MLP-1/3 `{1e-2,1e-3,1e-4}` and MLP-5 `{1e-3,1e-4,1e-5}`. P7A records reference counts 144, 720 and 2016, including the documented Table-2/batch-normalization count ambiguity.

The repository implements and registers all three depths. The depth is paper-exact; width 64, ReLU, Adam at 0.001, batch size 32, 200 epochs, no regularization, seed 42 and early stopping are project decisions. The estimator is leakage-safe when called through P6C nested-CV integration, returns `predict_proba[:, 1] = P(bad/default)`, uses deterministic seed derivation, and has atomic artifact, retry and resume contracts. This establishes implementation readiness, not scientific evidence.

## Fact: existing feasibility evidence and gap

P6C GC reduced validation completed 6/6 folds at four epochs; P6C TC reduced checkpoint completed 6/6 folds at two epochs after one retryable publication failure. These are non-publishable engineering artifacts. TC observed final-refit durations of 0.754–1.005 s (MLP-1), 0.818–1.427 s (MLP-3), and 1.606–1.906 s (MLP-5), but peak memory was unavailable and the documents explicitly prohibit extrapolation to a full workload.

Therefore, no repository evidence supports a bounded full-grid CPU/GPU budget, peak RSS/VRAM requirement, timeout threshold, or decision to exclude MLP-5. The current evidence also does not demonstrate that GPU is required; it only establishes that CPU completed reduced checkpoints and may be slow at larger scope.

## Decision-readiness conclusion

P7C.3 follows **Branch B**. MLP-1, MLP-3 and MLP-5 remain in the intended core replication scope; none is excluded. Their P7A reference grids remain reference-only and unlocked. A bounded, immutable, non-publishable engineering plan is recorded at `configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml`.

The plan fixes TC/GMC, one deterministic outer partition, five inner folds, two predeclared engineering candidates per depth, CPU-only sequential execution and 60 fits. It deliberately contains no predictive metric, ranking, candidate selection, outer refit, scientific result, or rule that can reduce the final grid.

The plan cannot be executed until the user/mentor explicitly approves the unresolved operational thresholds recorded in the plan: maximum wall time, peak RSS, timeout, and CPU/GPU decision thresholds. Until then, every compute decision is preregistered as **inconclusive**, not silently treated as pass or fail.

## Recommendation

Approve the operational thresholds and execution authorization for the immutable P7C.3 plan, or directly approve a scientifically justified full-grid compute budget. Do not move to P7C.4 or start scientific execution while P7C.3 remains in progress.
