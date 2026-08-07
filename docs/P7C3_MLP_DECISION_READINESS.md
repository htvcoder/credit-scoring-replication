# P7C.3 — MLP-1/3/5 decision-readiness and preregistered feasibility plan

## Status

**In progress — hotfix in verification.** `vm-run-001` at commit `c928c118da6711ed714bd9b0ddc1f73c1e098e00` failed 60/60 before training because the P7C.3 plan schema was passed directly to an incompatible MLP factory contract. The failed evidence is retained; it supplies no runtime/RSS/projection evidence and does not lock the final MLP scientific grids or compute budget.

## Fact: reference protocol and implementation readiness

The paper's Table 2 includes MLP with one, three and five hidden layers. Its shared tuning values are hidden units `{5,10,15,20}`, dropout `{0,0.25,0.5}`, L2 `{0.1,0.01,0.001,0}`, batch normalization `{Yes,No}`, and depth-specific learning rates: MLP-1/3 `{1e-2,1e-3,1e-4}` and MLP-5 `{1e-3,1e-4,1e-5}`. P7A records reference counts 144, 720 and 2016, including the documented Table-2/batch-normalization count ambiguity.

The repository implements and registers all three depths. The depth is paper-exact; width 64, ReLU, Adam at 0.001, batch size 32, 200 epochs, no regularization, seed 42 and early stopping are project decisions. The estimator is leakage-safe when called through P6C nested-CV integration, returns `predict_proba[:, 1] = P(bad/default)`, uses deterministic seed derivation, and has atomic artifact, retry and resume contracts. This establishes implementation readiness, not scientific evidence.

## Fact: existing feasibility evidence and gap

P6C GC reduced validation completed 6/6 folds at four epochs; P6C TC reduced checkpoint completed 6/6 folds at two epochs after one retryable publication failure. These are non-publishable engineering artifacts. TC observed final-refit durations of 0.754–1.005 s (MLP-1), 0.818–1.427 s (MLP-3), and 1.606–1.906 s (MLP-5), but peak memory was unavailable and the documents explicitly prohibit extrapolation to a full workload.

Therefore, no repository evidence supports a bounded full-grid CPU/GPU budget, peak RSS/VRAM requirement, timeout threshold, or decision to exclude MLP-5. The current evidence also does not demonstrate that GPU is required; it only establishes that CPU completed reduced checkpoints and may be slow at larger scope.

## Decision-readiness conclusion

P7C.3 follows **Branch B**. MLP-1, MLP-3 and MLP-5 remain in the intended core replication scope; none is excluded. Their P7A reference grids remain reference-only and unlocked. A bounded, immutable, non-publishable engineering plan is recorded at `configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml`.

The plan fixes TC/GMC, one deterministic outer partition, five inner folds, two predeclared engineering candidates per depth, CPU-only sequential execution and 60 fits. It deliberately contains no predictive metric, ranking, candidate selection, outer refit, scientific result, or rule that can reduce the final grid.

The execution-ready plan locks the approved project operational policy: CPU-only sequential execution, one concurrent fit, two PyTorch/BLAS threads, 30-minute per-fit timeout, one retry only for transient/infrastructure failure, 12-hour feasibility ceiling, 10 GiB RSS warning, 11.5 GiB RSS hard stop, and a 15 GiB disk floor. The 7-day target and 14-day hard CPU projection ceiling are project policy, not paper requirements. Runtime/RSS/projection evidence and the CPU/GPU decision remain **inconclusive** until the 60 fits run.

## Recommendation

Run the approved feasibility command only on the designated CPU VM, then validate its artifacts and produce the separate decision output. Do not move to P7C.4 or start scientific execution while P7C.3 remains in progress.
