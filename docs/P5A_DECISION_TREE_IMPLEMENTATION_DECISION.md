# P5A decision record: Decision Tree implementation for P5B

## Status

Unresolved pending primary-paper evidence. The implementation must not claim C4.5 unless the selected library is genuinely C4.5 and the paper evidence requires it.

## Evidence and options

The current repository plan says “Decision Tree/CART or C4.5 if feasible”, but contains no verified source establishing which exact tree algorithm/hyperparameters the original paper used. P5A therefore does not infer this choice.

Python options assessed for P5B are:

- `sklearn.tree.DecisionTreeClassifier` (CART): maintained through the existing pinned scikit-learn dependency, deterministic with `random_state`, binary classification/probabilities, nested-CV compatible, BSD-3-Clause, and no new fragile dependency.
- A third-party C4.5 implementation: cannot be selected without checking maintenance status, license, probability behavior, deterministic seed handling, and compatibility with the pinned Python/NumPy/scikit-learn stack.

## Recommendation for P5B

Use `DecisionTreeClassifier` only if the paper evidence remains unavailable or confirms a generic decision tree. Label it **CART decision tree**, retain model ID `decision_tree`, and propagate the explicit CART-vs-C4.5 deviation to model metadata and the final report. Do not label CART as C4.5.

Before choosing a true C4.5 dependency, add primary-paper evidence and document package version, maintenance, license, deterministic behavior, probability generation, and CI/runtime effects. No new dependency is justified in P5A.
