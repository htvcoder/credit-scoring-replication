# P5A decision record: Decision Tree implementation for P5B

## Status

Accepted: use sklearn CART (`sklearn.tree.DecisionTreeClassifier`) with stable model ID `decision_tree`. The algorithm is `cart`, its replication role is `approximation`, and its required deviation metadata is `c45_to_cart`.

## Evidence and options

The current repository plan says “Decision Tree/CART or C4.5 if feasible”, but contains no verified source establishing which exact tree algorithm/hyperparameters the original paper used. P5A therefore does not infer this choice.

Python options assessed for P5B are:

- `sklearn.tree.DecisionTreeClassifier` (CART): maintained through the existing pinned scikit-learn dependency, deterministic with `random_state`, binary classification/probabilities, nested-CV compatible, BSD-3-Clause, and no new fragile dependency.
- A third-party C4.5 implementation: cannot be selected without checking maintenance status, license, probability behavior, deterministic seed handling, and compatibility with the pinned Python/NumPy/scikit-learn stack.

## Accepted implementation

Use `DecisionTreeClassifier`, label it **CART decision tree**, retain model ID `decision_tree`, and propagate the explicit CART-vs-C4.5 deviation to model metadata and the final report. Do not label CART as C4.5.

The selected estimator is maintained, deterministic with `random_state`, supports binary probabilities, is compatible with nested CV, is part of the existing BSD-3-Clause scikit-learn dependency, and adds no fragile dependency. The final report must retain the deviation. No third-party C4.5 dependency is added.
