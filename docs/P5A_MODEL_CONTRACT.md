# P5A model contract and configuration foundation

P5A provides the common contract for the Phase 5 classical-model work; it does not implement model training, tuning grids, nested-CV model evaluation, or scientific results.

## Stable IDs and registry

`creditrep.models.registry.MODEL_REGISTRY` is the single source of truth for these stable IDs: `logistic_regression`, `decision_tree`, `random_forest`, and `xgboost`. Each capability declares its family, estimator/library, seed/probability support, expected classes `[0, 1]`, allowed hyperparameters, and whether its factory implementation is available. Duplicate and unknown IDs fail fast. Decision Tree and Random Forest are registered for configuration validation but remain explicitly unimplemented until P5B.

## Configuration

`parse_model_config` accepts a typed `ModelConfig` with `id`, `random_seed`, `hyperparameters`, and `tuning_profile`. The only profiles are `reduced` and `paper_reference`. P5A deliberately does not invent a paper-reference grid: it is a documented placeholder pending evidence. Canonical serialization and SHA-256 config hashes are deterministic.

## Probability and metadata

The positive-class contract remains `0 = good/non-default`, `1 = bad/default`, and `y_score = P(class 1)`. `positive_class_probabilities` maps using `classes_`, never a fixed probability column, and rejects absent class 1, malformed shape, row mismatch, NaN/Inf, and values outside `[0, 1]`. The legacy evaluation helper preserves the same behavior for smoke compatibility.

`ModelArtifactMetadata` records identity, configured/effective parameters, library version, seed, expected/observed classes, mapping, runtime, warning/convergence fields, tuning fields, scope, and publishability as JSON-safe data. Smoke output retains legacy keys while adding this normalized contract. Validation outputs must remain `publishable: false` and must not be represented as scientific results.

## Compatibility and deferred work

P2C smoke Logistic Regression/XGBoost configs, artifact readers, flat metric keys, and the P3/P4 nested-CV/metric contracts remain supported. P5B must implement only the registered classical estimators and select a decision-tree implementation according to the linked decision record. P5C must replace the P3 infrastructure-only fake estimator only with an outer-test-safe model-validation harness.
