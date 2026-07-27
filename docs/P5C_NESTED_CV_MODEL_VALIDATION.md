# P5C nested-CV model-validation harness

P5C connects the P3C deterministic nested folds, train-only Protocol A preprocessing, P5B classical model factory, and P4 metric registry. It is validation-only: artifacts use `result_scope: model_validation` and `publishable: false`; no scientific results or rankings are produced.

For every outer fold and model, candidates are fit only on inner-train partitions, scored on inner validation using the configured non-EMP metric, and deterministically ranked by score, canonical candidate JSON, then candidate index. A fresh preprocessor and estimator are then fitted on the complete outer train before the single outer-test prediction. Scores always use `P(class 1)`.

The runner emits fold-level predictions without features, P4 metric results (including EMP `unsupported`), preprocessing metadata, P5 metadata/timing, and the CART approximation provenance for `decision_tree`. P5C fixture validation covers LR, CART, Random Forest and XGBoost. Real GC/TC validation remains local-only and non-publishable; Phase 6/7 work is deferred.

Artifacts are written atomically under `artifacts/model-validation/<experiment>-<config-hash>/`; a temporary directory is validated before it is promoted. The contract is:

```
manifest.json                 # config/dataset/Git provenance and state
config.yaml
summary.json
folds/<outer-fold>__<model>/
  fold_metadata.json  tuning.json  preprocessing.json
  model_metadata.json metrics.json predictions.csv complete.json
failures/<fold>.json           # structured, bounded failure report when used
```

`predictions.csv` contains only `row_position`, outer repeat/fold, `partition`, `y_true`, `y_score`, and `y_pred`; feature values are prohibited. A directory existing is never evidence of completion: a fold is skipped only when its complete marker, required files, JSON/CSV parse, provenance, schema, prediction columns/row count, non-publishable scope, and (for CART) provenance all validate. Invalid fold directories are retained under `corrupt/` and rebuilt; provenance mismatch fails rather than mixing evidence.

Resume is now per-fold rather than whole-experiment. The state machine is `pending → running → completed`, or `pending/running → failed`; valid completed units become `skipped/resumed`, while incomplete/corrupt units are rebuilt. A failed unit is retried by `--resume` from fresh preprocessing, tuning, outer refit, predictions and metrics. Its failure report remains as resolved attempt history. Failure reports have an attempt count, first/latest timestamp, resolved state, retryability, stage, exception type and bounded sanitized message. `--fail-fast` stops after recording the first failure; without it the runner continues, reconciles the atomic summary from fold state, and exits non-zero if failures remain. `--validate-only` performs no training or artifact creation.

Malformed JSON/CSV and malformed failure evidence are rejected and never treated as completed or retryable state. Hard experiment, dataset, config, fold, model, or schema provenance mismatches fail instead of being silently rebuilt; recoverable incomplete/corrupt fold output is quarantined before a fresh run. Prediction artifacts require non-negative integral row positions, a single `test` partition, metadata-matching outer repeat/fold values, binary `y_true`/`y_pred`, and finite probabilities in `[0, 1]`.

Candidate failures are isolated: an invalid candidate is excluded when another candidate has valid finite inner scores, and diagnostics are retained in fold warnings. If no candidate remains valid, the unit fails at `inner_tuning`; no outer refit or outer-test prediction occurs. Valid ties are resolved deterministically by score, canonical parameter JSON, then candidate index.

`summary.json` is rebuilt atomically from validated current fold/failure state rather than trusted from a prior run. Planned, completed, failed, invalid, and pending unit identities are sorted and unique, while timestamps remain intentionally volatile. Resolved failures remain historical evidence but are not counted as current failures.

Fixture/CI check: `python -m pytest tests/test_p5c_model_validation.py -vv`.

Local-only reduced configs are `configs/model_validation_gc_reduced.yaml` and `configs/model_validation_tc_reduced.yaml`. Validate configuration, dataset, splits, models, protocol, and artifact target without training with `python scripts/run_model_validation.py --config configs/model_validation_gc_reduced.yaml --validate-only`. Run locally (only when the registered raw data is available) with `python scripts/run_model_validation.py --config configs/model_validation_gc_reduced.yaml`; use `--resume` only for a fully validated completed artifact. A non-zero exit means the invocation failed; no raw records are written to console. Failure reports have an allowed stage, exception type, bounded sanitized message, retryability, cleanup status, and no raw records or traceback. These local runs are non-publishable validation artifacts, not scientific results.
