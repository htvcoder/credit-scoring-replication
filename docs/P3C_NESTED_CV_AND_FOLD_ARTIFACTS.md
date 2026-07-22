# P3C nested CV and fold artifacts

P3C completes the Phase 3 preprocessing foundation. It adds deterministic nested CV, per-fold preprocessing fitting, infrastructure-only tuning isolation and non-publishable fold artifacts. It does not validate scientific metrics, run full model grids, run core experiments or publish results.

## Outer CV

The implemented outer strategy is `repeated_stratified_2fold`: each repeat creates one stratified two-fold split, and each row appears exactly once in outer test per repeat. Fold IDs are stable:

```text
repeat_00_fold_00
repeat_00_fold_01
```

This is a deterministic implementation of the paper's `N x 2` design, not a paper-exact reconstruction of original seeds or fold assignments.

## Inner CV

Inner CV uses `stratified_kfold`, defaulting to five folds in the full contract and two folds in reduced validation configs. Inner folds are generated only from the parent outer training rows. Outer test rows are forbidden in inner train and validation partitions.

## Reduced Config

`configs/experiments/nested_cv_gc_reduced.yaml` is infrastructure-only. It uses one outer repeat and two inner folds to validate artifacts, preprocessing isolation and tuning isolation. It is not a scientific experiment.

## Seed Derivation

Seeds are derived with SHA-256 over canonical JSON:

```python
derive_seed(base_seed, stage, repeat_index, outer_fold_index, inner_fold_index=None)
```

The rule is stable across Python processes and does not use Python built-in `hash()`.

## Hashes

Fold hashes and the nested CV hash use canonical JSON and SHA-256. Hash payloads include dataset ID/checksum, source file, strategies, seeds, row positions and target assignments. Absolute local paths are excluded.

## Leakage Boundaries

Each inner fold fits a fresh `ProtocolAPreprocessingPipeline` on inner train only, then transforms inner train and validation. After tuning, each outer fold fits a fresh final pipeline on full outer train only, then transforms outer train/test. Inner pipelines are not reused as final outer pipelines.

Outer test labels and features are not passed to the candidate evaluator. P3C tests mutate outer test labels/features and verify inner preprocessing metadata and selected candidates do not change.

## Tuning Isolation

P3C includes a fake deterministic estimator/scorer only to validate orchestration. Candidate scores are aggregated from inner validation folds. Ties are resolved by config order. Tuning summaries do not contain outer test metrics.

## Artifacts

The nested artifact is written atomically and refuses overwrite:

```text
artifacts/experiments/<experiment_id>/
├── manifest.json
├── config.yaml
└── nested_cv/
    ├── manifest.json
    ├── outer_folds.json
    ├── outer_folds.csv
    └── outer/<outer_fold_id>/
        ├── split.json
        ├── preprocessing.json
        ├── tuning_summary.json
        └── inner/<inner_fold_id>/
            ├── split.json
            └── preprocessing.json
```

Artifacts set:

```text
publishable: false
result_scope: preprocessing_validation
```

They contain fold definitions, hashes, preprocessing metadata and tuning summaries. They do not contain raw rows, transformed feature matrices, predictions, target vectors, model artifacts, scientific metrics, secrets or absolute local paths.

## CLI

```bash
python scripts/create_nested_cv_artifact.py --config configs/experiments/nested_cv_gc_reduced.yaml
```

The CLI loads data through P2A, verifies checksum, creates nested folds, fits per-fold preprocessing, writes the artifact and prints a short JSON summary.

## Failure Policies

Unsupported CV strategies, invalid split counts, insufficient minority-class rows, checksum mismatch, artifact overwrite, corrupt fold definitions and hash mismatch fail fast.

## Known Checksum Issues

The local raw-data full suite still has pre-existing checksum mismatches for:

- `data/raw/gmc/Data Dictionary.xls`
- `data/raw/tc/default of credit card clients.xls`
- `data/raw/th02/publicdict.xls`

These are not P3C regressions and are not remediated here.

## Out Of Scope

P3C does not implement validated Partial Gini, EMP, full model grids, C4.5 decisions, MLP/CatBoost/TabNet/FT-Transformer, full six-dataset runs, result aggregation, statistical comparison, robustness analysis, public result export or website publication.
