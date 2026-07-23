# P3B WOE, VIF and scaling

P3B extends the P3A train-only preprocessing foundation with categorical Weight of Evidence encoding, iterative VIF feature removal and optional standard scaling. It does not implement nested cross-validation, fold persistence, model tuning or publishable experiments.

## Pipeline Order

```text
Input features
  -> P3A mean/mode imputation
  -> categorical WOE encoding
  -> iterative VIF feature selection
  -> optional standard scaling
  -> numeric transformed feature matrix
```

Every `fit()` step uses only the training data passed by the caller. `transform()` never updates fitted state or reproducibility metadata. Per-call diagnostics are returned separately.

## WOE Scope

The paper says nominal input values are replaced by the log of their good:bad odds, or WOE. It does not specify numeric binning, number of bins, monotonic binning or smoothing. P3B therefore applies WOE only to categorical features and leaves imputed numeric features as numeric passthrough. Numeric supervised binning is a documented implementation gap for a future task, not paper-exact behavior.

## WOE Formula

Target convention follows the loader contract:

- `0 = good/non-default`
- `1 = bad/default`

P3B uses:

```text
WOE(category) = ln(smoothed_good_distribution / smoothed_bad_distribution)
```

with configurable smoothing `alpha > 0`:

```text
good_distribution = (good_count + alpha) / (total_good + alpha * category_count)
bad_distribution  = (bad_count  + alpha) / (total_bad  + alpha * category_count)
```

The default `alpha` is `0.5`. This keeps categories that appear in only one class finite. Categories unseen at transform time map to the neutral fallback `0.0`; the fallback is not learned from validation or test data and is recorded separately from fitted mappings.

## VIF

P3B computes:

```text
VIF_j = 1 / (1 - R^2_j)
```

where feature `j` is regressed on the remaining features using NumPy least squares with an intercept. If `R^2` is effectively 1, VIF is represented by a large finite sentinel.

Zero-variance filtering runs before iterative VIF removal. Features with training variance `<= VARIANCE_EPSILON` are unusable and are always removed in input feature order with reason `zero_variance`. `minimum_features_to_keep` does not protect constant features.

If all features are constant, fitting fails fast with a clear no-usable-features error. A single non-constant feature is valid and is kept without fitting a regression VIF model. If the number of usable features after zero-variance filtering is lower than configured `minimum_features_to_keep`, fitting fails fast because the config cannot be satisfied without retaining unusable features.

The iterative policy is:

1. Remove all zero-variance features before VIF calculations.
2. Compute VIF values on the current usable training feature set.
3. Stop when max VIF is at or below the threshold.
4. Remove the highest-VIF feature.
5. Recompute on the remaining features.
6. Stop before VIF-based removal would go below `minimum_features_to_keep`.

Default threshold is `10.0`, default minimum kept features is `1`, and ties are resolved by input feature order. Removal history is stored in metadata.

## Scaling

Standard scaling is optional and disabled by default. When enabled, means and population standard deviations are fitted only on the training matrix after WOE and VIF. Zero-variance feature scales are set to `1.0`, so transformed values remain finite. Scaling keeps DataFrame feature names and index.

## Metadata

Each transformer and the composed `ProtocolAPreprocessingPipeline` expose JSON-serializable metadata containing fitted feature order, learned mappings or statistics, selected features, VIF removal history and final output order. Metadata excludes raw rows, target vectors, validation/test data, local paths and transform diagnostics.

## Tests

P3B includes tests for hand-computed WOE, finite smoothing, unknown category fallback, invalid targets, VIF duplicate/constant/singular behavior, iterative VIF removal, scaler train-only statistics, metadata stability and end-to-end numeric finite output.

## Known Checksum Issues

At the time of P3B, the local full raw-data suite still had three pre-existing checksum mismatches unrelated to P3B:

- `data/raw/gmc/Data Dictionary.xls`
- `data/raw/tc/default of credit card clients.xls`
- `data/raw/th02/publicdict.xls`

These were reproduced at `p2-experiment-foundation-complete` with the same local data. The TH02 auxiliary dictionary checksum was re-baselined on 2026-07-23 to the currently distributed public artifact; see `docs/data-cards/TH02.md`. P3B itself did not modify raw data or `data/checksums-sha256.csv`.

## Remaining P3C Work

P3C still needs nested CV, outer/inner fold generation, fold persistence, per-fold pipeline fitting, artifact integration and model tuning isolation.
