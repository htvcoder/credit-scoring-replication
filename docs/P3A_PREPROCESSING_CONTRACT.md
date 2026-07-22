# P3A preprocessing contract

P3A adds the train-only preprocessing foundation for Protocol A. It does not replace the P2C smoke baseline and does not produce publishable experiment metrics.

## Contract

`LeakageSafePreprocessor` exposes:

```python
fit(X_train, y_train=None)
transform(X)
transform_with_diagnostics(X)
fit_transform(X_train, y_train=None)
get_metadata()
```

`fit()` learns state only from training features. `transform()` requires fitted state, validates the fitted schema exactly, copies input data before changing values and never refits or extends state. `transform_with_diagnostics()` returns `(transformed, diagnostics)` for per-call diagnostics such as unseen-category counts; those diagnostics are not fitted state and are not included in reproducibility metadata.

## Feature routing

Feature routing uses dataset metadata from `data/datasets.yaml` through the P2A loader metadata. Target, identifier and ignored columns must already be absent from `features`; if they appear, preprocessing fails fast. Numeric and categorical metadata must exactly cover the input DataFrame columns.

## Numeric imputation

Numeric columns use train-only mean imputation. A numeric feature that is entirely missing in training fails fast and names the feature. Missing numeric values during transform are filled with the fitted training mean.

## Categorical imputation

Categorical columns use train-only most-frequent imputation. If multiple values share the highest frequency, the selected mode is chosen by sorting on `(type(value).__name__, repr(value))`. This avoids ambiguity when distinct categorical values have the same string representation, such as integer `1` and string `"1"`. A categorical feature that is entirely missing in training fails fast and names the feature.

## Unseen categories

Each categorical feature stores its training vocabulary. During transform, values outside that vocabulary are mapped to the reserved unknown token from config, default `__UNKNOWN__`. Raw training or transform data containing that reserved token fails fast to avoid ambiguity.

## Metadata

`get_metadata()` returns JSON-serializable state:

- protocol name and version;
- fitted flag;
- numeric and categorical features;
- imputation strategies and fitted values;
- category vocabulary;
- unseen-category strategy and unknown token;
- fitted row count;
- fitted input column order;

It must not contain raw rows, targets, predictions, secrets, absolute local paths or Python objects.

Per-transform diagnostics are returned separately and must not alter `get_metadata()`.

## Not Implemented In P3A

P3A intentionally does not implement WOE, WOE smoothing, VIF, scaling, nested cross-validation, fold persistence, model tuning or model-specific preprocessing.
