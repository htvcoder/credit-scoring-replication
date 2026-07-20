# Smoke experiment runner P2C

P2C bổ sung smoke runner để kiểm tra pipeline end-to-end trên một số cấu hình nhỏ. Kết quả smoke chỉ chứng minh code chạy được và artifact hợp lệ; **không phải kết quả nghiên cứu**, không dùng để so sánh khoa học và không đưa lên website như metric chính thức.

## Config bắt buộc

Ba config smoke bắt buộc:

- `configs/experiments/smoke_gc_lr.yaml`: GC + Logistic Regression.
- `configs/experiments/smoke_gc_xgb.yaml`: GC + XGBoost.
- `configs/experiments/smoke_tc_lr.yaml`: TC + Logistic Regression.

Mỗi config đặt:

- `experiment.purpose: smoke_validation`
- `experiment.publishable: false`
- deterministic split giống P2B;
- `preprocessing.mode: smoke_baseline`;
- `evaluation.classification_threshold: 0.5`;
- model type chỉ là `logistic_regression` hoặc `xgboost`.

## Preprocessing smoke baseline

Preprocessing chỉ đủ để LR/XGBoost chạy được:

- numeric: median imputation; Logistic Regression thêm `StandardScaler`;
- categorical: most-frequent imputation, one-hot encoding, `handle_unknown="ignore"`;
- schema numeric/categorical lấy từ `data/datasets.yaml` qua metadata P2A.

Tất cả transformer nằm trong sklearn `Pipeline`/`ColumnTransformer` và chỉ fit trên training split. Test set chỉ được transform bằng state đã học từ training set. P2C không triển khai WOE, VIF, feature selection, scaling protocol khoa học đầy đủ, nested CV hoặc tuning.

## Model và metrics

Model factory chỉ hỗ trợ:

- Logistic Regression từ scikit-learn;
- XGBoost `XGBClassifier` chạy CPU, `n_jobs: 1`, không GPU, không early stopping, không tuning.

Runner kiểm tra `predict_proba`, xác định đúng xác suất class `1 = default/bad` qua `classes_`, reject NaN/Infinity hoặc probability ngoài `[0, 1]`.

Metrics test set:

- ROC AUC;
- accuracy;
- precision;
- recall;
- F1;
- log loss;
- Brier score;
- confusion matrix;
- predicted positive rate;
- row counts và classification threshold.

Không tính train metric như kết quả chính, không có confidence interval, statistical testing, calibration curve hoặc profit/cost metric.

## Artifact sau P2C

Runner mở rộng artifact P2B:

```text
artifacts/
└── experiments/
    └── <experiment_id>/
        ├── manifest.json
        ├── config.yaml
        ├── split.json
        ├── split.csv
        ├── metrics.json
        ├── predictions.csv
        └── model_metadata.json
```

`predictions.csv` chỉ chứa test rows với các cột:

```text
row_position,partition,y_true,y_score,y_pred
```

Không lưu raw features, transformed matrix, training records hoặc model binary.

`metrics.json` chứa `publishable: false`, `result_scope: smoke_validation`, `split_hash`, `prediction_hash`, threshold và metrics test set.

`model_metadata.json` chứa model type, parameters, preprocessing steps, library versions, fit/prediction duration, random seed, feature count trước preprocessing, transformed feature count và fit warnings nếu có.

`manifest.json` có `status: completed`, `result_scope: smoke_validation`, `publishable: false`, model metadata, metrics/predictions file và provenance Git. Nếu working tree đang có thay đổi chưa commit, `git_dirty` được ghi là `true`.

## CLI

Chạy smoke runner:

```powershell
python scripts/run_experiment.py --config configs/experiments/smoke_gc_lr.yaml
python scripts/run_experiment.py --config configs/experiments/smoke_gc_xgb.yaml
python scripts/run_experiment.py --config configs/experiments/smoke_tc_lr.yaml
```

CLI in summary an toàn, gồm experiment ID, dataset, model, row counts, split hash, ROC AUC, accuracy, F1, prediction hash và artifact directory. CLI không in raw records.

## Reproducibility

Cùng dataset checksum, config, split hash và seed sẽ giữ cùng split hash. Với Logistic Regression smoke, rerun cùng config cũng giữ cùng prediction hash và metrics trong kiểm tra hiện tại. Timestamp chỉ ảnh hưởng experiment ID và artifact directory, không ảnh hưởng split hash hoặc prediction hash.

Generated artifacts trong `artifacts/experiments/` bị ignore và không commit.
