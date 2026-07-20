# Hợp đồng split và experiment artifact P2B

Tài liệu này mô tả phần **P2B - Deterministic split và experiment artifact contract**. P2B chỉ tạo split và artifact metadata phục vụ reproducibility; không train model, không tính metric, không sinh prediction và không chạy smoke experiment P2C.

## Config split

Config YAML nằm trong `configs/experiments/`, ví dụ `configs/experiments/split_gc.yaml`:

```yaml
experiment:
  name: gc_split_validation

dataset:
  id: GC

split:
  strategy: stratified_holdout
  test_size: 0.2
  random_seed: 42
  shuffle: true

output:
  root_dir: artifacts/experiments
```

Contract được parse thành `ExperimentConfig`. Các path trong config phải là relative portable path, không dùng drive letter, path tuyệt đối hoặc `..`.

## Stratified deterministic holdout

Splitter chính là:

```python
from creditrep.splitting import create_split

split_result = create_split(
    dataset=loaded,
    strategy="stratified_holdout",
    test_size=0.2,
    random_seed=42,
)
```

Split dùng target đã chuẩn hóa từ P2A (`0 = non-default/good`, `1 = default/bad`). Mỗi class được shuffle bằng seed cố định, lấy số dòng test theo `round(class_count * test_size)`, rồi lưu row position vào train/test. Contract dùng `row_position`, không dùng business identifier.

Splitter fail-fast nếu dataset rỗng, target không phải `{0, 1}`, target chỉ có một class, class quá nhỏ, features/target lệch số dòng, index duplicate, split overlap hoặc split làm mất dòng.

## Hash reproducibility

`config_hash` là SHA-256 trên JSON canonical của config đã parse và normalize. Hash này không phụ thuộc YAML formatting hoặc thứ tự key.

`split_hash` là SHA-256 trên JSON canonical gồm dataset ID, source file portable, checksum active file, split strategy, test size, random seed, train indices và test indices. Hash không dùng Python built-in `hash()` và không chứa absolute local path.

Dataset checksum lấy từ `data/checksums-sha256.csv`. Artifact ghi cả checksum khai báo và checksum tính thực tế; nếu mismatch thì fail-fast.

## Artifact structure

CLI P2B tạo thư mục:

```text
artifacts/
└── experiments/
    └── <experiment_id>/
        ├── manifest.json
        ├── config.yaml
        ├── split.json
        └── split.csv
```

`manifest.json` chứa schema version, experiment ID/name, trạng thái `split_created`, dataset metadata, checksum, split counts, `split_hash`, `config_hash`, Git commit và dirty state. Các field `metrics`, `predictions`, `trained_model` và `plots` được để reserved `null` cho phase sau.

`split.csv` chỉ chứa:

```text
row_position,partition
0,train
1,test
```

File này không chứa feature values hoặc target values. `split.json` lưu metadata để validate lại `split.csv`, gồm `split_hash_payload` và số dòng.

Artifact được ghi atomic qua temporary directory rồi rename sang thư mục cuối. Writer không overwrite experiment directory đã tồn tại. Generated artifacts trong `artifacts/experiments/` bị ignore và không commit.

## CLI

Tạo split artifact:

```powershell
python scripts/create_split_artifact.py --config configs/experiments/split_gc.yaml
python scripts/create_split_artifact.py --config configs/experiments/split_tc.yaml
```

CLI load config, load dataset bằng P2A, kiểm tra checksum, tạo split, ghi artifact và in summary an toàn. CLI không in raw records.

## Load và validate split definition

P2C có thể dùng:

```python
from creditrep.artifacts.split_definition import load_split_csv, validate_split_definition

assignments = load_split_csv("artifacts/experiments/.../split.csv")
```

Validation kiểm tra duplicate row position, partition không hợp lệ, missing/extra rows và `split_hash` mismatch.

## Ngoài phạm vi P2B

P2B không làm imputation, encoding, scaling, WOE, VIF, model training, hyperparameter tuning, metric, prediction artifact, nested CV hoặc website result publishing. Các phần này thuộc P2C và các phase sau.
