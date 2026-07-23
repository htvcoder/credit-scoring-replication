# P4C - EMP và metric-validation harness

## 1. Scope

P4C hoàn tất checkpoint cuối của Phase 4:

- chốt quyết định cuối cho EMP;
- bổ sung metric config contract;
- nối metric computation vào nested-CV foundation của P3C;
- tạo metric-validation artifacts non-publishable;
- giữ backward compatibility cho smoke runner P2C.

P4C không triển khai model factory Phase 5, không chạy core replication và không tạo scientific result.

## 2. Audit summary

Audit trước khi sửa xác nhận:

- P3C đã có deterministic nested CV, fold hashes, outer-test isolation và non-publishable preprocessing-validation artifacts.
- P4B đã có production implementation/reference validation cho ROC AUC, Brier Score và Partial Gini.
- Smoke runner P2C vẫn phụ thuộc contract flat `test_metrics`.
- Repository chưa có EMP implementation hay dependency chuyên dụng.

## 3. EMP final decision

Quyết định cuối của P4C là:

- `metric_id = emp`
- `status = unsupported`
- `value = null`
- `exactness = not_applicable`

### Nguồn chính

- Gunnarsson et al. (2021), mục metric business.
- Verbraken et al. (2014), *Development and application of consumer credit scoring models using profit-based classification measures*.

### Formula paper nêu

```text
P(t; b1, c0, c*) = (b1 - c*) * pi1 * F1(t) - (c0 + c*) * pi0 * F0(t)
MP = max_t P(t; b1, c0, c*)
EMP = E_h[MP]
```

### Lý do unsupported

P4C không có provenance đủ rõ cho:

- `b1`
- `c0`
- `c_star`
- joint density `h(b1,c0)`
- threshold-selection policy để dùng trong evaluation mà không tạo leakage

Vì vậy repository không tự bịa business assumptions để ép EMP trả numeric value.

## 4. Sources và assumptions

- Positive class giữ `1 = bad/default`.
- `y_score` giữ là xác suất class `1`.
- Threshold outer test chỉ được dùng cho evaluation metric; không được chọn trên outer test để tune model hay làm operational threshold.
- Validation harness dùng estimator deterministic chỉ để chứng minh seam integration, không phải model replication của Phase 5.

## 5. Metric config schema

P4C thêm config parser riêng cho metric validation:

- `src/creditrep/config/metric_validation.py`
- `src/creditrep/metrics/registry.py`

Schema thực tế:

```yaml
experiment:
  name: metric_validation_gc_reduced
  result_scope: metric_validation
  publishable: false
dataset:
  id: GC
cross_validation:
  outer:
    strategy: repeated_stratified_2fold
    n_repeats: 1
    n_splits: 2
    shuffle: true
    random_seed: 42
  inner:
    strategy: stratified_kfold
    n_splits: 2
    shuffle: true
    random_seed_policy: derived_from_outer
preprocessing:
  protocol_config: configs/protocols/protocol_a.yaml
evaluation:
  validation_model: deterministic_probability_estimator
  metrics:
    - id: roc_auc
    - id: brier_score
    - id: partial_gini
      parameters:
        b: 0.4
    - id: emp
```

Validation rules:

- reject unknown metric ID;
- reject duplicate metric;
- reject unknown parameter;
- inject default `b = 0.4` cho Partial Gini;
- reject `b <= 0` hoặc `b >= 1`;
- EMP Phase 4 không nhận business parameter tùy ý.

## 6. Nested-CV integration point

Integration point nằm trong:

- `src/creditrep/experiments/metric_validation.py`

Flow:

```text
outer fold
-> fit preprocessing on outer-train
-> inner-fold candidate scoring on outer-train only
-> fit selected estimator on outer-train
-> predict probability on outer-test
-> compute configured metrics
-> persist fold-level metric artifacts
```

P3C isolation được giữ nguyên:

- outer test không dùng để fit preprocessing;
- outer test không dùng để tune candidate;
- outer test labels chỉ đọc ở bước metric evaluation;
- không có row-level prediction trong public validation artifact.

## 7. Artifact schema

Artifact writer mới:

- `src/creditrep/artifacts/metric_validation.py`

Files chính:

- `manifest.json`
- `config.yaml`
- `fold_metrics.json`
- `metrics_summary.json`
- `prediction_summary.json`
- `nested_cv/outer_folds.json`
- `nested_cv/outer_folds.csv`
- `nested_cv/outer/<fold>/split.json`
- `nested_cv/outer/<fold>/preprocessing.json`
- `nested_cv/outer/<fold>/tuning_summary.json`
- `nested_cv/outer/<fold>/metrics.json`

Flags bắt buộc:

```yaml
publishable: false
result_scope: metric_validation
```

Mỗi fold metric record lưu:

- `outer_fold_id`
- `metric_id`
- `metric_version`
- `value`
- `direction`
- `status`
- `parameters`
- `exactness`
- `warnings`
- `split_hash`
- `config_hash`
- `dataset_checksum`
- `protocol_config_hash`
- `git_commit`
- `seed`
- `result_scope`

## 8. Validation harness

CLI mới:

```powershell
python scripts/run_metric_validation.py --config configs/experiments/metric_validation_gc_reduced.yaml --dry-run
```

Config reduced đã được thêm:

- `configs/experiments/metric_validation_gc_reduced.yaml`

Harness này chỉ chứng minh:

- config validation;
- metric registry hoạt động;
- nested-CV evaluation seam đúng thứ tự;
- artifact contract deterministic;
- EMP unsupported được serialize có provenance.

## 9. Test evidence

Test mới:

- `tests/test_p4c_emp.py`
- `tests/test_p4c_metric_config_and_integration.py`

Coverage chính:

- EMP unsupported metadata ổn định;
- metric config validation và deterministic hash;
- integration chỉ dùng outer-test labels ở bước cuối;
- artifact round-trip và CLI dry-run;
- backward compatibility với smoke runner và P3C.

## 10. Commands

Các lệnh xác minh chính:

```powershell
python -m pytest tests/test_p4c_emp.py tests/test_p4c_metric_config_and_integration.py -q
python -m pytest tests/test_smoke_runner.py tests/test_p3c_nested_cv.py tests/test_p4a_metric_contract.py tests/test_p4b_validated_metrics.py -q
python scripts/run_metric_validation.py --config configs/experiments/metric_validation_gc_reduced.yaml --dry-run
```

## 11. Limitations

- EMP chưa được dùng như metric số.
- Validation estimator là deterministic fixture phục vụ integration, không phải baseline model của Phase 5.
- Metric-validation artifacts vẫn là non-publishable validation artifacts.

## 12. Ảnh hưởng tới Phase 5-9

- Phase 5 có thể tái sử dụng metric registry và artifact contract này.
- Phase 7/9 phải tiếp tục ghi rõ EMP là unsupported trừ khi có business specification có provenance mới và một decision record mới.
- Final report không được gọi EMP là replicated metric trong trạng thái hiện tại.
