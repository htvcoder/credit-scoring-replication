# Metric specification Phase 4

## 1. Phạm vi Phase 4

Phase 4 xác minh metric dự báo và metric kinh doanh trước khi chạy core replication. Phase này không tạo scientific results.

| Checkpoint | Phạm vi | Ngoài phạm vi |
|---|---|---|
| P4A | Audit metric hiện có, chốt specification, contract config/artifact và decision record cho Partial Gini/EMP | Không chạy experiment, không publish metric, không triển khai công thức production của Partial Gini hoặc EMP |
| P4B | Implement/reference-validate ROC AUC, Brier Score và Partial Gini | Không tối ưu business threshold trên outer test |
| P4C | EMP, metric-validation harness, integration vào artifact fold-level | Không công bố validation metrics như kết quả nghiên cứu |

Target chuẩn toàn dự án là `0 = good / non-default` và `1 = bad / default`. `y_score` luôn là xác suất dự đoán cho class `1`, tức xác suất default/bad. Metric nhận probability score phải validate miền `[0, 1]`.

## 2. Metric inventory hiện tại

| Metric | Implementation hiện tại | Nơi dùng | Trạng thái | Thiếu | Backward compatibility |
|---|---|---|---|---|---|
| ROC AUC | `src/creditrep/evaluation/metrics.py::compute_binary_metrics` gọi `sklearn.metrics.roc_auc_score` | P2C smoke runner, `metrics.json`, CLI summary | Implemented cho smoke validation | Chưa có typed `MetricResult`, chưa có fold-level scientific contract | Giữ flat key `roc_auc` trong smoke `test_metrics` |
| Brier Score | `brier_score_loss` trong `compute_binary_metrics` | P2C smoke runner, `metrics.json` | Implemented cho smoke validation | Chưa có version/exactness/warnings | Giữ flat key `brier_score` |
| Accuracy/precision/recall/F1/log loss/confusion matrix | `compute_binary_metrics` | P2C smoke validation | Technical smoke only | Không thuộc metric core của paper, không dùng scientific reporting chính | Giữ để smoke runner không vỡ |
| Classification threshold | `evaluation.classification_threshold` trong smoke config | `build_prediction_frame`, metrics payload | Fixed config value, mặc định fixture là `0.5` | Chưa có nguồn tham số metric/business rõ cho Phase 4 | Giữ behavior hiện tại |
| Partial Gini | Chưa implemented | Chưa dùng | Specification only trong P4A | Cần định nghĩa tính, normalization, tie policy, reference tests | Không ảnh hưởng smoke runner |
| EMP | Chưa implemented | Chưa dùng | `not_implemented_due_to_insufficient_specification` trong P4A | Thiếu business parameters/distribution và threshold policy | Không ảnh hưởng smoke runner |

## 3. Quy ước chung

### Target và score

- `y_true` chỉ hợp lệ khi chứa nhãn binary `{0, 1}`.
- `y_score` là probability cho class `1 = bad/default`.
- `y_score` phải finite và nằm trong `[0, 1]`.
- Fold chỉ có một class làm ROC AUC, Partial Gini và các ranking metric tương tự bị `undefined`, không được trả về giá trị có vẻ hợp lệ.

### Chống leakage

- Không chọn threshold trên outer test fold.
- Không fit hoặc tối ưu bất kỳ tham số metric nào bằng outer test labels.
- Threshold hoặc business parameters phải có nguồn rõ: fixed từ config, chọn trong inner CV, hoặc lấy từ reference đã xác minh.
- P4 validation artifacts phải ghi nguồn tham số trong `parameters`.

### Edge cases bắt buộc

Mọi metric phải xử lý rõ:

- input rỗng;
- độ dài `y_true` và `y_score` không khớp;
- NaN hoặc infinity;
- `y_score` ngoài `[0, 1]`;
- `y_true` không binary;
- fold chỉ có một class;
- prediction score ties;
- metric không xác định;
- business parameters thiếu hoặc không hợp lệ.

Khi metric không xác định, `MetricResult.status` phải là `undefined`, `unsupported` hoặc `failed`; không được lặng lẽ trả về `0.0`.

## 4. Specification từng metric

### ROC AUC

| Trường | Nội dung |
|---|---|
| Tên chuẩn | ROC AUC |
| Metric ID | `roc_auc` |
| Mục đích | Đo khả năng phân biệt default và non-default trên toàn score distribution |
| Input | `y_true`, `y_score` |
| Output | scalar float |
| Công thức/nguồn | Area under ROC curve; paper mô tả là xác suất defaulter nhận score cao hơn non-defaulter |
| Direction | `maximize` |
| Miền kỳ vọng | `[0, 1]` |
| Edge cases | Undefined nếu fold chỉ có một class; ties theo implementation reference được chọn trong P4B |
| Parameters | `labels: [0, 1]` |
| Exactness | `exact` nếu dùng reference implementation đã test |
| Scientific reporting | Có, sau P4B validation |

### Brier Score

| Trường | Nội dung |
|---|---|
| Tên chuẩn | Brier Score |
| Metric ID | `brier_score` |
| Mục đích | Đo sai số bình phương trung bình giữa probability dự báo và binary response |
| Input | `y_true`, `y_score` |
| Output | scalar float |
| Công thức/nguồn | Mean squared error giữa `p(+1\|x)` và nhãn binary, như paper mô tả |
| Direction | `minimize` |
| Miền kỳ vọng | `[0, 1]` với binary probability |
| Edge cases | Undefined/failed nếu input rỗng, length mismatch, NaN/Infinity hoặc score ngoài `[0, 1]` |
| Parameters | `positive_label: 1` |
| Exactness | `exact` nếu dùng reference implementation đã test |
| Scientific reporting | Có, sau P4B validation |

### Partial Gini với `b = 0.4`

| Trường | Nội dung |
|---|---|
| Tên chuẩn | Partial Gini at probability cutoff 0.4 |
| Metric ID | `partial_gini_b_0_4` |
| Mục đích | Đo khả năng phân biệt trong vùng score thấp hơn ngưỡng chấp nhận của credit scoring |
| Input | `y_true`, `y_score` |
| Output | scalar float hoặc undefined |
| Công thức/nguồn | Paper nói tập trung phần score distribution `p(+1\|x) <= b`, chọn `b = 0.4` theo Lessmann et al. (2015) |
| Direction | `maximize` |
| Miền kỳ vọng | Chưa chốt exact cho production; P4B phải ghi rõ normalization trước khi dùng scientific reporting |
| Edge cases | Undefined nếu không còn đủ hai class trong vùng `y_score <= 0.4`; ties phải deterministic và documented |
| Parameters | `b: 0.4`, `positive_label: 1`, `score_region: y_score <= b` |
| Exactness | `approximate` cho đến khi P4B reference validation chốt normalization/tie policy |
| Scientific reporting | Chưa, chỉ sau P4B |

P4A không triển khai công thức Partial Gini production. Quyết định chi tiết nằm ở `docs/decisions/P4A_PARTIAL_GINI_AND_EMP.md`.

### Expected Maximum Profit - EMP

| Trường | Nội dung |
|---|---|
| Tên chuẩn | Expected Maximum Profit |
| Metric ID | `emp` |
| Mục đích | Ước lượng profit business tối đa kỳ vọng khi áp dụng classifier |
| Input | `y_true`, `y_score`, business parameters/distribution |
| Output | scalar profit hoặc unsupported |
| Công thức/nguồn | Paper dẫn Verbraken et al. (2014) và trình bày công thức MP/EMP tổng quát |
| Direction | `maximize` |
| Miền kỳ vọng | Phụ thuộc business parameters; không ép về `[0, 1]` nếu là profit |
| Edge cases | `unsupported` nếu thiếu business parameters/distribution; `undefined` nếu threshold set không hợp lệ |
| Parameters | `b1`, `c0`, `c_star`, distribution hoặc reference parameter source |
| Exactness | `not_applicable` trong P4A vì chưa implemented; decision status là `not_implemented_due_to_insufficient_specification` |
| Scientific reporting | Chưa, chỉ sau P4C và phải label exact/approximate rõ |

## 5. Artifact contract

Metric-validation artifacts sau P4A nên dùng object:

```json
{
  "metric_id": "roc_auc",
  "metric_version": "1.0",
  "value": 0.75,
  "direction": "maximize",
  "status": "valid",
  "parameters": {"labels": [0, 1]},
  "exactness": "exact",
  "warnings": []
}
```

Enums:

- `direction`: `maximize`, `minimize`
- `status`: `valid`, `undefined`, `unsupported`, `failed`
- `exactness`: `exact`, `approximate`, `not_applicable`

Trường `value` phải finite khi `status: valid`; với `undefined`, `unsupported` hoặc `failed`, `value` có thể là `null` và phải có `warnings` giải thích.

Smoke runner P2C vẫn giữ `metrics.json.test_metrics` dạng flat dictionary để backward compatibility. P4 artifacts có thể thêm danh sách `metric_results` mà không xóa flat smoke keys.

## 6. Config contract

Config Phase 4 nên khai báo:

```yaml
evaluation:
  metrics:
    - id: roc_auc
    - id: brier_score
    - id: partial_gini_b_0_4
      parameters:
        b: 0.4
    - id: emp
      parameters:
        parameter_source: reference_or_config
```

Rules:

- Metric IDs phải thuộc registry được validate.
- Partial Gini phải khai báo `b = 0.4` nếu dùng paper-aligned Protocol A.
- EMP phải khai báo nguồn business parameters; không tự bịa tham số.
- Threshold hoặc business parameters không được chọn bằng outer test labels.
- Artifact phải lưu `metric_version`, `parameters`, `exactness`, `status` và `warnings`.

## 7. Publishability

P4A không tạo kết quả publishable. Metric-validation artifacts sau này phải có:

```yaml
publishable: false
result_scope: metric_validation
```

Validation metrics không được đưa lên website như kết quả nghiên cứu. Website chỉ được công bố kết quả sau khi artifact đã aggregate, validate, sanitize và scope publish được duyệt.
