# P4B - Xác minh ROC AUC, Brier Score và Partial Gini

## 1. Scope P4B

P4B triển khai production implementation và reference validation cho ba metric của paper:

- ROC AUC
- Brier Score
- Partial Gini với tham số mặc định `b = 0.4`

P4B không triển khai EMP, không tạo scientific results và không tạo metric-validation artifacts publishable. Phase 4 sau P4B vẫn ở trạng thái `in_progress`; P4C còn phần EMP, config/integration và artifact harness.

## 2. Tóm tắt implementation

- Production metric API nằm trong `src/creditrep/metrics/`.
- `compute_roc_auc(...)` dùng định nghĩa rank-based tương đương xác suất một defaulter có score cao hơn một non-defaulter.
- `compute_brier_score(...)` dùng đúng công thức binary probability `mean((y_score - y_true)^2)`.
- `compute_partial_gini(...)` lấy subset thỏa `y_score <= b`, tính ROC AUC trên subset đó và chuẩn hóa bằng `2 * AUC_subset - 1`.
- Smoke runner P2C vẫn giữ artifact flat `test_metrics`; AUC/Brier được tái sử dụng qua adapter nên không đổi key hay schema cũ.

## 3. Quyết định khoa học

### ROC AUC

- Positive class cố định là `1 = bad/default`.
- `y_score` là probability cho class `1`.
- Ties xử lý theo convention chuẩn của ROC AUC: nửa điểm cho cặp tie.
- Metric là `exact` theo specification đã chốt.

### Brier Score

- Dùng binary Brier Score không scale, không nhân hệ số.
- Miền giá trị kỳ vọng là `[0, 1]`.
- Metric là `exact`.

### Partial Gini

- Nguồn bằng chứng mạnh nhất là paper Gunnarsson et al. (2021) và Lessmann et al. (2015) trong phần mô tả performance indicators.
- P4B chốt rằng `b = 0.4` là cutoff trên xác suất default dự đoán, tức chỉ giữ các quan sát thỏa `y_score <= b`.
- Trên tập con đó, metric được định nghĩa là Gini chuẩn hóa từ ROC AUC:

```text
PartialGini(b) = 2 * AUC({i : y_score_i <= b}) - 1
```

- Không dùng normalization khác ngoài biến đổi chuẩn `Gini = 2*AUC - 1`.
- Sign convention là maximize: tốt hơn thì lớn hơn.
- Miền giá trị là `[-1, 1]`.
- Random model hoặc constant scores cho kỳ vọng `0`.
- Perfect ranking trong vùng đánh giá cho giá trị `1`; perfect reversed ranking cho giá trị `-1`.
- Ties dùng cùng convention với ROC AUC nên deterministic và bất biến theo permutation của input.
- Nếu vùng `y_score <= b` không có quan sát hoặc chỉ còn một class, metric là `undefined`.
- Theo specification đã chốt, implementation được gắn `exact` đối với chính specification này; đây không phải tuyên bố exact replication của mọi cách hiểu ngoài literature, mà là exact implementation của quyết định P4B dựa trên bằng chứng hiện có.

## 4. API và contract

Mỗi metric trả `MetricResult` với tối thiểu:

- `metric_id`
- `metric_version`
- `value`
- `direction`
- `status`
- `parameters`
- `exactness`
- `warnings`

Chính sách xử lý lỗi:

- Input invalid theo contract trả `status: failed`.
- Metric không xác định về mặt toán học trả `status: undefined`.
- Không silently clip xác suất ngoài `[0, 1]`.
- Không trả giá trị giả hợp lệ khi metric undefined.

## 5. Validation matrix

- `tests/test_p4a_metric_contract.py`: giữ backward compatibility của contract và smoke metric shape.
- `tests/test_p4b_validated_metrics.py`: reference validation cho ROC AUC, Brier Score và Partial Gini.
- Partial Gini có reference implementation độc lập trong test, dùng pairwise definition để tránh lặp lại nguyên production algorithm.

## 6. Known limitations

- P4B chưa tích hợp typed `MetricResult` vào artifact fold-level của Phase 7/P4C.
- EMP vẫn chưa được triển khai.
- Decision hiện tại cho Partial Gini dựa trên wording của paper và Lessmann et al. (2015); nếu P4C hoặc nhiệm vụ sau tìm được primary source mạnh hơn mâu thuẫn với định nghĩa này, cần mở decision update thay vì sửa im lặng.

## 7. P4C còn lại

- Chốt policy exact/approximate/unsupported cho EMP.
- Mở rộng config validation cho metric selection/business parameters.
- Tích hợp typed metric results vào metric-validation artifacts.
- Nối metric-validation harness với nested-CV/fold-level evaluation mà không tạo publishable artifacts.
