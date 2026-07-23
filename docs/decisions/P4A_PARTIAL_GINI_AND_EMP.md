# P4A decision record: Partial Gini và EMP

## Trạng thái

Accepted for P4A specification. Chưa phải implementation production.

## Nguồn đã đối chiếu

- `paper/Gunnarsson_et_al_2021_Deep_Learning_for_Credit_Scoring.md`, mục 4.3.1 và 4.3.2.
- `docs/EXPERIMENT_FEASIBILITY_ASSESSMENT.md`, mục metric và feasibility.
- `docs/EXPERIMENT_IMPLEMENTATION_PLAN.md`, roadmap Phase 4.
- `docs/SMOKE_EXPERIMENT_RUNNER.md`, contract smoke metrics hiện tại.

Không dùng nguồn ngoài repository trong P4A vì các nguồn local đã đủ để xác định phần nào đã rõ và phần nào còn thiếu.

## Quy ước target và score

- `0 = good / non-default`.
- `1 = bad / default`.
- `y_score = p(+1|x)`, tức xác suất default/bad.
- Score càng cao nghĩa là risk/default càng cao.

## Partial Gini

### Paper nói gì

Paper mô tả AUC là xác suất một defaulter ngẫu nhiên nhận score cao hơn một non-defaulter ngẫu nhiên. Paper nói Partial Gini tập trung vào phần score distribution dưới ngưỡng `p(+1|x) <= b`, với `b = 0.4`, theo Lessmann et al. (2015).

### Paper chưa đặc tả đủ gì

Paper trong repository chưa nêu:

- công thức Partial Gini rời rạc chính xác;
- normalization có dùng hay không;
- cách xử lý ties;
- expected range sau normalization;
- reference toy values;
- có tích phân trực tiếp theo score threshold hay biến đổi qua ROC/acceptance curve.

### Quyết định P4A

P4A chốt metric ID:

```text
partial_gini_b_0_4
```

P4A chốt tham số bắt buộc:

```yaml
b: 0.4
score_region: y_score <= b
positive_label: 1
```

Direction là `maximize`.

P4A chưa tuyên bố exact replication cho Partial Gini. Trạng thái exactness trong contract là `approximate` cho đến khi P4B có reference validation. Đây là implementation decision/deviation có kiểm soát, không phải khẳng định công thức nguyên bản của paper.

### Tiêu chí cho P4B

P4B phải làm ít nhất:

- ghi rõ công thức rời rạc;
- ghi rõ normalization;
- ghi rõ sign convention;
- ghi rõ tie policy;
- tạo toy reference tests bằng tay;
- test fold chỉ có một class;
- test trường hợp vùng `y_score <= 0.4` không còn đủ hai class;
- ghi warnings/status khi metric undefined.

Nếu sau khi nghiên cứu thêm Lessmann et al. (2015) hoặc source đáng tin cậy khác cho thấy công thức khác với quyết định tạm thời, P4B phải cập nhật decision record và label deviation.

## EMP

### Paper nói gì

Paper dẫn Verbraken et al. (2014) và mô tả average classification profit:

```text
P(t; b1, c0, c*) =
(b1 - c*) * pi1 * F1(t)
- (c0 + c*) * pi0 * F0(t)
```

Paper định nghĩa `MP = max_t P(t; b1, c0, c*)` và EMP là kỳ vọng của profit theo joint density `h(b1, c0)`.

### Paper/repo chưa đủ gì

Repository hiện chưa có:

- giá trị hoặc phân phối của `b1`;
- giá trị hoặc phân phối của `c0`;
- giá trị `c*`;
- quy ước threshold grid chính thức;
- cách estimate `F1(t)` và `F0(t)` trong implementation;
- reference values;
- quyết định threshold chọn trong inner CV hay fixed config.

### Quyết định P4A

EMP decision status:

```text
not_implemented_due_to_insufficient_specification
```

Metric ID:

```text
emp
```

Direction là `maximize`.

P4A không tự tạo business parameters. Nếu P4C triển khai approximate EMP, artifact phải ghi:

- `exactness: approximate`;
- business parameter source;
- threshold selection source;
- warnings/deviation;
- ảnh hưởng đến RQ và final report.

### Không để EMP thành blocker vô thời hạn

EMP không được chặn toàn bộ project nếu paper/repo không cung cấp đủ business specification. P4C có thể chọn một trong hai hướng:

1. `not_implemented_due_to_insufficient_specification`: bỏ EMP khỏi scientific comparison chính, ghi limitation rõ.
2. `approximate`: triển khai theo reference/assumption được ghi trong config và decision record, không gọi là exact.

### Tiêu chí cho P4C

P4C chỉ được tính EMP khi:

- business parameters có nguồn rõ;
- threshold không được chọn bằng outer test labels;
- config validation reject missing/invalid parameters;
- artifact ghi `publishable: false` và `result_scope: metric_validation`;
- final report label exact/approximate/unsupported rõ ràng.

## Ảnh hưởng đến RQ và final report

- AUC/Brier có thể dùng cho scientific reporting sau P4B.
- Partial Gini chỉ dùng sau khi công thức và reference tests được chốt.
- EMP nếu không exact thì không được dùng để overclaim replication; phải ghi là approximate hoặc unsupported.
- Nếu EMP unsupported, RQ chính vẫn có thể trả lời bằng AUC/Brier/Partial Gini validated, nhưng limitation phải nêu rõ thiếu metric profit-based exact.
