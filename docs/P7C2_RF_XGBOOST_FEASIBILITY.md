# P7C.2 — Pilot engineering-feasibility cho Random Forest/XGBoost

## Phạm vi và ranh giới khoa học

P7C.2 được chia thành ba checkpoint: P7C.2.1 chuẩn bị immutable plan, execution harness, CLI, artifact validator và cơ chế resume; P7C.2.2 mới chạy research-data pilot; P7C.2.3 lập decision record và quyết định protocol. P7C.2.1 không chạy dữ liệu nghiên cứu, không tạo metric khoa học, ranking, selection, outer refit hay final nested-CV result. Synthetic integration tests chỉ xác minh đường đi kỹ thuật của preprocessing và estimator, không phải feasibility pilot thật.

Reference grid P7A vẫn là nguồn tham chiếu, chưa phải final grid đã khóa: RF có `5 n_estimators × 6 max_features_multiplier_of_sqrt_m = 30` candidates; XGBoost có `3 n_estimators × 3 max_depth × 2 learning_rate × 2 colsample_bytree × 3 subsample = 108` candidates. P7C.2.1 không thêm đường chạy full-grid và không khóa RF/XGBoost final search space hoặc final compute budget.

## Immutable pilot plan

Plan tại `configs/protocols/p7c/p7c2_rf_xgboost_pilot_plan.yaml` có canonical digest:

`1f3a6cd5b9f4d766fe89b34676ba66cf3ec731b49b27ff3769af042d83f08516`

Ba RF candidates đều thuộc reference grid:

| Candidate | `n_estimators` | `max_features_multiplier_of_sqrt_m` |
| --- | ---: | ---: |
| `rf_low` | 100 | 0.1 |
| `rf_medium` | 500 | 1 |
| `rf_high` | 1000 | 4 |

Ba XGBoost candidates đều thuộc reference grid:

| Candidate | `n_estimators` | `max_depth` | `learning_rate` | `colsample_bytree` | `subsample` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `xgb_low` | 50 | 1 | 0.3 | 0.6 | 0.5 |
| `xgb_medium` | 100 | 2 | 0.4 | 0.8 | 0.75 |
| `xgb_high` | 150 | 3 | 0.4 | 0.8 | 1.0 |

AC và GMC được chọn để tạo hai mức tải đại diện nhỏ/lớn trong số dataset đã khóa, không phải vì predictive performance. Pilot dùng duy nhất outer partition xác định trước `repeat_00/fold_00` và năm inner folds. Tổng số fit là `2 datasets × 2 models × 3 candidates × 5 inner folds = 60`: 30 RF và 30 XGBoost. Outer partition và candidates không được chọn bằng metric.

Stable fit identity là SHA-256 canonical từ đúng các trường: plan digest, model ID, dataset ID, outer repeat/fold, candidate ID, inner-fold index và seed. Timestamp và absolute path không tham gia identity. Runner chỉ tạo các identity có trong immutable plan.

## Isolation, estimator và resource policy

Mỗi fit tạo preprocessing thật của repository, fit chỉ trên inner-training rows rồi dùng fitted preprocessor để transform inner-validation rows. Runner yêu cầu estimator class order đúng `[0, 1]` và chỉ xác minh xác suất hữu hạn ở `predict_proba[:, 1]`, tương ứng `bad/default = 1`; xác suất không được lưu vào artifact.

RF nhận `n_estimators`, seed và `max_features` được ánh xạ xác định từ multiplier, với `n_jobs=1`. XGBoost nhận nguyên candidate parameters và seed, dùng `n_jobs=1`, `tree_method=hist`, `eval_metric=logloss`. Toàn pilot chỉ chạy một fit tại một thời điểm. Plan validator từ chối `n_jobs=-1`, nested/concurrent fits, GPU hoặc threading-policy mutation; harness không có cloud execution path.

## Artifact, telemetry và atomic finalize

Runtime output phải là thư mục con của `artifacts/p7c2-rf-xgboost-feasibility/`, và root này được Git ignore. Mỗi fit có một JSON artifact chứa schema version, stable identity, plan digest, Git provenance, model/dataset/candidate, outer repeat/fold, inner fold, seed, configured/effective thread count, timestamps, wall-clock duration, process CPU time, outcome, sanitized error, PID, RSS start/peak/delta, library versions và XGBoost tree method khi áp dụng.

Writer ghi temporary sibling trong artifact root rồi finalize nguyên tử bằng `os.replace`. Completed artifact hợp lệ không bị overwrite; corrupt hoặc incomplete artifact không bị tự động xóa. Artifact cấm predictions, model weights, raw rows, transformed matrices, feature values, secret và absolute user-specific path.

RSS là process-local sampling theo chu kỳ 0.05 giây, không bao phủ toàn bộ child-process/system memory và có thể bỏ lỡ peak rất ngắn. Windows/Linux, system load, filesystem và thermal state có thể làm wall time/RSS khác nhau. Telemetry chỉ là engineering evidence, không phải predictive hoặc scientific result.

## Resume, retry và artifact validation

`resume` yêu cầu execution plan, digest, threading policy và Git provenance khớp. Nó bỏ qua completed artifact hợp lệ, chỉ xử lý missing fit hoặc failed fit còn trong retry budget. Policy cho phép tối đa một retry và chỉ retry `OSError`/`TimeoutError`; fit đã hết ngân sách không chạy lại. Corrupt, duplicate, temporary hoặc policy-mismatched artifact làm resume bị từ chối để bảo toàn evidence.

`validate-artifacts` không fit model. Validator đối chiếu đủ 60 identities và báo expected/completed/failed/missing/unexpected; đồng thời phát hiện duplicate identity, corrupt JSON/schema, incomplete temporary file, plan mismatch, provenance/telemetry thiếu, completed artifact không hợp lệ, forbidden payload và failure vượt retry contract.

## Gate cho P7C.2.2 và giới hạn ngoại suy

P7C.2.2 chỉ pass khi plan/digest được giữ nguyên, cả 60 planned fits có completed artifact hợp lệ, không còn failed/missing/unexpected/duplicate/corrupt artifact, artifact validator pass và không vi phạm single-thread/CPU-only policy. Interrupt có thể được tiếp tục bằng `resume`; completion của pilot không tự động khóa final protocols.

Runtime của 60 fits trên một outer partition không được ngoại suy tuyến tính thành cam kết thời gian cho full 90 outer partitions, full RF/XGBoost grids hoặc model khác. Chi phí preprocessing, dataset size, tree growth, cache, OS và tải máy đều có thể phi tuyến. P7C.2.3 phải review engineering evidence và lập decision record trước mọi thay đổi final grid/budget.

Sau closeout P7C.2.1: research pilot vẫn **Not Run**; RF final protocol/search space, XGBoost final protocol/search space và final compute budget vẫn **Not Locked**; full scientific execution vẫn **NOT READY**.
