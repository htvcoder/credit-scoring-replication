# P7C.2.3 — Phân tích feasibility pilot RF/XGBoost và quyết định final protocol

## Tóm tắt điều hành

`run-001` là bằng chứng kỹ thuật, không phải kết quả dự báo hay kết quả khoa học có thể công bố. Phân tích chỉ-đọc của toàn bộ 60 `result.json`, cùng với `validate-plan` và `validate-artifacts`, xác nhận artifact hoàn chỉnh và nhất quán. Pilot chạy tuần tự trên CPU, với một thread cho mỗi estimator; không có outer refit, final nested-CV, selection hay ranking theo AUC/Brier/metric dự báo nào.

Pilot cho thấy cả hai implementation đều chạy ổn định trên AC và GMC tại ba mức độ phức tạp được định trước. Tuy nhiên, tài liệu khóa trước pilot không quy định ngưỡng pass/fail định lượng cho runtime, RSS, hay quy tắc ánh xạ từ ba candidate sang full grid; `docs/P7C_FINAL_PROTOCOL_PLAN.md` còn nêu rõ RF/XGBoost cần CPU/backend feasibility **và approval** trước khi final grid/budget được khóa. Vì vậy, bằng chứng đủ để lập ngân sách tham khảo có kiểm soát, nhưng không đủ để tự động khóa scientific protocol mà không tạo quy tắc hậu nghiệm.

Kết luận sau approval `user_task_instruction`:

| Model | Kết luận P7C.2.3 | Final protocol |
| --- | --- | --- |
| Random Forest | `FULL_REFERENCE_GRID_APPROVED` | `LOCKED` (30 candidates) |
| XGBoost | `FULL_REFERENCE_GRID_APPROVED` | `LOCKED` (108 candidates) |

Trước approval, blocker là governance/scientific-decision criterion bị thiếu, không phải hỏng artifact hay failure kỹ thuật. Approval của task P7C.2.3 chọn fidelity với full reference grids, không reduced grid và không additional pilot. Decision record chính thức là `docs/P7C2_RF_XGBOOST_DECISION.md`; full scientific execution vẫn chưa được authorize.

## Provenance và phạm vi

- Branch lúc phân tích: `feature/p7c2-rf-xgboost-feasibility`.
- HEAD chạy pilot và có trong mọi fit: `4f1100d03a7318f0ed4c18a54dfb7665c49308ec`; working tree khi chạy: `clean`.
- Immutable plan: `configs/protocols/p7c/p7c2_rf_xgboost_pilot_plan.yaml`.
- Plan digest trong plan, `execution_plan.json`, `environment.json`, `engineering_summary.json` và 60 artifact: `1f3a6cd5b9f4d766fe89b34676ba66cf3ec731b49b27ff3769af042d83f08516`.
- Artifact: `artifacts/p7c2-rf-xgboost-feasibility/run-001`; outer partition cố định `repeat_00/fold_00`; 5 inner folds; AC và GMC; 3 candidate/model.
- Mục tiêu khoa học của P7C.2 là quyết định feasibility/protocol cho RF và XGBoost của partial replication sáu dataset P7A. Pilot không cung cấp bằng chứng predictive, không dùng để chọn candidate, và không được diễn giải là scientific result.

P7A là nguồn tham chiếu Table 2: RF có `5 × 6 = 30` candidate và XGBoost có `3 × 3 × 2 × 2 × 3 = 108` candidate. Pilot chỉ phủ RF `rf_low/rf_medium/rf_high` và XGBoost `xgb_low/xgb_medium/xgb_high`, không phải full search space.

## Kiểm chứng artifact

`validate-plan` trả về `valid=true`, `expected_fits=60`, `unique_fit_ids=60`. `validate-artifacts` trả về `valid=true`, `completion_status=completed`, `resumable=false`.

| Kiểm tra | Kết quả |
| --- | ---: |
| Expected / completed / failed | 60 / 60 / 0 |
| Missing / unexpected / duplicate | 0 / 0 / 0 |
| Temporary / corrupt artifact | 0 / 0 |
| Stable identities | đúng 60, mỗi model×dataset×candidate có folds `0,1,2,3,4` |
| Ngoài plan (model/dataset/candidate/outer partition) | không có |
| configured/effective thread | `1/1` cho mọi fit |
| GPU | không dùng |
| XGBoost tree method | `hist` cho mọi XGBoost fit |
| Telemetry bắt buộc | có đủ mọi artifact |

Analyzer `scripts/analyze_p7c2_feasibility.py` đọc trực tiếp từng `fits/*/result.json`; không import estimator/preprocessing/runner và không gọi đường fit.

## Phương pháp tổng hợp telemetry

Mỗi hàng dưới đây gồm đủ năm inner folds `0–4`. `p90` dùng nearest-rank percentile (`ceil(0.90 × n)`, với `n=5`, tức giá trị lớn nhất). `CV` là độ lệch chuẩn population chia cho mean. Thời gian là wall-clock giây; CPU/wall là mean CPU time / mean wall-clock. RSS báo cáo theo MiB (`bytes / 2^20`), ưu tiên `rss_peak`; không cộng RSS giữa các fit tuần tự.

## Kết quả Random Forest

| Dataset | Candidate | Folds | wall: min / median / mean / p90 / max (s) | Tổng wall (s) | CV | CPU/wall | peak RSS: median / max (MiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| AC | `rf_low` | 0–4 | 0.428 / 0.464 / 0.490 / 0.592 / 0.592 | 2.452 | 0.120 | 0.943 | 155.18 / 155.47 |
| AC | `rf_medium` | 0–4 | 0.875 / 1.046 / 1.020 / 1.218 / 1.218 | 5.099 | 0.121 | 0.950 | 155.89 / 155.89 |
| AC | `rf_high` | 0–4 | 1.745 / 1.899 / 1.934 / 2.272 / 2.272 | 9.672 | 0.100 | 0.977 | 157.14 / 157.15 |
| GMC | `rf_low` | 0–4 | 1.592 / 1.792 / 1.807 / 2.118 / 2.118 | 9.034 | 0.104 | 0.991 | 211.51 / 211.68 |
| GMC | `rf_medium` | 0–4 | 15.207 / 15.974 / 15.864 / 16.361 / 16.361 | 79.319 | 0.024 | 0.981 | 211.57 / 212.22 |
| GMC | `rf_high` | 0–4 | 70.330 / 73.116 / 73.669 / 76.298 / 76.298 | 368.345 | 0.031 | 0.975 | 211.83 / 214.48 |

RF high/low median-runtime ratio là 4.10 trên AC và 40.79 trên GMC; GMC/AC ratio theo low/medium/high lần lượt là 3.86, 15.28 và 38.49. Đây là bằng chứng rõ ràng rằng không nên dùng một hệ số tuyến tính duy nhất cho toàn grid hoặc mọi dataset.

## Kết quả XGBoost

| Dataset | Candidate | Folds | wall: min / median / mean / p90 / max (s) | Tổng wall (s) | CV | CPU/wall | peak RSS: median / max (MiB) |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| AC | `xgb_low` | 0–4 | 0.094 / 0.100 / 0.106 / 0.119 / 0.119 | 0.528 | 0.101 | 1.006 | 158.69 / 158.73 |
| AC | `xgb_medium` | 0–4 | 0.099 / 0.108 / 0.106 / 0.110 / 0.110 | 0.530 | 0.036 | 0.973 | 158.75 / 158.76 |
| AC | `xgb_high` | 0–4 | 0.108 / 0.112 / 0.111 / 0.113 / 0.113 | 0.555 | 0.017 | 1.013 | 158.78 / 158.79 |
| GMC | `xgb_low` | 0–4 | 0.764 / 0.776 / 0.778 / 0.800 / 0.800 | 3.891 | 0.016 | 1.008 | 211.90 / 211.94 |
| GMC | `xgb_medium` | 0–4 | 1.021 / 1.028 / 1.046 / 1.097 / 1.097 | 5.229 | 0.028 | 0.986 | 214.50 / 214.74 |
| GMC | `xgb_high` | 0–4 | 0.849 / 0.897 / 0.890 / 0.922 / 0.922 | 4.452 | 0.030 | 0.955 | 214.48 / 214.68 |

XGBoost high/low median-runtime ratio là 1.12 trên AC và 1.16 trên GMC; medium/low là 1.08 và 1.32. GMC/AC ratio theo low/medium/high lần lượt là 7.75, 9.50 và 8.01. Candidate behavior chỉ là engineering cost behavior, không phải ranking dự báo.

## Outlier, chất lượng dữ liệu và giới hạn đo

Tổng wall-clock fit là 489.106 s; timestamp đầu/cuối là `2026-08-05T13:17:12.824848Z` / `2026-08-05T13:25:23.391985Z`, nên elapsed quan sát là 490.567 s. Chênh lệch 1.475 s là overhead giữa fits; không có overlap (minimum gap 0.000 s, maximum 1.319 s), phù hợp sequential policy. Toàn tập có median 0.908 s, p90 16.054 s và max 76.298 s; các RF GMC medium/high là outlier có nguyên nhân cấu hình và dataset-load có thể quan sát được, không phải failure.

Không có RSS âm, thiếu hay bằng 0 bất thường. Peak lớn nhất là 225,173,504 bytes (214.74 MiB, XGBoost GMC medium); RF peak lớn nhất là 224,894,976 bytes (214.48 MiB). `rss_delta` không được coi là tổng memory estimator: sampler là process-local, không bao phủ child process, lấy mẫu 0.05 s có thể bỏ lỡ peak ngắn, và Windows/Linux, load hệ thống, filesystem và thermal state có thể làm RSS/runtime thay đổi. Vì fit là tuần tự, tổng các `rss_peak` không phải RAM requirement.

CPU/wall xấp xỉ 0.94–1.01; mức hơi trên 1 ở các XGBoost AC là sai số clock/sampling nhỏ có thể quan sát được, không phải bằng chứng parallelism. Không có telemetry nào cho thấy mutation thread policy.

## Công thức và ước lượng final compute

P7A có 90 outer partitions: AC/GC/TH02 mỗi dataset 20, HMEQ/TC/GMC mỗi dataset 10. Với 5 inner folds:

| Model | Inner search | Outer refit | Tổng estimator fit |
| --- | ---: | ---: | ---: |
| RF | `30 × 5 × 90 = 13,500` | `90` | `13,590` |
| XGBoost | `108 × 5 × 90 = 48,600` | `90` | `48,690` |
| Tổng RF/XGBoost | 62,100 | 180 | 62,280 |

Theo dataset, RF có 3,000 inner fits cho mỗi AC/GC/TH02 và 1,500 cho mỗi HMEQ/TC/GMC; XGBoost tương ứng 10,800 và 5,400. Outer refit được tách riêng vì candidate được chọn chỉ sau inner search; pilot không thực hiện outer refit.

Ước lượng dưới đây là worksheet **không phải cam kết runtime**. Để tránh một hệ số chung, nó giữ model/dataset/candidate riêng: AC được dùng làm proxy cho 60 outer partitions của AC/GC/TH02, GMC cho 30 partitions của HMEQ/TC/GMC; RF phân 6/18/6 reference candidates theo `n_estimators` 100 / 250–750 / 1000 vào low/medium/high; XGBoost phân 36/36/36 theo `n_estimators` 50/100/150. Đây chỉ là ánh xạ cost rõ ràng, không khẳng định AC/GMC đại diện hoàn hảo cho dataset khác.

| Model | Central (median group) | Conservative (p90 group) | Diễn giải |
| --- | ---: | ---: | --- |
| RF inner + refit | 34.1 giờ | 35.8 giờ | refit được budget bằng candidate high cùng proxy; chưa tính system buffer |
| XGBoost inner + refit | 5.0 giờ | 5.3 giờ | CPU `hist`, không suy ra GPU speed-up |
| Cả hai | 39.1 giờ | 41.1 giờ | nếu đặt 5% scheduling/system buffer: khoảng 41.1–43.2 giờ wall-clock tuần tự |

RAM planning bảo thủ cho chạy tuần tự là peak process-local đã thấy khoảng 215 MiB, cộng headroom của OS/preprocessing; không có suy luận rằng 62,280 fit cần cộng dồn RAM. Ước lượng runtime có bất định đáng kể do dataset chưa pilot, tree-growth/cache, thay đổi load laptop/OS và cách ánh xạ candidate; vì thế không nên coi nó là basis duy nhất để lock budget.

## Đối chiếu criteria đã khóa và decision matrix

| Criterion có trước pilot | RF | XGBoost | Bằng chứng |
| --- | --- | --- | --- |
| Plan/digest bất biến; đủ 60 fit; không failure/missing/unexpected/corrupt | PASS | PASS | validator PASS, 60/60 |
| CPU-only, sequential, 1 estimator thread | PASS | PASS | `1/1`, không GPU |
| XGBoost `tree_method=hist` | N/A | PASS | mọi XGBoost artifact |
| Không predictive ranking/selection/outer refit | PASS | PASS | payload không chứa prediction/metric; pilot scope |
| Ngưỡng runtime/RAM/failure định lượng để lock final grid | **Không được định nghĩa** | **Không được định nghĩa** | plan/docs chỉ yêu cầu review feasibility |
| Quy tắc extrapolation/candidate mapping để lock budget | **Không được định nghĩa** | **Không được định nghĩa** | docs cấm blind extrapolation |
| Approval final grid/budget | Pending | Pending | DR-P7C-01/02 vẫn Open |

Trước approval, ba criteria cuối khiến `DECISION_BLOCKED` là kết luận trung thực duy nhất không biến telemetry sau hoc thành threshold. Approval của task P7C.2.3 hiện đã chọn `LOCK_FULL_REFERENCE_SPACE`; `LOCK_REDUCED_SPACE` không được chọn và `ADDITIONAL_PILOT_REQUIRED` không cần thiết.

## Đề xuất protocol RF và XGBoost

RF: `FULL_REFERENCE_GRID_APPROVED`, final protocol `LOCKED`. Không gian chính xác là `n_estimators ∈ {100,250,500,750,1000}` × `max_features_multiplier_of_sqrt_m ∈ {0.1,0.25,0.5,1,2,4}` (30). Không dùng evidence predictive hoặc rule hậu nghiệm để loại vùng grid.

XGBoost: `FULL_REFERENCE_GRID_APPROVED`, final protocol `LOCKED`. Không gian chính xác là `n_estimators ∈ {50,100,150}` × `max_depth ∈ {1,2,3}` × `learning_rate ∈ {0.3,0.4}` × `colsample_bytree ∈ {0.6,0.8}` × `subsample ∈ {0.5,0.75,1.0}` (108). Pilot không dùng để chọn candidate nhanh hơn.

Các phương án không chọn: reduced lock (không có rationale khoa học độc lập để giảm grid); additional pilot (không có failure và approval chọn không chạy thêm); GPU/cloud (ngoài scope của pilot và không được suy ra từ telemetry CPU).

## Scientific limitations và nội dung cần phê duyệt

Pilot chỉ có một outer partition, hai trong sáu dataset, năm inner folds và telemetry Windows process-local. Nó không xác minh predictive quality, chọn model/candidate, biến thiên giữa outer partitions, hay final nested-CV. Runtime/RSS không phải publication result.

Approval `user_task_instruction` đã chọn full reference spaces RF 30 và XGBoost 108, đồng thời chấp nhận worksheet khoảng 41.1–43.2 giờ sequential CPU có buffer cho planning. Đây không phải runtime guarantee, scheduling authorization hay full scientific execution authorization. Không chạy additional pilot và không chọn reduced space.

## Trạng thái

- P7C.2.2 execution: `COMPLETED`.
- P7C.2.2 artifact validation: `PASS`.
- P7C.2.3 evidence analysis: `COMPLETED`.
- RF final protocol: `LOCKED` (full 30-candidate P7A/Table-2 reference grid).
- XGBoost final protocol: `LOCKED` (full 108-candidate P7A/Table-2 reference grid).
- Compute worksheet: accepted for P7C planning only; scheduling/concurrency/retry policy remains P7C.7 work.
- Full scientific execution: `NOT STARTED` and not authorized by this decision.
- P7C.2.3 closeout: `COMPLETED`.
