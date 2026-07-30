# P7A — Giao thức thực nghiệm và candidate manifest

P7A khóa các trường khoa học bất biến cho tái lập trên sáu dataset công khai: AC, GC, HMEQ, TH02, TC và GMC. Đây là tái lập một phần vì bốn dataset độc quyền của bài báo không có trong repository.

Manifest machine-readable là `configs/protocols/p7a/p7a_candidate_manifest.yaml`. Hash dùng JSON canonical, SHA-256 và bao phủ toàn bộ manifest trừ chính block `lock`. Runtime chỉ được override `artifact_root`, `device`, `threads`, `checkpoint_root`, hoặc `log_level`; không được override dataset, fold, seed, preprocessing, search space hay metric.

## Giao thức khóa

- Target: class `0` là good/non-default; class `1` là bad/default; `y_score=P(class 1)`.
- Outer CV: AC, GC, TH02 dùng 10×2-fold; HMEQ, TC, GMC dùng 5×2-fold. Inner CV stratified 5-fold. Seed 42 là quyết định tái lập của dự án vì paper không công bố seed.
- Preprocessing dùng Protocol A, fit duy nhất trên training partition tương ứng.
- Primary metric là ROC AUC; secondary gồm Brier, log loss, accuracy, precision, recall, F1 và confusion matrix. Threshold 0.5 chỉ dùng cho metric cần nhãn.
- Retry chỉ dành cho lỗi hạ tầng tạm thời, tối đa một lần; resume yêu cầu protocol hash khớp; artifact invalid bị reject hoặc quarantine.

## Search space và trạng thái

Table 2 của paper là `reference_search_space`: LR không tuning; RF có 30, XGBoost 108 cấu hình. MLP công bố 144/720/2016 cấu hình cho 1/3/5 layer và cấm số neuron tăng theo layer. Bảng đồng thời liệt kê batch normalization nhưng số đếm chỉ khớp khi không nhân theo tham số này; manifest ghi rõ bất nhất đó.

`candidate_search_space` CART-A đã được phê duyệt riêng cho P7B: `max_depth` là `[3, 5, 7, 9]` và `min_samples_leaf` dạng tỷ lệ là `[0.005, 0.01, 0.02]`, tổng `4×3=12` candidate. Tỷ lệ được sklearn quy đổi theo `ceil(fraction × inner_training_row_count)` và được ghi canonical là `0.005`, `0.01`, `0.02`. Đây không phải ánh xạ confidence-based pruning của C4.5, không phải final scientific search space P7C, và vẫn mang deviation `c45_to_cart`.

Pilot P7B chỉ dùng AC, HMEQ, GMC; mỗi dataset dùng `repeat_00_fold_00` được dẫn xuất từ seed 42, bốn candidate đã khóa theo coverage và 5 inner folds, tổng 60 inner fits. Pilot không thực hiện scientific model selection/final refit; metric nếu có chỉ là non-publishable pipeline validation. Full theoretical CART Grid 2 có 5.400 inner candidate-evaluation fits và, nếu P7C sau này được khóa/chạy, 90 outer selected-model refits tách biệt.

CART-B (`ccp_alpha`) vẫn là phương án dự phòng; CART-C không được chọn. P7B closeout phải ghi decision record và chỉ khi đó mới được khóa final scientific search space P7C.

RF, XGBoost, MLP-1 và MLP-3 giữ reference grid làm candidate scientific search space nhưng final budget chỉ khóa sau P7B. MLP-5 là `decision_pending`; CatBoost và TabNet là `conditionally_required`; FT-Transformer là `optional_extension` và không tính vào main completeness matrix.

## P7B và P7C

P7B chỉ đo feasibility: thời gian, RAM/VRAM, dung lượng artifact, failure/resume và chi phí dự kiến. Ngưỡng định lượng còn `pending_user_approval`. Mọi pilot artifact và predictive metric là non-publishable, không được dùng để quyết định model inclusion dựa trên performance.

P7B closeout phải tạo decision records, khóa `final_scientific_search_space` và sinh final locked P7C manifest trước khi P7C được phép bắt đầu.

```powershell
.\.venv\Scripts\python.exe -m creditrep.protocols.cli verify configs/protocols/p7a/p7a_candidate_manifest.yaml
.\.venv\Scripts\python.exe -m creditrep.protocols.cli render configs/protocols/p7a/p7a_candidate_manifest.yaml
```
