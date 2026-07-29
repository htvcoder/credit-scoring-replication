# Xác thực nested-CV MLP

## P6C.1B-b

Ngay sau mỗi lần `fit` MLP, runner chụp deep-copy JSON-safe của training summary,
epoch history và metadata split early-stopping. Evidence inner candidate được giữ theo
candidate và inner fold; final outer refit được giữ riêng, liên kết với candidate đã chọn.

Artifacts nằm trong thư mục fold P5C hiện có, tại `neural/candidates/...` và
`neural/final_refit/...`, với `neural/neural_manifest.json` dùng đường dẫn tương đối.
Writer ghi tất cả vào thư mục tạm của fold, parse và xác thực từng payload, đối chiếu
summary/history, xác thực split và toàn bộ neural artifact set trước khi atomic rename.

Neural artifacts luôn `publishable: false` và
`result_scope: mlp_nested_cv_validation`. Chúng không chứa Tensor, model/state_dict,
optimizer, checkpoint, dữ liệu hàng thô hoặc đường dẫn tuyệt đối. Fold classical không
cần và không tạo neural artifacts.

P6C.1C còn lại: failure stages neural, phân loại retry, resume/retry chuyên biệt và
deterministic reconciliation mở rộng.
