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

## P6C.1C

P5C failure/resume được mở rộng tại cùng execution-unit boundary, không có neural
runner song song và không resume giữa epoch. Các stage neural là
`early_stopping_split`, `neural_model_initialization`, `neural_training`,
`neural_metadata_capture`, `neural_artifact_validation`,
`neural_artifact_publication` và `neural_reconciliation`.

Failure neural ghi identity SHA-256 từ experiment, dataset checksum, config fingerprint,
model, outer/inner fold, candidate và training scope. Attempt được tăng từ artifact
failure hợp lệ; retry giữ nguyên seed model/dataloader/split. Chính sách mặc định cho
phép một retry cho lỗi I/O/transient; lỗi split, schema, contract, non-finite và mismatch
không retry. Fold completed luôn được validate trước khi skip; fold corrupt được
quarantine và không sửa tại chỗ. Message lỗi được redaction secret/token/password,
raw-row và local path. Không có checkpoint, state_dict hay weights trong retry evidence.

P6C.2 còn lại: GC reduced validation, TC resource checkpoint, đánh giá runtime/tài
nguyên, hoàn thiện tài liệu và status synchronization cuối Phase 6.
