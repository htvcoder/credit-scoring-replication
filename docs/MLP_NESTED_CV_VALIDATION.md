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

## P6C.2A — GC reduced engineering validation

Profile `configs/p6c_gc_reduced_validation_v1.yaml` chạy duy nhất để kiểm chứng
engineering, không phải kết quả khoa học hay bảng xếp hạng mô hình. Profile dùng GC,
2 outer folds, 2 inner folds, 1 candidate mỗi MLP, CPU, tối đa 4 epochs và cùng
`fair_budget_id: p6b_shared_v1`; chỉ hidden depth khác nhau. Nó giữ
`publishable: false`, `result_scope: model_validation` ở artifact P5C và
`validation_purpose: engineering_reduced_validation`.

Preflight ngày 2026-07-29: GC có checksum
`B21F3D81DB8071257D5FF1DEAEBA1FD4303B62712E6FCC9715C7A86202CB5871`, 1.000 dòng,
20 features, class 0/1 là 700/300, không missing. Lệnh chạy là
`python scripts/run_model_validation.py --config configs/p6c_gc_reduced_validation_v1.yaml --resume`.
Experiment `p6c_gc_reduced_validation_v1-fb1d5844d728` hoàn tất 6/6 fold, không failure
hay retry; artifact local trong `results/` (ignored), 105 files, 301.853 bytes.

Quan sát runtime final-refit theo fold (giây) là: MLP-1 0,079 và 0,143; MLP-3 0,115
và 0,298; MLP-5 0,300 và 0,357. Parameter count tương ứng là 1.409, 9.729 và 18.049.
Đây chỉ là evidence vận hành; không suy diễn chất lượng tương đối. Peak memory là
**NOT AVAILABLE** vì project không có dependency portable đáng tin cậy để đo peak
process memory. Lần resume thứ hai skip 6 fold hợp lệ, không tạo duplicate evidence.

Ước lượng cho P6C.2B/TC là **estimated**, không phải observed: số fit và chi phí
preprocessing sẽ tăng theo số fold, candidate, rows và transformed features; disk cũng
tăng theo history artifacts. Khuyến nghị thực hiện TC reduced checkpoint trước, ghi
runtime/memory/disk quan sát, rồi mới quyết định CPU local hay hạ tầng khác. Không chạy
TC, không checkpoint, không resume giữa epoch và không công bố metric reduced.

## P6C.2B — TC reduced resource checkpoint

Profile `configs/p6c_tc_reduced_checkpoint_v1.yaml` là checkpoint engineering, dùng
TC trên CPU với 2 outer × 2 inner folds, một candidate mỗi MLP và hai epochs; cả ba
depth giữ cùng batch size, optimizer, learning rate, weight decay, early-stopping
policy, seed và `fair_budget_id: p6b_shared_v1`. Mục đích là
`engineering_resource_checkpoint`, không phải kết quả khoa học.

Preflight TC pass: checksum `30C6BE3ABD8DCFD3E6096C828BAD8C2F011238620F5369220BD60CFC82700933`,
30.000 dòng, 23 features sau bỏ ID, class 0/1 là 23.364/6.636 và không missing.
Experiment `p6c_tc_reduced_checkpoint_v1-fe5cf2f64473` hoàn tất 6/6 folds. Lần đầu
có một `PermissionError` khi atomic publication của một fold MLP-5; failure artifact
được phân loại retryable. Resume skip 5 folds valid, retry một fold, sau đó artifact
validator pass; lần resume tiếp theo skip đủ 6 fold.

Observed final-refit duration (giây/fold) là MLP-1: 0,754 và 1,005; MLP-3: 0,818 và
1,427; MLP-5: 1,906 và 1,606. Parameter counts lần lượt khoảng 1.345–1.409, 9.665
và 17.985; mỗi final refit có 12.000 train rows, 3.000 validation rows. Artifact local
có 106 files, 3.369.384 bytes. Peak memory vẫn **NOT AVAILABLE** theo policy không thêm
dependency đo memory. Những số này chỉ là observed engineering evidence, không dùng để
ranking model hay so sánh paper.

Quyết định tài nguyên: **local CPU feasible nhưng runtime dài** cho checkpoint reduced.
Đây là extrapolation thận trọng: TC có 30× rows GC và artifact lớn hơn khoảng 11×, trong
khi checkpoint chỉ dùng 2 epochs và 1 candidate. Full scientific run sẽ tăng theo epochs,
candidates và fold/repeat; rủi ro local gồm sleep/restart, thermal throttling, disk và
thời gian dài. Khuyến nghị trước P6C scientific work là thêm checkpoint/profiling theo
scope đã duyệt; cloud CPU là lựa chọn thực dụng nếu budget lớn hơn đáng kể. Evidence hiện
có chưa chứng minh GPU bắt buộc. Không có checkpoint weights, resume giữa epoch hay metric
scientific được công bố.

## Quyết định đóng P6C và Phase 6

P6C hoàn tất ở boundary **infrastructure/hardening**: MLP-1/3/5 đã qua nested-CV
leakage-safe, early stopping deterministic, artifact atomic, retry/resume/reconciliation
và reduced validation thật trên GC/TC. Các reduced artifacts vẫn non-publishable và không
được dùng để xếp hạng MLP hay so sánh paper. Theo implementation plan, core scientific
replication trên các dataset/model đã chốt, fold-level metrics và evidence RQ1/RQ2 thuộc
**Phase 7**. Vì vậy Phase 6 không phụ thuộc một full scientific TC run để đóng.

Phase 7 là bước tiếp theo. Tag đề xuất cho Phase 6 là
`p6-mlp-depth-replication-complete`; tag này chưa được tạo trong Git. Công việc Phase 7
còn lại gồm core replication approved scope, aggregation/scientific comparison ở các phase
sau và website chỉ dùng artifacts đã được phép công bố.
