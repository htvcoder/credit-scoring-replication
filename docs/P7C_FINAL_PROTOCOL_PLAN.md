# P7C.1 — Inventory và kế hoạch final scientific protocol

## Mục tiêu, phạm vi và scientific boundary

P7C.1 lập inventory, readiness matrix và decision register cho P7C; không chạy training, nested CV, candidate selection, ranking, outer refit hoặc tạo metric khoa học. Nguồn máy đọc được là `configs/protocols/p7c/p7c_protocol_inventory.yaml`. P7A giữ protocol sáu dataset, repeated stratified two-fold outer CV (90 outer partitions), five-fold inner CV và seed 42; P7B chỉ khóa CART-A qua decision record, không cung cấp kết quả khoa học.

## Readiness và budget worksheet

| Model | Vai trò | Search space hiện có | Final status | Count | Backend | Bằng chứng/blocker | Checkpoint |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| Logistic Regression | replication baseline | Paper: không tuning | Chưa có final contract | 1 | CPU | Cần ghi contract baseline | P7C.7 |
| CART | replication baseline | CART-A depth 3/5/7/9 × leaf .005/.01/.02 | Locked | 12 | CPU | P7B decision + final manifest | P7C.7 |
| Random Forest | replication baseline | Full P7A/Table-2 reference grid | Locked | 30 | CPU | P7C.2.2 validated; DR-P7C-01 approved | P7C.7 |
| XGBoost | replication baseline | Full P7A/Table-2 reference grid | Locked | 108 | CPU/GPU | P7C.2.2 validated; DR-P7C-02 approved | P7C.7 |
| MLP-1 | replication baseline | Paper reference | Unlocked | 144 | CPU/GPU | P7C.3 canonical `vm-run-003` accepted; final budget/backend còn chờ phê duyệt | P7C.7 |
| MLP-3 | replication baseline | Unlocked | 720 | CPU/GPU | P7C.3 canonical `vm-run-003` accepted; final budget/backend còn chờ phê duyệt | P7C.7 |
| MLP-5 | replication baseline | Paper reference | Decision pending | 2,016 | CPU/GPU | P7C.3 canonical `vm-run-003` accepted; không có predictive exclusion; scope/budget còn chờ phê duyệt | P7C.7 |
| CatBoost | main-results extension | Chưa có grid | Decision pending | TBD | CPU/GPU | Protocol A/B, grid, budget | P7C.4 |
| TabNet | main-results extension | Chưa có grid | Feasibility required | TBD | GPU khuyến nghị | GPU + protocol + grid | P7C.5 |
| FT-Transformer | optional extension | Chưa có grid | Feasibility required | TBD | GPU khuyến nghị | Scope approval + GPU + grid | P7C.6 |

`candidate_count` của RF/XGBoost/MLP là số tham chiếu từ Table 2, không phải final locked budget. Với model tuning, inner fits = candidate count × 5 × outer partitions; outer refits = outer partitions. CART đã có 5,400 inner fits và 90 refits. Các thời gian chỉ là số đo engineering CART hoặc ngoại suy đã gắn caveat; không có benchmark đủ để báo thời gian chính xác cho model khác.

## Checkpoint plan

| Checkpoint | Nội dung/output | Entry → exit/validation | Có feasibility training? |
| --- | --- | --- | --- |
| P7C.1 | Inventory, validator, readiness matrix, decision register | P7B closed → inventory validated, docs/status/build pass | Không |
| P7C.2 | RF/XGBoost decision records và final manifests | Completed: full grids/provenance locked and protocol tests pass | P7C.2.2 pilot đã completed/validated; không phải scientific execution |
| P7C.3 | MLP-1/3/5 feasibility closeout | **Completed — canonical feasibility pilot accepted:** `vm-run-003` 60/60 completed, validator PASS, CPU/memory/stability PASS; final scope/budget/backend chưa khóa | Engineering feasibility non-publishable; không dùng predictive metric để loại MLP-5 hoặc giảm grid |
| P7C.4 | Model decision and compute-planning track | **In progress:** P7C.4A completed, P7C.4B blocked awaiting human approval; CatBoost decision remains open | Không chạy benchmark; không có scientific result hay quyết định backend |
| P7C.4A | MLP search strategy và compute benchmark plan | **Completed — benchmark plan ready for human review:** decision study, ba proposed budget, threshold định lượng và benchmark matrix đã được kiểm tra | DR-P7C-03/04 và final manifest vẫn mở; không chạy benchmark |
| P7C.4B | Approval-to-execution handoff | **Blocked awaiting human approval:** chỉ bắt đầu sau DR-P7C-03/04, threshold và policy được phê duyệt | Không tạo runner hay benchmark trước approval |
| P7C.5 | TabNet feasibility/final decision | P7C.1 → GPU evidence + bounded protocol decision | Có thể, nhưng non-publishable pilot riêng |
| P7C.6 | FT-Transformer extension decision | P7C.1 → scope/GPU/budget decision | Có thể, nhưng non-publishable pilot riêng |
| P7C.7 | Unified final manifest and readiness gate | P7C.2–.6 decisions resolved/deferred explicitly → all references/hashes/tests pass | Không; đây là gate trước execution |

P7C.2 được triển khai theo ba sub-checkpoint: P7C.2.1 hoàn tất immutable 60-fit plan và execution harness; P7C.2.2 đã completed và artifact validation pass với 60/60 fits; P7C.2.3 đã hoàn tất analysis/decision record. Theo approval `user_task_instruction`, DR-P7C-01/02 khóa full P7A/Table-2 grids cho RF/XGBoost trong `configs/protocols/p7c/p7c_rf_xgboost_final_manifest.yaml`. Việc hoàn tất P7C.2 không làm full scientific execution ready.

P7C.3 đã closeout theo Branch B: `vm-run-003` là canonical accepted evidence; `vm-run-001` historical/non-canonical và `vm-run-002` historical invalid không được dùng cho projection hay kết luận khoa học. Kế hoạch feasibility bất biến không phải final search-space/budget manifest; evidence xác nhận CPU/memory/stability, không cho phép ranking, loại MLP-5 hoặc giảm grid hậu nghiệm. GPU không bắt buộc cho correctness/feasibility, nhưng backend/budget tối ưu thời gian còn chờ workload projection và phê duyệt.

P7C.4 overall vẫn in progress. P7C.4A đã hoàn tất planning tại `docs/P7C4A_MLP_SEARCH_AND_COMPUTE_BENCHMARK_PLAN.md` và `configs/protocols/p7c/p7c4a_mlp_compute_benchmark_plan.yaml`; P7C.4B blocked awaiting human approval. Ba scenario candidate budget và toàn bộ threshold chỉ là đề xuất để human review; benchmark CPU/GPU bounded chưa chạy, không có runner và không khóa DR-P7C-03/04 hay final MLP manifest.

P7C.7 chỉ có thể tuyên bố full protocol ready khi dataset/CV/preprocessing/seed policy, model scope, concurrency/retry/retention policy và tất cả search-space/budget decision bắt buộc đều có manifest/decision evidence. P7C.1 complete không làm full scientific execution ready.

## Decision register

| ID | Quyết định | Trạng thái | Evidence cần có | Checkpoint | Blocking | Hệ quả/artifact |
| --- | --- | --- | --- | --- | --- | --- |
| DR-P7C-01 | RF final grid/budget | Resolved | Full P7A/Table-2 grid, validated CPU pilot, user-task approval | .2 | Không | Shared RF/XGB manifest/record |
| DR-P7C-02 | XGBoost final grid/budget | Resolved | Full P7A/Table-2 grid, validated CPU pilot, user-task approval | .2 | Không | Shared RF/XGB manifest/record |
| DR-P7C-03 | Final MLP-1/3 candidate budget and search strategy | Open — pending human approval | Chọn một scenario candidate budget định trước cho MLP-1/3, compute/benchmark evidence, không dùng pilot metric | .4B | Có | MLP manifest/record |
| DR-P7C-04 | MLP-5 core-scope inclusion decision | Open — pending human approval | Giữ hoặc loại MLP-5 theo scope/compute evidence, không dùng predictive performance | .4B | Có | Decision record |
| DR-P7C-05 | CatBoost grid and Protocol A/B | Open | Extension scope, native-preprocessing decision | .4 | Có cho extension | Manifest/record |
| DR-P7C-06 | TabNet grid/GPU feasibility | Open | GPU feasibility pilot and budget | .5 | Có cho extension | Manifest/record |
| DR-P7C-07 | FT-Transformer scope | User approval required | Extension-only vs main scope, GPU budget | .6 | Không cho core | Decision record |
| DR-P7C-08 | Unified seed policy | Open | Determinism contract | .7 | Có | Unified manifest |
| DR-P7C-09 | Concurrency, retry/failure, retention | Open | Operational resource and provenance policy | .7 | Có | Unified manifest |
| DR-P7C-10 | Change to locked manifest | Open | New decision record, compatibility/provenance impact | .7 | Có | Superseding manifest |

Rủi ro chính: C4.5-to-CART deviation; HMEQ provenance caveat; unmeasured compute/GPU; MLP/deep-model seed variance; và Protocol A/B không được trộn. Các lựa chọn DR-P7C-04, DR-P7C-07, thu hẹp paper grids hoặc thay CV/dataset scope không được tự động quyết định.
