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
| MLP-1 | replication baseline | Paper reference | Unlocked | 144 | CPU/GPU | P7C.3 `vm-run-001` failed before training on old contract; hotfix verification; runtime/budget decision outstanding | P7C.3 |
| MLP-3 | replication baseline | Unlocked | 720 | CPU/GPU | P7C.3 `vm-run-001` failed before training on old contract; hotfix verification; runtime/budget decision outstanding | P7C.3 |
| MLP-5 | replication baseline | Paper reference | Decision pending | 2,016 | GPU khuyến nghị | P7C.3 `vm-run-001` failed before training on old contract; no predictive exclusion; scope/budget decision outstanding | P7C.3 |
| CatBoost | main-results extension | Chưa có grid | Decision pending | TBD | CPU/GPU | Protocol A/B, grid, budget | P7C.4 |
| TabNet | main-results extension | Chưa có grid | Feasibility required | TBD | GPU khuyến nghị | GPU + protocol + grid | P7C.5 |
| FT-Transformer | optional extension | Chưa có grid | Feasibility required | TBD | GPU khuyến nghị | Scope approval + GPU + grid | P7C.6 |

`candidate_count` của RF/XGBoost/MLP là số tham chiếu từ Table 2, không phải final locked budget. Với model tuning, inner fits = candidate count × 5 × outer partitions; outer refits = outer partitions. CART đã có 5,400 inner fits và 90 refits. Các thời gian chỉ là số đo engineering CART hoặc ngoại suy đã gắn caveat; không có benchmark đủ để báo thời gian chính xác cho model khác.

## Checkpoint plan

| Checkpoint | Nội dung/output | Entry → exit/validation | Có feasibility training? |
| --- | --- | --- | --- |
| P7C.1 | Inventory, validator, readiness matrix, decision register | P7B closed → inventory validated, docs/status/build pass | Không |
| P7C.2 | RF/XGBoost decision records và final manifests | Completed: full grids/provenance locked and protocol tests pass | P7C.2.2 pilot đã completed/validated; không phải scientific execution |
| P7C.3 | MLP-1/3/5 scope và budget decisions | **In progress:** `vm-run-001` failed before training at the old plan/factory contract boundary; immutable CPU-only plan remains unchanged and the runner hotfix is under verification, so valid runtime/RSS/projection, scope/budget and final manifests remain unresolved | Chỉ feasibility execution được phê duyệt riêng; không dùng predictive metric để loại MLP-5 hoặc giảm grid |
| P7C.4 | CatBoost Protocol A/B và final decision | P7C.1 → preprocessing scope, grid/budget locked | Chỉ pilot được phê duyệt riêng |
| P7C.5 | TabNet feasibility/final decision | P7C.1 → GPU evidence + bounded protocol decision | Có thể, nhưng non-publishable pilot riêng |
| P7C.6 | FT-Transformer extension decision | P7C.1 → scope/GPU/budget decision | Có thể, nhưng non-publishable pilot riêng |
| P7C.7 | Unified final manifest and readiness gate | P7C.2–.6 decisions resolved/deferred explicitly → all references/hashes/tests pass | Không; đây là gate trước execution |

P7C.2 được triển khai theo ba sub-checkpoint: P7C.2.1 hoàn tất immutable 60-fit plan và execution harness; P7C.2.2 đã completed và artifact validation pass với 60/60 fits; P7C.2.3 đã hoàn tất analysis/decision record. Theo approval `user_task_instruction`, DR-P7C-01/02 khóa full P7A/Table-2 grids cho RF/XGBoost trong `configs/protocols/p7c/p7c_rf_xgboost_final_manifest.yaml`. Việc hoàn tất P7C.2 không làm full scientific execution ready.

P7C.3 chọn Branch B: `configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml` là kế hoạch engineering-feasibility bất biến, không phải final search-space/budget manifest. `vm-run-001` trên commit cũ thất bại trước training do contract mismatch và không được resume hoặc tái sử dụng; lần chạy sau hotfix phải dùng output directory mới. Kế hoạch khóa CPU-only sequential policy và cấm dùng kết quả feasibility/predictive để loại MLP-5 hoặc chọn/reduce grid hậu nghiệm.

P7C.7 chỉ có thể tuyên bố full protocol ready khi dataset/CV/preprocessing/seed policy, model scope, concurrency/retry/retention policy và tất cả search-space/budget decision bắt buộc đều có manifest/decision evidence. P7C.1 complete không làm full scientific execution ready.

## Decision register

| ID | Quyết định | Trạng thái | Evidence cần có | Checkpoint | Blocking | Hệ quả/artifact |
| --- | --- | --- | --- | --- | --- | --- |
| DR-P7C-01 | RF final grid/budget | Resolved | Full P7A/Table-2 grid, validated CPU pilot, user-task approval | .2 | Không | Shared RF/XGB manifest/record |
| DR-P7C-02 | XGBoost final grid/budget | Resolved | Full P7A/Table-2 grid, validated CPU pilot, user-task approval | .2 | Không | Shared RF/XGB manifest/record |
| DR-P7C-03 | MLP-1/3 budget | Open | Training budget + feasibility evidence | .3 | Có | MLP manifest/record |
| DR-P7C-04 | Keep or exclude MLP-5 | User approval required | Scope and compute evidence | .3 | Có | Decision record |
| DR-P7C-05 | CatBoost grid and Protocol A/B | Open | Extension scope, native-preprocessing decision | .4 | Có cho extension | Manifest/record |
| DR-P7C-06 | TabNet grid/GPU feasibility | Open | GPU feasibility pilot and budget | .5 | Có cho extension | Manifest/record |
| DR-P7C-07 | FT-Transformer scope | User approval required | Extension-only vs main scope, GPU budget | .6 | Không cho core | Decision record |
| DR-P7C-08 | Unified seed policy | Open | Determinism contract | .7 | Có | Unified manifest |
| DR-P7C-09 | Concurrency, retry/failure, retention | Open | Operational resource and provenance policy | .7 | Có | Unified manifest |
| DR-P7C-10 | Change to locked manifest | Open | New decision record, compatibility/provenance impact | .7 | Có | Superseding manifest |

Rủi ro chính: C4.5-to-CART deviation; HMEQ provenance caveat; unmeasured compute/GPU; MLP/deep-model seed variance; và Protocol A/B không được trộn. Các lựa chọn DR-P7C-04, DR-P7C-07, thu hẹp paper grids hoặc thay CV/dataset scope không được tự động quyết định.
