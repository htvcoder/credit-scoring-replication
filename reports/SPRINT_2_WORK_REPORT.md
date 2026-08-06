# BÁO CÁO TIẾN ĐỘ CÔNG VIỆC – SPRINT 2

- **Người báo cáo:** Hoàng Trọng Vĩnh
- **Người hướng dẫn:** Trần Công Phú Khánh
- **Thời gian:** Sprint 2 (03/08/2026 - 15/08/2026)
- **Đơn vị:** VinSmart Future - Fintech
- **Dự án:** Credit Scoring Replication
- **Phạm vi Sprint:** Phase 5 đến hết Phase 8
- **Ngày cập nhật:** 05/08/2026

> Báo cáo phản ánh trạng thái thực tế tại thời điểm tạo; Sprint 2 chưa kết thúc và các công việc chưa hoàn thành sẽ tiếp tục được triển khai đến ngày 15/08/2026.

## Tình hình thực hiện công việc

| STT | Mô tả | Trạng thái | Ghi chú |
| --: | ----- | ---------- | ------- |
| 1 | Phase 5 – P5A: Xây dựng model contract và cấu hình dùng chung cho mô hình truyền thống. | Hoàn thành | Chuẩn hóa stable model ID, metadata, seed và `y_score = P(class 1 = bad/default)`. |
| 2 | Phase 5 – P5B: Triển khai Logistic Regression, CART, Random Forest và XGBoost. | Hoàn thành | CART được ghi nhận rõ là deviation C4.5-to-CART; các default reduced không phải paper-reference grid. |
| 3 | Phase 5 – P5C: Tích hợp classical models với nested-CV model-validation harness. | Hoàn thành | Có atomic per-fold artifact, resume/retry, provenance và regression tests; artifact validation không phải scientific result. |
| 4 | Phase 6 – P6A: Xây dựng nền tảng huấn luyện PyTorch MLP. | Hoàn thành | Typed config, logits/probability contract, seed/device policy và early stopping foundation đã có. |
| 5 | Phase 6 – P6B: Triển khai MLP-1, MLP-3 và MLP-5. | Hoàn thành | Độ sâu theo paper; width/training budget là project decision có provenance. |
| 6 | Phase 6 – P6C: Tích hợp neural nested-CV, artifacts, resume/retry và engineering validation. | Hoàn thành có lưu ý | GC/TC reduced validation chỉ là evidence kỹ thuật non-publishable, không có ranking hoặc kết quả khoa học. |
| 7 | Phase 7 – P7A: Khóa protocol, candidate manifest và kiểm tra integrity trước pilot. | Hoàn thành | Protocol sáu dataset, outer/inner CV, seed, preprocessing boundary và manifest hash đã được xác định. |
| 8 | Phase 7 – P7B.1: Hoàn thiện CART feasibility runner, preflight và hardening. | Hoàn thành có lưu ý | Runner/CLI, provenance, telemetry, artifact identity và resume phục vụ feasibility engineering, không phải scientific execution. |
| 9 | Phase 7 – P7B.2: Closeout CART engineering-feasibility và decision record. | Hoàn thành có lưu ý | `60/60` fit, không failed/pending; final CART-A 12 candidates đã khóa, nhưng telemetry chỉ là non-publishable engineering evidence. |
| 10 | Phase 7 – P7C.1: Lập inventory, readiness matrix, decision register và validator cho final protocol. | Hoàn thành | Không chạy training, ranking, outer refit hay tạo metric khoa học. |
| 11 | Phase 7 – P7C: Chuẩn bị final scientific protocol và các quyết định còn lại. | Đang thực hiện | `progress.yaml` xác nhận P7C in progress; CART đã locked nhưng các model/budget khác chưa khóa. |
| 12 | Phase 7 – P7C.2: Quyết định final grid/budget cho Random Forest và XGBoost. | Chưa thực hiện | RF 30 và XGBoost 108 candidate hiện chỉ là reference search space; cần manifest, budget và evidence riêng. |
| 13 | Phase 7 – P7C.3: Quyết định scope/budget cho MLP-1, MLP-3 và MLP-5. | Chưa thực hiện | MLP-5 cần quyết định inclusion và compute budget; đây là dependency của unified readiness gate. |
| 14 | Phase 7 – P7C.4: Quyết định CatBoost và Protocol A/B. | Chưa thực hiện | Cần chốt preprocessing scope, grid và budget trước feasibility pilot hoặc final protocol. |
| 15 | Phase 7 – P7C.5: Feasibility/final decision cho TabNet. | Chưa thực hiện | Phụ thuộc GPU evidence, protocol và grid; chỉ nên chạy pilot non-publishable khi có resource checkpoint. |
| 16 | Phase 7 – P7C.6: Quyết định FT-Transformer extension. | Chưa thực hiện | Cần user scope approval, GPU và budget; extension này không phải điều kiện core nếu có no-go hợp lệ. |
| 17 | Phase 7 – P7C.7: Unified final manifest và readiness gate. | Chưa thực hiện | Chỉ mở core scientific execution khi các quyết định P7C.2–P7C.6 được resolve/defer có bằng chứng. |
| 18 | Phase 7 – Core replication run trên sáu dataset. | Chưa thực hiện | Chưa có scientific execution hay scientific result; phụ thuộc P7C.7 và protocol/budget đã khóa. |
| 19 | Phase 8 – CatBoost modern reassessment. | Chưa thực hiện | CatBoost là minimum evidence bắt buộc cho RQ3; phụ thuộc baseline/protocol/metric tương thích và Phase 7 closeout phù hợp. |
| 20 | Phase 8 – TabNet modern reassessment. | Chưa thực hiện | Should/Optional theo resource checkpoint; cần kiểm tra GPU/CPU config, protocol separation và probability output. |
| 21 | Phase 8 – FT-Transformer modern reassessment. | Chưa thực hiện | Optional theo resource checkpoint; no-go có lý do hợp lệ không làm hỏng core scope. |

## Tổng hợp tiến độ Sprint 2

Báo cáo theo dõi 21 công việc/checkpoint trong phạm vi Phase 5–8: 7 **Hoàn thành**, 3 **Hoàn thành có lưu ý**, 1 **Đang thực hiện**, 10 **Chưa thực hiện**, 0 **Tạm hoãn** và 0 **Bị chặn** theo bằng chứng hiện có. Phase hiện tại là Phase 7; checkpoint đang thực hiện là P7C, trong đó P7C.1 đã closeout và P7C.2 là bước quyết định tiếp theo. Phase gần nhất hoàn thành toàn bộ là Phase 6. Phase 8 chưa bắt đầu; vì vậy chưa đủ cơ sở để kết luận mục tiêu hoàn thành Phase 8 đã đạt mức an toàn, dù nền tảng Phase 5–6 và một phần protocol/feasibility Phase 7 đã sẵn sàng.

## Công việc còn lại để hoàn thành Sprint 2

### Ưu tiên 1 – Hoàn thành readiness và core protocol của Phase 7

- Hoàn thành P7C.2–P7C.7: khóa hoặc defer có căn cứ các grid/budget RF, XGBoost, MLP, CatBoost, TabNet và FT-Transformer; chốt seed, concurrency, retry/retention và unified manifest.
- Chỉ bắt đầu core replication sau P7C.7; thực hiện nested-CV theo protocol đã khóa, validate incremental artifacts, resume failures và giữ HMEQ caveat.
- Tách rõ engineering-feasibility của P7B khỏi scientific execution; CART 12-candidate grid đã khóa không tự động khóa search space/budget của các model khác.

### Ưu tiên 2 – Bắt đầu và triển khai Phase 8

- Triển khai CatBoost trước vì đây là minimum evidence cho RQ3; so sánh trên dataset/metric/protocol phù hợp với XGBoost baseline.
- Xác định và kiểm tra rõ Protocol A/B, không trộn artifacts hoặc kết quả giữa hai protocol.
- Chỉ quyết định chạy TabNet và FT-Transformer sau resource checkpoint; nếu no-go, lưu lý do kỹ thuật và giới hạn phạm vi rõ ràng.

### Ưu tiên 3 – Closeout Phase 8 và Sprint 2

- Validate modern artifacts, probability output, provenance và resource evidence; chỉ công bố aggregate đã được sanitize khi đủ điều kiện.
- Đồng bộ tài liệu/website/project status sau khi acceptance criteria Phase 8 thực sự pass; không coi smoke, reduced run hoặc engineering validation là scientific result.

## Đánh giá khả năng hoàn thành mục tiêu Sprint

Tại ngày 05/08/2026 còn 10 ngày đến mốc 15/08/2026. Tiến độ nền tảng từ Phase 5–6 và P7A/P7B/P7C.1 là phù hợp, nhưng Phase 7 core scientific execution chưa bắt đầu và nhiều quyết định protocol/budget còn mở. Phase 8 cũng chưa bắt đầu, trong khi CatBoost là yêu cầu bắt buộc cho RQ3 và TabNet/FT-Transformer phụ thuộc resource checkpoint/GPU. Vì vậy mục tiêu hoàn thành Phase 8 chỉ có khả năng đạt nếu các decision gate P7C được xử lý nhanh, compute phù hợp và artifact validation không phát sinh failure; chưa đủ bằng chứng để kết luận chắc chắn.

## Kết quả Sprint 2 tại thời điểm báo cáo

Phase 5 đã hoàn thiện contract, classical/ensemble models và nested-CV model-validation harness; Phase 6 đã hoàn thiện MLP-1/3/5 cùng neural hardening. P7A đã khóa protocol bất biến, còn P7B đã closeout CART engineering-feasibility và khóa final CART-A grid 12 candidates. P7C.1 đã cung cấp inventory, readiness matrix và decision register, nhưng P7C cùng full core scientific protocol vẫn đang thực hiện. Chưa có scientific experiment thực tế hoặc scientific result từ Phase 7; các artifact validation/feasibility chỉ là evidence kỹ thuật non-publishable. Phase 8 chưa bắt đầu, nên mục tiêu còn lại của Sprint là hoàn thành readiness/core dependency phù hợp và modern reassessment trước khi Sprint kết thúc.

## Lưu ý

- Đây là báo cáo tiến độ giữa Sprint, không phải báo cáo tổng kết cuối Sprint; Sprint 2 kéo dài từ 03/08/2026 đến 15/08/2026.
- Báo cáo chỉ phản ánh trạng thái có bằng chứng trong repository tại thời điểm 05/08/2026. Nguồn trạng thái ưu tiên là `website/content/progress.yaml`.
- Các công việc chưa hoàn thành được nêu để theo dõi tiến độ, không phải kết quả đã đạt được.
- Synthetic run, dry-run, smoke run, reduced run và engineering validation không được coi là scientific result.
- Runtime artifacts, predictions, model weights và dữ liệu nghiên cứu không được commit khi không được artifact contract/chính sách repository cho phép.
- `progress.yaml`, README, website methods và P7C plan cùng ghi P7C.2 là bước tiếp theo; do đó báo cáo không ghi P7C.2 hoặc Phase 8 là đã bắt đầu dù có các foundation/decision documents liên quan.
- Tại thời điểm rà soát, Git HEAD là `b3de9f9` (merge P7C.1); không có manifest, runner hoặc closeout record P7C.2.1 trong working tree hiện tại. Báo cáo ưu tiên trạng thái repository hiện hữu và không suy diễn checkpoint này đã hoàn thành.
