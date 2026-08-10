# Phương pháp dự kiến

Website hiện mô tả phương pháp ở mức kế hoạch và trạng thái nghiệm thu kỹ thuật. P7C.3: Completed — canonical feasibility pilot accepted. P7C.4B.1: Completed — harness CPU sequential/parallel-2 đã qua independent operational readiness review. P7C.4B.2a đã khóa scientific scope MLP balanced 24/48/48 và giữ MLP-5. P7C.4B.2b executable bounded-preflight harness đã sẵn sàng và chỉ được kiểm thử bằng fixture; target compute preflight, runtime/cost, canonical mode và canonical/GPU execution vẫn pending/NO-GO. Đây không phải kết quả hiệu năng dự báo hoặc kết quả khoa học cuối cùng.

## Replication core

- Sử dụng 6 dataset công khai đã được xác minh ở Phase 0.
- Dùng dataset registry và loader thống nhất; target được chuẩn hóa về `0 = non-default` và `1 = default`.
- Tạo stratified deterministic split có dataset checksum, config hash và split hash.
- Giữ nguyên nguyên tắc partial replication và không tuyên bố full replication.
- Tách rõ preprocessing, split, model fitting và metric để tránh leakage; preprocessing chỉ được fit trên train, test chỉ được transform bằng state đã học từ train.
- Lưu experiment artifacts và Git provenance để hỗ trợ tái tạo.
- So sánh các baseline chính, XGBoost và MLP theo độ sâu trong các phase sau.

## Nền tảng Phase 2

- Smoke runner đã chạy thành công với Logistic Regression và XGBoost để kiểm tra end-to-end pipeline.
- Smoke preprocessing chỉ dùng để nghiệm thu runner, gồm imputation tối thiểu, one-hot encoding và scaling cho Logistic Regression khi cần.
- Smoke metrics chỉ xác nhận artifact và prediction probability hợp lệ; không dùng để so sánh mô hình hoặc kết luận nghiên cứu.

## Nền tảng Phase 3

- P3A đã hoàn thành train-only numeric mean imputation, train-only categorical mode imputation, deterministic mode tie-break, unseen-category handling, immutable fitted preprocessing metadata và transform diagnostics tách khỏi fitted state.
- P3B đã hoàn thành WOE cho categorical features, numeric passthrough sau imputation vì paper không đặc tả numeric binning, WOE smoothing mặc định `0.5`, unknown WOE fallback `0.0`, iterative VIF threshold mặc định `10.0`, zero-variance filtering trước VIF và optional train-only standard scaling.
- P3C đã hoàn thành deterministic repeated stratified two-fold outer CV, stratified inner CV, deterministic seed derivation, fold hashes, nested CV hash, fresh preprocessing pipeline cho từng fold, outer-test isolation và tuning chỉ dựa trên inner validation.
- P3C artifacts là non-publishable preprocessing-validation artifacts; chúng không phải scientific results.

## Bước tiếp theo

- P7C.4B.2a đã ghi nhận DR-P7C-03/04 approved và digest-locked proposed MLP scientific scope; đây là reduced deterministic subset, không phải exhaustive replication. P7C.4B.2b đã cung cấp CLI/guards, process-isolated runner, atomic artifact/resume/retry validator, telemetry reconstruction và fail-closed projection/RAM/cost/execution-plan contracts. Target single-VM preflight vẫn là bước tiếp theo. Fixture validation chỉ là engineering evidence non-publishable; không dùng predictive metric/ranking để giảm grid.
- Core replication chỉ được bắt đầu cho scope đã khóa sau P7C.7 readiness gate. Không có pilot metric nào được dùng để chọn CART candidate.

## P7C.4B.2c outer-refit/overhead preflight

- Implementation đã hoàn thành ở mức code/contract: population 270 outer refit được suy ra từ 90 outer partitions × ba MLP; sampling phân tầng theo dataset, model, candidate proxy và CPU mode.
- Execution core dùng process spawn, tách warmup khỏi measured phase, instrument canonical preprocessing/refit path, ghi artifact atomic, resume sample hợp lệ và quarantine state hỏng.
- Synthetic validation chỉ kiểm tra lifecycle bằng fixture cực nhỏ và luôn là non-scientific evidence. Target outer-refit/overhead preflight chưa chạy, runtime/cost và canonical mode vẫn unknown, execution plan vẫn fail closed.

## P7C.4B.2d target authorization readiness

- Decision-review contract đã hoàn thành: plan 324 task được kiểm tra theo strata, canary/staged stop gates, target-environment input gồm VM count do operator xác nhận, cost formula không giả định giá và authorization proposal non-effective.
- Chưa có target environment, price/budget, canary approval hay effective authorization. Vì vậy target preflight chưa chạy và canonical execution vẫn NO-GO.

## Modern reassessment

- CatBoost là phần Must của Phase 8 để có minimum evidence cho RQ3.
- TabNet và FT-Transformer là mở rộng có điều kiện theo resource checkpoint.
- Protocol A và Protocol B phải được báo cáo riêng, không trộn kết quả.

## Công bố kết quả

- Chỉ công bố artifact đã tổng hợp, kiểm chứng và sanitize.
- Không công bố raw data, processed data nội bộ, prediction cấp bản ghi, local path, secret hoặc log nhạy cảm.
