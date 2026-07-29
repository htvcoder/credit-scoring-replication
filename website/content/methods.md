# Phương pháp dự kiến

Website hiện mô tả phương pháp ở mức kế hoạch và trạng thái nghiệm thu kỹ thuật. Phase 4 đã hoàn thành metric validation cho ROC AUC, Brier Score, Partial Gini và EMP unsupported handling ở mức non-publishable validation artifact; core replication run và kết quả nghiên cứu chính thức vẫn chưa được công bố.

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

- Phase 6 đã hoàn thành MLP infrastructure/hardening validation; Phase 7 - Core replication run là bước tiếp theo. Các reduced validation artifacts vẫn không phải kết quả khoa học.
- Core replication chỉ được bắt đầu sau khi model factory và baseline models dùng lại đúng preprocessing, nested-CV và metric-validation foundation của Phase 3-4.

## Modern reassessment

- CatBoost là phần Must của Phase 8 để có minimum evidence cho RQ3.
- TabNet và FT-Transformer là mở rộng có điều kiện theo resource checkpoint.
- Protocol A và Protocol B phải được báo cáo riêng, không trộn kết quả.

## Công bố kết quả

- Chỉ công bố artifact đã tổng hợp, kiểm chứng và sanitize.
- Không công bố raw data, processed data nội bộ, prediction cấp bản ghi, local path, secret hoặc log nhạy cảm.
