# Phương pháp dự kiến

Website hiện mô tả phương pháp ở mức kế hoạch và trạng thái nghiệm thu kỹ thuật. Phase 2 đã hoàn thành nền tảng experiment có thể tái tạo, nhưng chưa có kết quả nghiên cứu chính thức được công bố.

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
- Scientific preprocessing protocol chống leakage đầy đủ, bao gồm WOE/VIF và các quyết định protocol chính thức, sẽ được triển khai ở Phase 3.
- Smoke metrics chỉ xác nhận artifact và prediction probability hợp lệ; không dùng để so sánh mô hình hoặc kết luận nghiên cứu.

## Modern reassessment

- CatBoost là phần Must của Phase 8 để có minimum evidence cho RQ3.
- TabNet và FT-Transformer là mở rộng có điều kiện theo resource checkpoint.
- Protocol A và Protocol B phải được báo cáo riêng, không trộn kết quả.

## Công bố kết quả

- Chỉ công bố artifact đã tổng hợp, kiểm chứng và sanitize.
- Không công bố raw data, processed data nội bộ, prediction cấp bản ghi, local path, secret hoặc log nhạy cảm.
