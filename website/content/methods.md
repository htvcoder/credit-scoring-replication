# Phương pháp dự kiến

Website hiện chỉ mô tả phương pháp ở mức kế hoạch. Chưa có pipeline thực nghiệm, metrics hoặc kết luận nghiên cứu được công bố.

## Replication core

- Sử dụng 6 dataset công khai đã được xác minh ở Phase 0.
- Giữ nguyên nguyên tắc partial replication và không tuyên bố full replication.
- Tách rõ preprocessing, split, model fitting và metric để tránh leakage.
- So sánh các baseline chính, XGBoost và MLP theo độ sâu trong các phase sau.

## Modern reassessment

- CatBoost là phần Must của Phase 8 để có minimum evidence cho RQ3.
- TabNet và FT-Transformer là mở rộng có điều kiện theo resource checkpoint.
- Protocol A và Protocol B phải được báo cáo riêng, không trộn kết quả.

## Công bố kết quả

- Chỉ công bố artifact đã tổng hợp, kiểm chứng và sanitize.
- Không công bố raw data, processed data nội bộ, prediction cấp bản ghi, local path, secret hoặc log nhạy cảm.
