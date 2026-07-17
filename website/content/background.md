# Bối cảnh và mục đích đề tài

Đề tài thực tập nối tiếp paper gốc bằng một phạm vi thận trọng hơn: kiểm chứng lại những phần có thể truy cập công khai, ghi rõ deviation và không tuyên bố full replication.

## Partial replication

- Kiểm tra xem các kết luận chính của paper có còn xuất hiện trên 6 dataset công khai đã xác minh hay không.
- Giữ HMEQ caveat trong mọi diễn giải liên quan.
- Không suy rộng kết quả từ 6 dataset thành kết luận cho toàn bộ 10 dataset của paper.

## MLP depth check

- So sánh MLP-1, MLP-3 và MLP-5 trong điều kiện preprocessing, split và metric nhất quán.
- Ghi rõ budget, seed và deviation nếu không tái tạo được full grid của paper.

## Modern reassessment

- Đánh giá lại kết luận ưu tiên XGBoost với tối thiểu CatBoost.
- TabNet và FT-Transformer chỉ thực hiện nếu resource checkpoint cho phép.
- Protocol A và Protocol B phải được báo cáo riêng.

## Vai trò website

- Giới thiệu đề tài thực tập.
- Trình bày phương pháp và giới hạn.
- Theo dõi tiến độ.
- Công bố các kết quả aggregated đã được kiểm chứng ở phase sau.
- Không phải môi trường chạy source code hoặc tải raw dataset.
