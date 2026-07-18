# Bối cảnh và mục đích đề tài

Đề tài `Tái lập và đánh giá lại mô hình tính điểm tín dụng` nối tiếp paper gốc bằng một phạm vi thận trọng: kiểm chứng lại những phần có thể truy cập công khai, ghi rõ deviation và không tuyên bố full replication.

## Giới hạn phạm vi dữ liệu

- Paper gốc sử dụng 10 dataset.
- Repository hiện có 6 dataset công khai đã xác minh: AC, GC, HMEQ, TH02, TC và GMC.
- Vì vậy dự án không tuyên bố tái lập đầy đủ toàn bộ thực nghiệm của paper.
- Giới hạn 6/10 dataset phải được xét đến khi diễn giải kết quả.
- Giữ HMEQ caveat trong mọi diễn giải liên quan.

## MLP depth check

- So sánh MLP-1, MLP-3 và MLP-5 trong điều kiện preprocessing, split và metric nhất quán.
- Ghi rõ budget, seed và deviation nếu không tái tạo được full grid của paper.

## Modern reassessment

- Đánh giá lại kết luận ưu tiên XGBoost với tối thiểu CatBoost.
- TabNet và FT-Transformer chỉ thực hiện nếu resource checkpoint cho phép.
- Protocol A và Protocol B phải được báo cáo riêng.

## Vai trò website

- Báo cáo tiến độ dự án.
- Trình bày bối cảnh, dữ liệu và phương pháp.
- Công bố các kết quả aggregated đã được kiểm chứng ở phase sau.
- Cung cấp thông tin truy vết version hoặc commit đang triển khai.
- Không phải môi trường chạy thực nghiệm.
- Không cung cấp raw dataset.
- Không phải mục tiêu nghiên cứu chính của đề tài.
