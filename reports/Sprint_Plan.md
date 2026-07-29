# Kế hoạch dự án — phân chia Sprint

## Tổng quan Sprint

| STT | Sprint | Tên / Chủ đề Sprint | Mục tiêu chính | Vấn đề trọng tâm cần giải quyết | Ngày bắt đầu | Ngày kết thúc | Trạng thái | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Sprint 1 | Nền tảng dự án và thực nghiệm | Hoàn thiện nền tảng dữ liệu, website, pipeline thực nghiệm, tiền xử lý và metric. | Tạo môi trường nghiên cứu có thể kiểm chứng, tránh rò rỉ dữ liệu và đánh giá đúng. | 16/07/2026 | 25/07/2026 | Hoàn thành | P0–P4 |
| 2 | Sprint 2 | Triển khai mô hình và chạy thực nghiệm | Xây dựng các mô hình, chạy replication cốt lõi và đánh giá mô hình hiện đại. | Bảo đảm kết quả mô hình có thể tái lập, so sánh công bằng và quản lý tài nguyên chạy. | 27/07/2026 | 08/08/2026 | Đang thực hiện | P5–P8 |
| 3 | Sprint 3 | Phân tích, hoàn thiện báo cáo và bàn giao | Phân tích thống kê, kiểm tra độ bền vững, hoàn thiện báo cáo và website kết quả. | Đưa ra kết luận đáng tin cậy và đóng gói sản phẩm có thể tái lập. | 10/08/2026 | 22/08/2026 | Chưa bắt đầu | P9–P11 |

## Sprint 1

| STT | Vấn đề cần giải quyết | Câu hỏi nhỏ cần giải quyết | Phương pháp áp dụng | Đánh giá kết quả theo phương pháp | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| 1 | Xác minh và chuẩn hóa dữ liệu nghiên cứu | Dữ liệu, metadata và các lưu ý về chất lượng đã đủ tin cậy chưa? | Kiểm tra dataset, checksum, data card và test xác minh. | Dataset đạt kiểm tra; các caveat được ghi nhận rõ. | P0 |
| 2 | Xây dựng nền tảng website và triển khai | Làm thế nào để giới thiệu đề tài và duy trì website an toàn, ổn định? | Phát triển website, Docker, CI/CD, triển khai production và kiểm tra vận hành. | Website truy cập được; build, deploy, health check và rollback đạt yêu cầu. | P1 |
| 3 | Xây dựng pipeline thực nghiệm tái lập | Làm thế nào để tải dữ liệu, chia tập và lưu artifact nhất quán? | Thiết kế loader, deterministic split, cấu hình và smoke runner. | Pipeline chạy được, artifact có provenance và smoke validation hợp lệ. | P2 |
| 4 | Bảo đảm tiền xử lý và đánh giá không rò rỉ dữ liệu | Tiền xử lý và metric có được áp dụng đúng trên từng fold không? | Xây pipeline train-only, nested CV; kiểm chứng AUC, Brier, Partial Gini và EMP. | Leakage test và metric reference test đạt; metric được tài liệu hóa rõ. | P3–P4 |

## Sprint 2

| STT | Vấn đề cần giải quyết | Câu hỏi nhỏ cần giải quyết | Phương pháp áp dụng | Đánh giá kết quả theo phương pháp | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| 1 | Triển khai mô hình truyền thống và ensemble | Các mô hình LR, cây quyết định, RF và XGBoost có chạy nhất quán theo protocol không? | Xây model factory, cấu hình tuning và kiểm tra đầu ra xác suất. | Reduced validation đạt; cấu hình, seed và artifact được ghi nhận. | P5 |
| 2 | Đánh giá ảnh hưởng độ sâu MLP | MLP-1, MLP-3 và MLP-5 có thể huấn luyện công bằng, ổn định không? | Triển khai PyTorch MLP, scaling train-only, early stopping và logging. | Validation trên các dataset đại diện đạt; runtime và artifact hợp lệ. | P6 |
| 3 | Chạy replication cốt lõi | Các mô hình có tạo kết quả hợp lệ trên các dataset theo Protocol A không? | Dry run, chạy thực nghiệm theo fold, kiểm tra artifact, resume và xử lý lỗi. | Mỗi tổ hợp dataset–mô hình có kết quả hợp lệ hoặc trạng thái lỗi rõ ràng. | P7 |
| 4 | Đánh giá lại bằng mô hình hiện đại | Mô hình hiện đại có thay đổi kết luận so với baseline XGBoost không? | Triển khai CatBoost; đánh giá TabNet/FT-Transformer theo khả năng tài nguyên; tách Protocol A/B. | Kết quả so sánh hợp lệ, protocol không bị trộn và quyết định no-go được ghi rõ. | P8 |

## Sprint 3

| STT | Vấn đề cần giải quyết | Câu hỏi nhỏ cần giải quyết | Phương pháp áp dụng | Đánh giá kết quả theo phương pháp | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| 1 | So sánh thống kê kết quả mô hình | Khác biệt giữa các mô hình có ý nghĩa và được phân tích đúng đơn vị thống kê không? | Tổng hợp theo dataset; xếp hạng, Friedman/Rom, Nemenyi và Bayesian signed-rank. | Bảng thống kê tái tạo được từ artifact; nêu rõ giới hạn số dataset. | P9 |
| 2 | Kiểm tra độ bền vững của kết luận | Kết luận có ổn định trước seed, dataset caveat và lựa chọn preprocessing/protocol không? | Phân tích sensitivity theo seed, HMEQ/GMC, VIF, calibration và protocol. | Các ảnh hưởng quan trọng có kết luận hoặc no-go được tài liệu hóa. | P10 |
| 3 | Hoàn thiện báo cáo và gói tái lập | Làm thế nào để công bố kết quả chính xác, an toàn và tái lập được? | Sinh bảng/biểu đồ, viết báo cáo và hướng dẫn tái lập; sanitize artifact; cập nhật và deploy website. | Báo cáo và website trả lời RQ1–RQ3; không lộ dữ liệu nhạy cảm; clean clone có thể tái tạo. | P11 |
