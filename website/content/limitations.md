# Hạn chế và điều kiện tái thực hiện

## Hạn chế từ paper gốc

- Paper sử dụng 10 dataset, trong đó một số dataset là dữ liệu từ financial institutions và không có trong repository hiện tại.
- Paper được công bố năm 2021, trước khi nhiều mô hình tabular hiện đại trở thành lựa chọn phổ biến trong các benchmark gần đây.
- Các deep learning architecture trong paper gồm MLP và DBN; DBN là kiến trúc mang tính lịch sử hơn so với các mô hình tabular hiện đại.
- Một số chi tiết triển khai như seed, fold split gốc, tie-breaking rule, smoothing WOE hoặc một số hyperparameter vận hành không đủ để tái lập bit-level.

## Hạn chế của đề tài hiện tại

- Repository hiện chỉ có 6 dataset công khai đã được xác minh: AC, GC, HMEQ, TH02, TC và GMC.
- Dự án là partial replication, không phải full replication.
- HMEQ có caveat provenance: artifact full khớp shape/schema/class distribution nhưng checksum không phải checksum SAS artifact kỳ vọng.
- DBN không thuộc core scope hiện tại.
- Các khác biệt phương pháp phải được ghi trong deviation register trước khi công bố kết quả.
- Đề tài thực tập chưa công bố kết quả thực nghiệm.
