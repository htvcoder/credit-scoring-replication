# Deviation và giới hạn hiện tại

- Repository chỉ có 6 dataset công khai trong 10 dataset của paper gốc, nên phạm vi là partial replication.
- DBN không nằm trong core scope hiện tại.
- HMEQ có caveat provenance: artifact full khớp shape/schema/class distribution nhưng checksum không phải checksum SAS kỳ vọng.
- Decision Tree C4.5 và mô hình baseline Phase 5 vẫn chưa được triển khai.
- EMP exact không được tái lập trong Phase 4; metric hiện được giữ ở trạng thái `unsupported` vì thiếu business parameters/distribution có provenance rõ ràng.
- Phase 1 website/CI/CD đã Completed: production vận hành qua HTTP/public IP, manual rollback production PASS và automatic failed-deployment rollback PASS. Domain và HTTPS vẫn Optional.
- Phase 2 experiment foundation đã Completed: loader, deterministic split, artifact contract và smoke runner đã được nghiệm thu kỹ thuật.
- Chưa có kết quả thực nghiệm khoa học chính thức; mọi bảng metric, ranking hoặc kết luận định lượng từ smoke runs đều ngoài phạm vi công bố.
