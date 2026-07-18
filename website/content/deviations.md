# Deviation và giới hạn hiện tại

- Repository chỉ có 6 dataset công khai trong 10 dataset của paper gốc, nên phạm vi là partial replication.
- DBN không nằm trong core scope hiện tại.
- HMEQ có caveat provenance: artifact full khớp shape/schema/class distribution nhưng checksum không phải checksum SAS kỳ vọng.
- Decision Tree C4.5, EMP exact, WOE/VIF chống leakage và nested CV chi tiết sẽ được chốt trong các phase thực nghiệm sau.
- Phase 1 website/CI/CD đã vận hành production qua HTTP/public IP, nhưng manual rollback production và automatic failed-deployment rollback chưa được xác nhận kiểm thử thật.
- Chưa có kết quả thực nghiệm; mọi bảng metric, ranking hoặc kết luận định lượng đều ngoài phạm vi.
