# Credit Scoring Replication

## Tổng quan

Repository này chuẩn bị cho một nghiên cứu **partial replication** và mở rộng bài toán credit scoring từ paper **Deep Learning for Credit Scoring: Do or Don't?**. Dự án tái chạy các phần có thể kiểm chứng trên những bộ dữ liệu công khai, đồng thời thử nghiệm thêm một số mô hình tabular hiện đại để đánh giá lại kết luận của nghiên cứu gốc.

## Paper gốc

- Tên: **Deep Learning for Credit Scoring: Do or Don't?**
- Tác giả: Björn Rafn Gunnarsson và cộng sự
- Tạp chí: *European Journal of Operational Research*
- Nhà xuất bản: Elsevier
- Năm: 2021
- Volume 295, Issue 1, trang 292-305
- DOI: `10.1016/j.ejor.2021.03.006`

## Chất lượng học thuật và xếp hạng nguồn công bố

### Chất lượng và xếp hạng của tạp chí

Bài báo được công bố trên *European Journal of Operational Research* (EJOR), một tạp chí quốc tế có phản biện khoa học trong lĩnh vực Operations Research, Management Science, mô hình hóa và hỗ trợ ra quyết định. EJOR được xuất bản bởi Elsevier và phối hợp với Association of European Operational Research Societies (EURO).

Theo SCImago Journal & Country Rank, EJOR thuộc nhóm **SCImago Q1 (2024)**. Theo số liệu tạp chí năm 2024 do Elsevier/ScienceDirect công bố, EJOR có **CiteScore 13,2 (2024)** và **Journal Impact Factor 6,0 (2024)**. Báo cáo EJOR/EURO năm 2025 cũng ghi nhận EJOR ở **87,3 percentile trong Operations Research & Management Science (2024)**. Các chỉ số này cho thấy EJOR là một tạp chí có uy tín và ảnh hưởng cao trong các lĩnh vực liên quan, nhưng cần hiểu đúng phạm vi của chúng:

> Q1 là xếp hạng của tạp chí trong một hoặc nhiều nhóm ngành, không phải xếp hạng riêng của bài báo. Nói chính xác hơn: bài báo được công bố trên *European Journal of Operational Research*, một tạp chí thuộc nhóm Q1 theo SCImago năm 2024.

Thông tin xếp hạng và chỉ số tạp chí được kiểm tra lần gần nhất vào tháng 07/2026; số liệu bibliometric được ghi kèm năm tham chiếu 2024.

### Đánh giá chất lượng phương pháp nghiên cứu

Paper gốc có thiết kế thực nghiệm tương đối toàn diện và nghiêm ngặt so với nhiều nghiên cứu credit scoring chỉ đánh giá trên một hoặc một số ít bộ dữ liệu. Nghiên cứu thực nghiệm trên 10 bộ dữ liệu credit scoring và so sánh 10 cấu hình mô hình: Logistic Regression, Decision Tree C4.5, Random Forest, XGBoost, MLP với 1, 3 và 5 lớp ẩn, cùng DBN với 1, 3 và 5 lớp ẩn.

Việc đánh giá không chỉ dựa trên một chỉ số đơn lẻ mà kết hợp nhiều góc độ: AUC, partial Gini, Brier Score và Expected Maximum Profit (EMP). Nhờ đó, paper xem xét đồng thời hiệu quả phân loại, calibration của xác suất dự báo và giá trị kinh tế của mô hình. Paper cũng sử dụng cả kiểm định thống kê frequentist và Bayesian, phù hợp với vai trò của một nghiên cứu benchmark và là nền tảng hợp lý cho replication.

### Hạn chế cần lưu ý

- 4 trong 10 bộ dữ liệu của nghiên cứu gốc là dữ liệu riêng tư và không thể tải công khai; vì vậy dự án hiện tại chỉ thực hiện **partial replication** trên 6 bộ dữ liệu công khai.
- Deep Belief Network là một kiến trúc tương đối cũ so với các mô hình tabular hiện đại.
- Kết luận của paper năm 2021 không mặc nhiên áp dụng cho CatBoost, TabNet, FT-Transformer hoặc các kiến trúc mới hơn.
- Xếp hạng cao của tạp chí không có nghĩa là mọi kết luận của paper không cần được kiểm chứng lại bằng replication.
- Các chỉ số CiteScore, Impact Factor, percentile và quartile có thể thay đổi theo từng năm.

Nguồn tham khảo cho mục này:

- Elsevier/ScienceDirect, trang tạp chí *European Journal of Operational Research*: https://www.sciencedirect.com/journal/european-journal-of-operational-research
- Elsevier/ScienceDirect, Journal Insights của EJOR: https://www.sciencedirect.com/journal/european-journal-of-operational-research/about/insights
- SCImago Journal & Country Rank, *European Journal of Operational Research*: https://www.scimagojr.com/journalsearch.php?q=24201&tip=sid
- DOI của paper gốc: https://doi.org/10.1016/j.ejor.2021.03.006
- Bản paper lưu tại LIRIAS, KU Leuven: https://lirias.kuleuven.be/retrieve/632937

## Mục tiêu nghiên cứu

- Tái lập một phần pipeline đánh giá credit scoring trên các bộ dữ liệu công khai.
- Tái kiểm chứng vai trò của XGBoost và ảnh hưởng của độ sâu trong các cấu hình MLP.
- So sánh XGBoost với các mô hình tabular hiện đại gồm CatBoost, TabNet và FT-Transformer như một phần mở rộng nghiên cứu.
- Đánh giá mô hình không chỉ theo hiệu quả dự báo, mà còn theo calibration, chi phí tài chính, khả năng giải thích và chi phí tính toán.

## Phạm vi tái lập

Đây là **partial replication** trên phần dữ liệu công khai có thể truy cập và kiểm chứng trong repository này, không phải tái lập đầy đủ toàn bộ thiết kế thực nghiệm của paper gốc.

Dự án gồm hai thành phần:

- **Replication component**: tái chạy các baseline chính và các cấu hình MLP trên những bộ dữ liệu công khai có thể kiểm chứng.
- **Modern reassessment component**: bổ sung CatBoost, TabNet và FT-Transformer để đánh giá lại kết luận của paper bằng các mô hình tabular hiện đại hơn.

DBN là mô hình có trong nghiên cứu gốc, nhưng không được tái lập trong dự án hiện tại. Lý do là DBN là kiến trúc học sâu mang tính lịch sử, không còn đại diện tốt cho deep learning hiện đại trên dữ liệu bảng; hệ sinh thái thư viện và khả năng tái lập hiện nay hạn chế hơn; chi phí triển khai và tuning lớn; đồng thời DBN có hiệu năng thấp trong nghiên cứu gốc. Việc không tái lập DBN là một giới hạn của partial replication này.

## Câu hỏi nghiên cứu

- Kết quả chính của paper gốc có tái xuất hiện trên các bộ dữ liệu công khai hay không?
- MLP nhiều lớp ẩn có tiếp tục không tốt hơn MLP một lớp ẩn hay không?
- CatBoost, TabNet hoặc FT-Transformer có làm thay đổi kết luận rằng XGBoost nên được ưu tiên cho credit scoring hay không?

## Mô hình dự kiến

### 1. Baseline từ nghiên cứu gốc

- Logistic Regression: baseline truyền thống, dễ giải thích.
- Decision Tree: baseline cây đơn.
- Random Forest: ensemble theo hướng bagging.
- XGBoost: mô hình tốt nhất trong nghiên cứu gốc và là benchmark bắt buộc.

### 2. Mạng neural dùng để tái kiểm chứng ảnh hưởng của độ sâu

- MLP-1: MLP với 1 lớp ẩn.
- MLP-3: MLP với 3 lớp ẩn.
- MLP-5: MLP với 5 lớp ẩn.

### 3. Mô hình hiện đại dùng để mở rộng nghiên cứu

- CatBoost: gradient boosting hiện đại, hỗ trợ tốt biến phân loại.
- TabNet: deep learning chuyên biệt cho dữ liệu bảng.
- FT-Transformer: kiến trúc Transformer dành cho dữ liệu bảng.

## Tiêu chí đánh giá

- ROC-AUC
- PR-AUC
- F1
- Recall của lớp default
- Brier Score
- Calibration
- Chi phí tài chính
- Khả năng giải thích
- Thời gian huấn luyện và suy luận

## Datasets

| Mã | Dataset | Số mẫu | Số biến đầu vào | Default rate |
|---|---|---:|---:|---:|
| AC | Australian Credit Approval | 690 | 14 | 44,5% |
| GC | German Credit Data | 1.000 | 20 | 30,0% |
| TH02 | Thomas et al. 2002 | 1.225 | 14 | 26,4% |
| HMEQ | Home Equity Loan | 5.960 | 12 | 19,9% |
| TC | Default of Credit Card Clients | 30.000 | 23 | 22,1% |
| GMC | Give Me Some Credit | 150.000 | 10 | 6,7% |

## Cấu trúc thư mục hiện tại

```text
.
├── data/
│   ├── checksums-sha256.csv
│   ├── processed/
│   └── raw/
├── paper/
├── results/
└── scripts/
```

## Quy ước dữ liệu

- `1 = bad/default`
- `0 = good/non-default`
- Không chỉnh sửa trực tiếp file trong `data/raw/`.
- Dữ liệu sau xử lý phải ghi vào `data/processed/`.

## Chính sách Git

- Không commit raw data.
- Không commit processed data.
- Không commit PDF của paper.
- Không commit Kaggle token hoặc credential.
- Chỉ commit source code, tài liệu, checksum và script tải hoặc kiểm tra dữ liệu.

## Kiểm tra checksum

Chạy lệnh PowerShell sau để tạo lại checksum cho các file trong `data/raw/`:

```powershell
Get-ChildItem "data\raw" -Recurse -File | Get-FileHash -Algorithm SHA256 | Select-Object Path, Algorithm, Hash | Export-Csv "data\checksums-sha256.csv" -NoTypeInformation -Encoding UTF8
```

## Trạng thái hiện tại

- Đã tải paper.
- Đã tải đủ 6 dataset công khai.
- Đã tạo checksum tại `data/checksums-sha256.csv`.
- Bước tiếp theo là xây dựng `scripts/audit_datasets.py`.
