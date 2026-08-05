# Báo cáo audit tính khả thi tái lập và mở rộng thực nghiệm

## Mục lục

1. Phạm vi audit và nguồn bằng chứng
2. Thiết kế thực nghiệm của bài báo gốc
3. Đối chiếu bài báo và repository
4. Audit dữ liệu hiện có
5. Feasibility theo thành phần partial replication
6. Feasibility phần modern reassessment
7. Risk register
8. Đánh giá tài nguyên tính toán
9. Thiết kế thực nghiệm khả thi theo phase
10. Ma trận thực nghiệm tối thiểu
11. Câu hỏi nghiên cứu và khả năng trả lời
12. Deviation so với bài báo gốc
13. Kết luận khả thi

## 1. Phạm vi audit và nguồn bằng chứng

Audit này chỉ đọc, kiểm tra và lập báo cáo. Không triển khai model, không viết training pipeline, không tải thêm dataset, không sửa raw data, không tạo processed dataset, không chạy grid search, không commit và không push.

Ghi chú cập nhật sau P1C: thuật ngữ phase thực nghiệm trong báo cáo feasibility này được lập trước khi roadmap website/CI/CD được chốt. Roadmap hiện hành đã đưa website, Docker, CI và production deployment vào Phase 1; smoke experiment chuyển sang Phase 2. Phase 1 đã Completed: P1A, P1B và P1C đều Completed; production deployment operational tại `http://34.142.206.15`; manual rollback production và automatic failed-deployment rollback đã PASS.

### 1.1. File và thư mục đã kiểm tra

| Hạng mục | Tình trạng repository | Ghi chú |
|---|---|---|
| `AGENTS.md` | Không tồn tại | Không có hướng dẫn agent ở root. |
| `.agents/` | Tồn tại nhưng không có file con | Không có hướng dẫn bổ sung. |
| `.codex/` | Tồn tại nhưng không có file con | Không có hướng dẫn bổ sung. |
| `README.md` | Có | README nói đã tải paper, 6 dataset công khai và checksum. |
| `paper/Gunnarsson_et_al_2021_Deep_Learning_for_Credit_Scoring.pdf` | Có trên đĩa, đang bị Git ignore | Không được commit theo chính sách README. |
| `paper/Gunnarsson_et_al_2021_Deep_Learning_for_Credit_Scoring.md` | Có | Nguồn chính để trích thiết kế thực nghiệm. |
| `paper/Gunnarsson_et_al_2021_Deep_Learning_for_Credit_Scoring_vi.md` | Có | Bản dịch tiếng Việt, dùng để đối chiếu thuật ngữ. |
| `data/checksums-sha256.csv` | Có | Đã chuyển sang relative portable paths từ repository root; verifier từ chối absolute path và `..`. |
| `data/datasets.yaml` | Có | Registry metadata chính cho 6 dataset công khai: target mapping, numeric/categorical/identifier metadata, expected counts và caveat. |
| `data/raw/` | Có 6 thư mục dataset công khai | `ac`, `gc`, `gmc`, `hmeq`, `tc`, `th02`; đều đang bị Git ignore. Sau remediation, HMEQ full nằm tại `data/raw/hmeq/hmeq_full.csv`. |
| `data/processed/` | Có `data/processed/th02/th02.csv` | Artifact chuyển đổi từ TH02 raw BIFF2; không phải dữ liệu tiền xử lý mô hình. |
| `scripts/` | Có script verification/conversion/download | `scripts/verify_credit_datasets.py`, `scripts/convert_th02.py`, `scripts/download_hmeq.py`; chưa có training pipeline. |
| `requirements.txt`, `requirements-dev.txt` | Có | Pin dependencies cho Phase 0 verification và tests. |
| `results/` | Chỉ có `.gitkeep` | Chưa có kết quả thực nghiệm. |

### 1.2. Phân loại bằng chứng

Trong báo cáo này:

- **Fact từ bài báo**: trích từ bản Markdown của paper, đặc biệt Mục 4, Mục 5, Bảng 2, Bảng 3, Bảng 4, Bảng 5, Bảng 6 và các hình 4-7.
- **Fact từ repository**: trích từ cấu trúc file, checksum, README và thống kê metadata raw data trong workspace.
- **Giả định**: điểm cần quyết định khi triển khai vì paper hoặc repo chưa đủ thông tin.
- **Đề xuất của người đánh giá**: protocol, phase, matrix và go/no-go.

## 2. Thiết kế thực nghiệm của bài báo gốc

### 2.1. Dataset trong bài báo

**Fact từ bài báo, Bảng 3 và Mục 4.1.**

| Dataset | Cases | Inputs | Prior default rate | N x 2 CV | Nguồn bài báo mô tả | Công khai/độc quyền |
|---|---:|---:|---:|---:|---|---|
| AC | 690 | 14 | 0.445 | 10 | UCI Australian credit | Công khai |
| GC | 1,000 | 20 | 0.300 | 10 | UCI German credit | Công khai |
| TH02 | 1,225 | 14 | 0.264 | 10 | Thomas et al. (2002) | Công khai hoặc bán công khai theo repo, cần xác minh license |
| Bene1 | 3,123 | 27 | 0.667 | 10 | Major financial institution, Benelux | Độc quyền |
| Bene3 | 3,450 | 8 | 0.016 | 10 | Major financial institution, Benelux | Độc quyền |
| HMEQ | 5,960 | 12 | 0.199 | 5 | Baesens et al. (2016) | Công khai theo README/repo, cần xác minh phiên bản |
| Bene2 | 7,190 | 26 | 0.300 | 5 | Major financial institution, Benelux | Độc quyền |
| UK | 30,000 | 14 | 0.040 | 5 | Major financial institution, UK | Độc quyền |
| TC | 30,000 | 23 | 0.221 | 5 | UCI Default of Credit Card Clients | Công khai |
| GMC | 150,000 | 10 | 0.067 | 5 | Kaggle Give Me Some Credit | Công khai, điều kiện sử dụng Kaggle cần tuân thủ |

Điểm đủ để tái lập: số quan sát, số biến đầu vào, tỷ lệ default và số vòng outer CV được nêu rõ trong Bảng 3; nguồn tổng quát được nêu trong Mục 4.1.

Điểm thiếu hoặc mơ hồ: paper không cung cấp file raw cụ thể, checksum, schema chi tiết, tên target, mapping nhãn từng dataset, phiên bản chính xác của file tải, seed, fold split gốc, license chi tiết và code gốc.

### 2.2. Mô hình gốc

**Fact từ bài báo, Mục 3, Mục 4.1 và Bảng 2.**

| Nhóm | Model | Grid/tuning trong paper | Độ đủ để tái lập |
|---|---|---|---|
| Conventional | Logistic Regression | Không grid, 1 model | Khá đủ, nhưng thiếu implementation/library cụ thể. |
| Conventional | Decision Tree C4.5 | 36 cấu hình; pruning confidence threshold, minimum leaf size | Không đủ chính xác trong scikit-learn vì scikit-learn không có C4.5/pruning tương đương trực tiếp. |
| Ensemble | Random Forest | 30 cấu hình; số cây CART và số input sampled | Khả thi, cần diễn giải công thức `sqrt(m)[...]` và thư viện. |
| Ensemble | XGBoost | 108 cấu hình; số cây, depth, learning rate, input fraction, row fraction | Khả thi, cần cố định version và mapping tham số XGBoost hiện đại. |
| Neural | MLP-1 | 144 cấu hình | Khả thi nhưng chi phí cao; thiếu seed, batch size/epoch/early stopping cụ thể nếu không có trong code gốc. |
| Neural | MLP-3 | 720 cấu hình | Khả thi nhưng không thể tái lập chính xác nếu thiếu implementation detail. |
| Neural | MLP-5 | 2,016 cấu hình | Khả thi nhưng rất tốn compute. |
| Neural | DBN-1 | 324 cấu hình | Không nằm trong partial replication hiện tại; hệ sinh thái hiện đại kém chuẩn. |
| Neural | DBN-3 | 1,620 cấu hình | Không nên triển khai ở phase core. |
| Neural | DBN-5 | 4,536 cấu hình | Không nên triển khai ở phase core. |

### 2.3. Tiền xử lý

**Fact từ bài báo, Mục 4.1-4.2.**

- Missing value: mean replacement cho numeric input, mode replacement cho nominal input.
- Nominal input: thay toàn bộ giá trị bằng log good:bad odds, tức Weight of Evidence.
- Feature reduction: dùng variance inflation factor để xử lý multicollinearity; `VIF <= 10` được xem là chấp nhận được.
- Class imbalance: paper không dùng class balancing.
- Cross-validation: outer `N x 2` CV, với N theo Bảng 3; inner five-fold CV trong từng vòng outer để grid search.

Điểm thiếu hoặc mơ hồ:

- Paper không nói rõ WOE và VIF được fit ở cấp nào trong nested CV. Để tránh leakage, các bước này phải fit chỉ trên training fold của từng inner/outer split.
- Không có fold split gốc, seed, tie-breaking rule, cách xử lý category hiếm/unknown, smoothing WOE, hoặc chuẩn hóa numeric nếu có.
- Cách triển khai VIF sau WOE và imputation cần quy định lại trong pipeline.

### 2.4. Metric

**Fact từ bài báo, Mục 4.3.**

| Metric | Mô tả paper | Độ đủ để tái lập |
|---|---|---|
| AUC | Đánh giá discriminative ability trên toàn score distribution | Đủ. |
| Brier Score | Mean-squared error giữa PD dự báo và binary response | Đủ. |
| Partial Gini | Tập trung vùng score có `p(+1\|x) <= b`, với `b = 0.4` | Cần xác định công thức tính partial Gini cụ thể. |
| EMP | Expected Maximum Profit theo Verbraken et al. (2014) | Có công thức tổng quát nhưng thiếu tham số cost/benefit cụ thể trong repo; chỉ có thể approximate nếu không bổ sung implementation/parameter. |

### 2.5. Kiểm định thống kê

**Fact từ bài báo, Mục 4.4 và Mục 5.1-5.2.**

| Kiểm định | Paper sử dụng | Độ đủ để tái lập |
|---|---|---|
| Friedman test | So sánh average ranks, Bảng 4 | Khả thi. |
| Rom procedure | Post-hoc so với best classifier, adjusted p-value trong Bảng 4 | Khả thi nhưng cần implementation riêng hoặc thư viện phù hợp. |
| Nemenyi multiple comparison | So sánh pairwise neural networks, Hình 4 | Khả thi. |
| Bayesian signed-rank test | So sánh classifier bằng posterior distribution, Bảng 5-6 | Khả thi nhưng cần thư viện/implementation rõ. |
| ROPE | AUC/Partial Gini: 0.01; Brier: 0.0025; EMP: 0.001 | Đủ mức tham số chính. |
| Posterior probability/odds | Bảng 5, ví dụ posterior odds XGBoost vs LR | Khả thi nếu lưu metric fold-level đầy đủ. |

## 3. Đối chiếu bài báo và repository

| Chủ đề | Bài báo gốc | Repository hiện tại | Đánh giá |
|---|---|---|---|
| Số dataset | 10 dataset | Có file raw cho 6 dataset công khai; không có Bene1/Bene2/Bene3/UK | Partial replication, không full replication. |
| Raw data | Không phát hành trong paper | Có raw data ignored: AC, GC, HMEQ, TC, GMC, TH02 | Cần data card và license. |
| Checksum | Không cung cấp | `data/checksums-sha256.csv` có 19 checksum; kiểm tra lại khớp cho file đã đọc/tính | Tốt nhưng cần bổ sung quy trình verify. |
| Code gốc | Không có trong repo | `scripts/` rỗng | Chưa triển khai được ngay. |
| Processed data | Không áp dụng | `data/processed/` rỗng | Đúng phạm vi hiện tại. |
| Results | Paper có Bảng 4-6 và hình | `results/` rỗng | Chưa có thực nghiệm local. |
| README | Không áp dụng | README nói đã tải đủ 6 dataset công khai | Phần lớn đúng, nhưng HMEQ/TH02 cần kiểm chứng phiên bản. |
| PDF paper | Paper yêu cầu đọc PDF | PDF có trên đĩa nhưng ignored; bản Markdown tracked | Có thể audit từ Markdown, PDF không commit. |

## 4. Audit dữ liệu hiện có

### 4.1. Dataset tìm thấy trong `data/raw/`

**Fact từ repository.** Tìm thấy 6 thư mục dataset công khai: `ac`, `gc`, `gmc`, `hmeq`, `tc`, `th02`. Không tìm thấy `Bene1`, `Bene2`, `Bene3`, `UK`.

### 4.2. Bảng metadata dataset

Không in dữ liệu cá nhân hoặc toàn bộ nội dung dataset. Các số dưới đây là metadata/schema và thống kê aggregate.

| Dataset | File chính | Nguồn tải | Checksum | Format | Dung lượng | Dòng | Cột | Target | Nhãn tốt/xấu | Numeric | Categorical | Missing | Default | ID | Duplicate | License/điều kiện | Khớp paper |
|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|---|---|
| AC | `data/raw/ac/australian.dat` | UCI Australian Credit Approval | Khớp | `dat` | 28,735 B | 690 | 15 | cột cuối | README: `1=bad/default`, `0=good/non-default`; doc gốc: `+` 307, `-` 383 | Theo file đọc tự động: 14 numeric; theo `australian.doc`: 6 numeric, 8 categorical | Theo doc: 8 | 0.000 sau bản đã impute | 0.444928 | Không thấy | 0 | Chưa xác định | Số dòng và default khớp; categorical đã mã hóa số nên cần dùng dictionary để tránh hiểu nhầm. |
| GC | `data/raw/gc/german.data` | UCI German Credit Data | Khớp | `data` | 79,793 B | 1,000 | 21 | cột cuối | `german.doc`: `1=Good`, `2=Bad`; cần map `2 -> 1 default`, `1 -> 0` | 7 | 13 | 0.000 | 0.300 nếu `2=bad` | Không thấy | 0 | Chưa xác định | Số dòng, số input và default khớp paper. |
| HMEQ old | `data/raw/hmeq/hmeq.csv` | Chưa xác định; cùng schema HMEQ | Khớp | `csv` | 39,890 B | 604 | 13 | `BAD` | `BAD=1 default/bad`, `0=good` | 10 | 2 | 0.120988 | 0.379139 | Không thấy | 0 | Chưa xác định | Không dùng cho core replication: đây là prefix/truncated version gần như 604 dòng đầu của HMEQ full, dòng cuối thiếu `DEBTINC` so với bản full. |
| HMEQ full | `data/raw/hmeq/hmeq_full.csv` | GitHub raw CSV khớp schema HMEQ; SAS docs xác nhận `Sampsio.Hmeq` 5,960 observations | Khớp checksum local, không khớp checksum SAS kỳ vọng trong yêu cầu | `csv` | 403,157 B | 5,960 | 13 | `BAD` | `BAD=1 default/bad`, `0=good` | 10 | 2 | 0.067971 | 0.1994966443 | Không thấy | 0 | Chưa xác định | Khớp paper về cases, inputs và default rate; usable cho core replication với ghi chú nguồn/checksum. |
| TC | `data/raw/tc/default of credit card clients.xls` | UCI Default of Credit Card Clients | Khớp | `xls` | 5,539,328 B | 30,000 | 25 gồm ID/target | `Y / default payment next month` | `1=default`, `0=non-default` | 24 nếu tính ID; 23 input nếu loại ID | 0 ở mức type đọc; một số biến là categorical mã số | 0.000 | 0.2212 | `ID` | 0 | Chưa xác định | Khớp paper nếu loại ID: 30,000 cases, 23 inputs, default 0.221. |
| GMC | `data/raw/gmc/cs-training.csv` | Kaggle Give Me Some Credit | Khớp | `csv` | 7,564,965 B | 150,000 | 12 gồm index/target | `SeriousDlqin2yrs` | `1=default/delinquency`, `0=non-default` | 10 input nếu loại index và target | 0 | 0.018697 | 0.06684 | `Unnamed: 0` | 0 | Điều kiện Kaggle cần tuân thủ | Khớp paper: 150,000 cases, 10 inputs, default 0.067. |
| TH02 | `data/raw/th02/public.xls`; converted `data/processed/th02/th02.csv` | Thomas et al. (2002) | Khớp raw và processed checksum local | BIFF2 `xls` raw, `csv` processed | 261,129 B raw | 1,225 | 15 | `BAD` | `BAD=1 default/bad`, `0=good` | 12 | 2 | 0.000000 | 0.2636734694 | Không thấy generated index | 23 | Chưa xác định | Usable sau conversion/read step bằng `xlrd`; duplicate giữ nguyên để bảo toàn 1,225 observations. |

### 4.3. Checksum

`data/checksums-sha256.csv` có checksum cho các file raw sau: AC, GC, GMC, HMEQ, TC, TH02. Sau remediation, file này cũng ghi `data/raw/hmeq/hmeq_full.csv` và `data/processed/th02/th02.csv`.

Ghi chú provenance TH02 ngày 2026-07-23: chỉ checksum của file data dictionary phụ trợ `data/raw/th02/publicdict.xls` được re-baseline từ checksum cũ `639DBECD784A62D4BFEF16154FEB71BDE71E11D7DEE8626E23CAA76B31B0A62F` sang artifact hiện đang được phân phối từ cùng nguồn công khai, SHA-256 `DDCF7ABB426521DCCB51A9A081D81890632BAAA2CEC0F5253863751545B5A7C5`. Artifact mang checksum cũ không còn khả dụng. `publicdict.xls` không được `scripts/convert_th02.py` sử dụng để tạo processed dataset; `data/processed/th02/th02.csv` vẫn giữ SHA-256 `C9CB96394DDE17CF82C3ABC5D8FFE465DC1F68CE442D680E868A7B9224AD59B7`. Shape, schema, target mapping, class counts và default rate của TH02 vẫn pass; thay đổi này không ảnh hưởng dữ liệu active, target mapping hoặc schema. Sau cập nhật, toàn bộ verifier và pytest đã pass.

### 4.4. Trả lời các câu hỏi dữ liệu bắt buộc

1. Trong 10 dataset của bài báo, repo hiện có file cho 6 dataset công khai: AC, GC, TH02, HMEQ, TC, GMC.
2. Bene1, Bene2, Bene3 và UK không có trong repo và theo bài báo là dữ liệu từ financial institutions, không tiếp cận công khai.
3. AC, GC, HMEQ full, TH02, TC và GMC khớp các kích thước/tỷ lệ chính của paper ở mức Phase 0 data verification. HMEQ 604 dòng không phải bản dùng cho core replication.
4. HMEQ 604 dòng là prefix/truncated version của HMEQ full: 603/604 dòng đầu khớp hoàn toàn với 604 observation đầu của file full; dòng cuối chỉ lệch `DEBTINC` do file cũ thiếu giá trị này.
5. Mapping nhãn đủ rõ cho GC, HMEQ, TH02, TC, GMC; AC cần quyết định nhất quán giữa README và `australian.doc`.
6. Checksum hiện tại đầy đủ cho file raw đang có và hợp lệ ở mức kiểm tra lại hash.
7. Rủi ro leakage chính nằm ở WOE/VIF nếu fit toàn dataset, loại ID/index không đúng, dùng HMEQ sai phiên bản, và chọn threshold/cost trên test fold.

## 5. Feasibility theo thành phần partial replication

### 5.1. Feasibility theo dataset

| Dataset | Tình trạng | Phân loại feasibility | Lý do |
|---|---|---|---|
| AC | Có, khớp row/default, categorical đã mã hóa số | Khả thi nhưng cần giả định | Cần áp dictionary để phân loại numeric/categorical đúng và giữ mapping nhãn. |
| GC | Có, khớp paper | Khả thi trực tiếp | Có `german.doc`, target mapping rõ. |
| TH02 | Raw BIFF2 đọc được bằng `xlrd`; có CSV converted | Khả thi trực tiếp | `public.xls` pass 1,225 x 15, `BAD=0`: 902, `BAD=1`: 323; conversion script không sửa raw. |
| Bene1 | Không có | Không khả thi với dữ liệu hiện có | Dữ liệu độc quyền. |
| Bene2 | Không có | Không khả thi với dữ liệu hiện có | Dữ liệu độc quyền. |
| Bene3 | Không có | Không khả thi với dữ liệu hiện có | Dữ liệu độc quyền. |
| HMEQ | Có bản full 5,960 dòng và bản cũ 604 dòng | Khả thi trực tiếp với ghi chú nguồn/checksum | `hmeq_full.csv` pass schema/class counts của paper; file 604 dòng giữ lại nhưng không dùng cho core. |
| UK | Không có | Không khả thi với dữ liệu hiện có | Dữ liệu độc quyền. |
| TC | Có, khớp paper sau loại ID | Khả thi trực tiếp | 30,000 dòng, default 0.2212. |
| GMC | Có, khớp paper sau loại index | Khả thi trực tiếp | 150,000 dòng, default 0.06684. |

### 5.2. Feasibility theo mô hình baseline

| Thành phần | Phân loại | Lý do và bằng chứng |
|---|---|---|
| Logistic Regression | Khả thi trực tiếp | Paper dùng 1 model không grid; scikit-learn có implementation ổn định. Cần thống nhất solver/regularization. |
| Decision Tree C4.5 | Khả thi nhưng không thể tái lập chính xác | Paper dùng C4.5 với pruning confidence; scikit-learn CART không tương đương. Có thể dùng approximation hoặc thư viện C4.5 riêng và ghi deviation. |
| Random Forest | Khả thi trực tiếp | Grid rõ trong Bảng 2; cần map tham số hiện đại và seed. |
| XGBoost | Khả thi trực tiếp | Grid rõ trong Bảng 2; cần pin version và xác định objective/eval metric. |

### 5.3. Feasibility MLP depth comparison

| Thành phần | Phân loại | Lý do và bằng chứng |
|---|---|---|
| MLP-1 | Khả thi nhưng cần giả định | Grid chính có trong Bảng 2, nhưng thiếu epoch, batch size, initialization, early stopping, seed. |
| MLP-3 | Khả thi nhưng cần giả định | Grid lớn hơn, rủi ro phương sai seed cao. |
| MLP-5 | Khả thi nhưng không thể tái lập chính xác | 2,016 cấu hình mỗi inner loop là rất tốn kém; cần giảm budget hoặc fixed budget. |

### 5.4. Feasibility preprocessing

| Bước | Phân loại | Lý do |
|---|---|---|
| Mean/mode imputation | Khả thi trực tiếp | Có thể fit trong training fold bằng `Pipeline`. |
| WOE | Khả thi nhưng cần giả định | Cần smoothing, handling category unseen và fit trong fold để tránh leakage. |
| VIF | Khả thi nhưng cần giả định | Cần fit chỉ trên training fold; phải xác định thứ tự so với WOE/imputation. |
| Handling categorical features | Khả thi nhưng cần giả định | AC/TC có categorical mã số; cần dictionary để không xem nhầm là continuous. |
| Không class balancing | Khả thi trực tiếp | Paper nêu rõ không dùng balancing. |
| Nested CV | Khả thi nhưng chi phí cao | Protocol rõ, nhưng full grid không thực tế trên CPU-only. |

### 5.5. Feasibility metric và thống kê

| Thành phần | Phân loại | Lý do |
|---|---|---|
| AUC | Khả thi trực tiếp | `predict_proba` đủ. |
| Brier Score | Khả thi trực tiếp | Cần xác suất calibrated/raw nhất quán. |
| Partial Gini | Khả thi nhưng cần giả định | Cần chọn công thức exact theo Lessmann et al. và threshold `b=0.4`. |
| EMP | Khả thi nhưng không thể tái lập chính xác hiện tại | Paper đưa công thức nhưng repo chưa có tham số cost/benefit và implementation. |
| Friedman | Khả thi trực tiếp | Có thư viện thống kê phổ biến. |
| Rom | Khả thi nhưng cần implementation riêng | Không phải API mặc định trong mọi stack Python. |
| Nemenyi | Khả thi nhưng cần thư viện/implementation | Có thể dùng `scikit-posthocs` hoặc tự triển khai. |
| Bayesian signed-rank | Khả thi nhưng cần implementation riêng | Cần lưu fold-level difference và ROPE. |
| ROPE | Khả thi trực tiếp | Paper nêu ngưỡng. |
| Posterior odds | Khả thi nếu có posterior probabilities | Cần kiểm định Bayes chạy đúng. |

## 6. Feasibility phần modern reassessment

### 6.1. Model hiện đại

| Model | Phù hợp tabular credit scoring | Dữ liệu nhỏ/vừa/lớn | CPU/RAM/GPU | Overfitting | Seed sensitivity | Preprocessing/categorical | Probability output | Nested CV | Công bằng với WOE baseline |
|---|---|---|---|---|---|---|---|---|---|
| CatBoost | Rất phù hợp cho dữ liệu bảng, đặc biệt categorical | Tốt trên nhỏ/vừa/lớn nếu tuning hợp lý | CPU chạy được; GPU hữu ích cho GMC/grid lớn | Trung bình, kiểm soát bằng depth/iterations/l2/early stopping | Trung bình | Có native categorical; ít cần WOE | Có `predict_proba` | Khả thi | Protocol native có lợi thế riêng; common preprocessing công bằng hơn nhưng làm mất điểm mạnh CatBoost. |
| TabNet | Phù hợp nhưng thường cần tuning và dữ liệu đủ lớn | Rủi ro trên AC/GC/TH02; hợp hơn TC/GMC | CPU chậm; GPU nên có | Cao trên dataset nhỏ | Cao | Cần encoding/embedding, scaling tùy implementation | Có xác suất | Khả thi nhưng tốn compute | Không nên so trực tiếp nếu preprocessing quá khác mà không tách protocol. |
| FT-Transformer | Phù hợp cho tabular hiện đại nhưng nhạy với setup | Rủi ro trên dataset nhỏ; hợp hơn TC/GMC | GPU rất nên có; CPU chậm | Cao trên nhỏ | Cao | Cần pipeline numeric/categorical riêng, embedding categorical | Có xác suất | Khả thi ở budget giới hạn | Nên là reassessment, không thay cho replication core. |

### 6.2. Protocol A và Protocol B

| Protocol | Mô tả | Ưu điểm | Nhược điểm | Khuyến nghị |
|---|---|---|---|---|
| Protocol A - Common preprocessing | Tất cả model dùng dữ liệu sau imputation/encoding/WOE/VIF hoặc pipeline tương đương thống nhất | So sánh fair hơn vì input giống nhau; dễ gắn vào replication gốc | Có thể bất lợi cho CatBoost/Transformer-native; WOE có thể làm mất thông tin categorical | Dùng làm protocol chính cho câu hỏi “so sánh trong điều kiện gần paper”. |
| Protocol B - Model-native preprocessing | CatBoost dùng categorical native; TabNet/FT-Transformer dùng pipeline riêng; baseline dùng pipeline truyền thống | Đánh giá model theo cách dùng thực tế tốt hơn | Không được trộn kết quả với Protocol A như cùng điều kiện; khó quy kết khác biệt do model hay preprocessing | Dùng làm protocol phụ/sensitivity analysis. |

**Đề xuất của người đánh giá:** Protocol A là protocol chính cho partial replication và câu hỏi so sánh với paper; Protocol B là modern reassessment phụ. Kết quả hai protocol phải báo cáo riêng.

## 7. Risk register

| Rủi ro | Mức độ | Ảnh hưởng | Biện pháp giảm thiểu | Residual risk |
|---|---|---|---|---|
| Không có Bene1/Bene2/Bene3/UK | Cao | Không thể full replication, giảm power của statistical comparison | Ghi rõ partial replication, không kết luận quá mức | Cao |
| Dùng nhầm HMEQ 604 dòng thay vì HMEQ full | Trung bình | Kết quả HMEQ lệch mạnh default rate và sample selection | Verifier mặc định trỏ tới `hmeq_full.csv`; data card ghi rõ file 604 dòng fail core validation | Thấp |
| HMEQ full không khớp checksum SAS kỳ vọng | Trung bình | Artifact nguồn chưa chứng minh là đúng byte-for-byte với SAS sample export | Ghi rõ source GitHub raw và checksum local; chỉ đánh dấu usable theo shape/schema/class counts, không tuyên bố khớp SAS artifact | Trung bình |
| TH02 raw là BIFF2 legacy | Thấp | Một số môi trường không đọc được nếu thiếu `xlrd` | Dùng `scripts/convert_th02.py`, fallback LibreOffice nếu cần | Thấp |
| Version thư viện hiện đại khác 2021 | Trung bình | Kết quả khác do implementation drift | Pin version, log environment | Trung bình |
| Không có source code/seed/fold gốc | Cao | Không tái lập bit-level | Lưu seed/fold mới, gọi là replication có điều kiện | Cao |
| C4.5 không tương đương scikit-learn | Cao | Baseline DT lệch | Dùng thư viện C4.5 hoặc ghi CART approximation | Trung bình |
| WOE fit ngoài training fold | Cao | Data leakage, metric inflated | Implement transformer fit trong fold | Thấp nếu kiểm thử kỹ |
| VIF tính toàn dataset | Cao | Feature-selection leakage | Fit VIF trong fold, log selected features | Thấp-trung bình |
| Nested CV + grid lớn | Cao | Không thực tế CPU-only | Reduced grid/fixed budget/Optuna và ghi deviation | Trung bình |
| Grid MLP/DBN rất lớn | Cao | Chi phí tăng mạnh, khó hoàn tất | Không chạy DBN; giảm MLP budget có kiểm soát | Trung bình |
| DBN không nằm trong partial replication | Trung bình | Không kiểm chứng đầy đủ toàn bộ DL gốc | Ghi rõ deviation; chỉ xem DBN optional | Trung bình |
| EMP thiếu tham số/implementation | Cao | Không tính chính xác như paper | Dùng approximate hoặc lấy Verbraken implementation/parameter | Cao |
| Rom/Bayesian signed-rank cần implementation riêng | Trung bình | Statistical phase chậm | Chọn thư viện, viết unit tests trên bảng toy | Trung bình |
| Dataset mất cân bằng nghiêm trọng | Trung bình | Metric/profit nhạy, model bỏ qua minority | Báo cáo PR-AUC phụ, stratified split, không balancing nếu replication | Trung bình |
| Deep learning seed variance | Cao | Kết luận MLP/TabNet/FT không ổn định | Nhiều seed trong robustness | Trung bình |
| Preprocessing khác nhau làm kết quả không so trực tiếp | Cao | Kết luận sai về model | Tách Protocol A/B rõ ràng | Thấp-trung bình |
| Dùng test fold chọn threshold/cost | Cao | Leakage và over-optimism | Threshold/cost chỉ chọn trong inner fold/training | Thấp nếu enforced |
| Kết luận quá mức từ dataset công khai hạn chế | Cao | Overclaim so với paper | Giới hạn diễn giải, report confidence/residual risk | Trung bình |

## 8. Đánh giá tài nguyên tính toán

Không có bằng chứng repo hiện tại có GPU. Vì vậy mọi ước lượng GPU chỉ là kịch bản giả định.

### 8.1. CPU-only

| Hạng mục | Ước lượng |
|---|---|
| Dataset có thể chạy ngay | AC, GC, HMEQ full, TH02, TC, GMC |
| Outer folds | AC/GC/TH02: 20 split mỗi dataset; TC/GMC/HMEQ: 10 split mỗi dataset theo paper |
| Inner folds | 5 fold mỗi outer training split |
| Fit baseline full paper grid trên 4 dataset đủ điều kiện | Rất lớn; XGBoost/RF còn khả thi, MLP-5 rất tốn |
| Model chi phí lớn nhất | MLP-5, TabNet, FT-Transformer; DBN nếu đưa vào sẽ rất nặng |
| Artifact storage | Từ vài GB đến hàng chục GB nếu lưu model/fold prediction đầy đủ; nên lưu prediction/metric/config, không lưu mọi model |
| Thời gian | Smoke: phút đến vài giờ; core reduced grid: nhiều giờ đến vài ngày; full grid MLP: không thực tế |

### 8.2. Có một GPU phổ thông

| Hạng mục | Ước lượng |
|---|---|
| Dataset có thể chạy | Tương tự CPU-only; GPU không giải quyết license/schema |
| Lợi ích chính | Tăng tốc MLP/TabNet/FT-Transformer và có thể XGBoost GPU |
| Rủi ro | Kết quả phụ thuộc driver/CUDA/library; seed determinism khó hơn |
| Thời gian | Smoke vẫn phút-giờ; modern reassessment khả thi hơn nhưng vẫn cần fixed budget |

### 8.3. Khuyến nghị tuning budget

Tái tạo nguyên grid search của paper, đặc biệt MLP/DBN, là không thực tế trong phase đầu. Nên dùng:

- Smoke: fixed small grid.
- Core partial replication: reduced grid có kiểm soát cho RF/XGBoost/MLP.
- Modern reassessment: Optuna hoặc fixed budget per model.
- Mọi thay đổi budget phải ghi là deviation.

## 9. Thiết kế thực nghiệm khả thi theo phase

| Phase | Mục tiêu | Đầu vào | Đầu ra | Điều kiện bắt đầu | Acceptance criteria | Rủi ro chính | Go/no-go |
|---|---|---|---|---|---|---|---|
| Phase 0 - Data verification | Xác minh checksum, schema, target, class balance, data card | Raw data và checksum | Data card từng dataset | Không cần model | AC/GC/HMEQ/TH02/TC/GMC pass | Dùng nhầm artifact HMEQ cũ | Go; HMEQ và TH02 đã pass remediation |
| Phase 2 - Smoke experiment | Xác minh pipeline end-to-end | Một dataset nhỏ và một vừa | Metric/log/artifact tối thiểu | Phase 1 Completed | LR và XGBoost chạy nested-lite, metric hợp lệ | Leakage, target mapping sai | Go nếu fold isolation pass |
| Phase 7 - Core partial replication | Chạy LR, DT, RF, XGB, MLP-1/3/5 trên dataset đủ điều kiện | Dataset pass Phase 0 và pipeline Phase 2-6 | Fold-level predictions/metrics | Smoke pass, preprocessing/metric/model foundation pass | AUC/Brier/Partial Gini tính ổn định; EMP nếu đủ | Compute và C4.5 approximation | Go nếu budget và metric OK |
| Phase 8 - Modern reassessment | Chạy CatBoost, TabNet, FT-Transformer theo protocol | Core pipeline | Kết quả Protocol A/B tách riêng | Phase 7 có baseline đáng tin | Không trộn protocol; seed logged | Overfitting và GPU/CPU cost | Go cho CatBoost trước, deep tabular sau |
| Phase 9 - Statistical comparison | Friedman, post-hoc, Bayesian signed-rank, ROPE | Fold-level metrics | Statistical tables | Có đủ dataset/fold metric | Reproduce table format tương tự paper | Ít dataset làm power thấp | Go nếu >=4 dataset pass |
| Phase 10 - Robustness checks | Multi-seed, sensitivity preprocessing/protocol | Core + modern results | Robustness appendix | Phase 9 sơ bộ | Kết luận không phụ thuộc một seed/protocol | Compute tăng | Optional/Should tùy tài nguyên |

## 10. Ma trận thực nghiệm tối thiểu

| Dataset | Model | Protocol | Hyperparameter strategy | CV strategy | Seeds | Metrics | Statistical group | Compute | Priority |
|---|---|---|---|---|---:|---|---|---|---|
| GC | LR | A | Fixed/default logged | 10x2 outer, 5-fold inner nếu có tuning | 1 | AUC, Brier, Partial Gini, EMP approximate | Baseline/core | Thấp | Must |
| GC | XGBoost | A | Reduced grid | 10x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini, EMP approximate | Baseline/core | Trung bình | Must |
| TC | LR | A | Fixed/default logged | 5x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini, EMP approximate | Baseline/core | Thấp | Must |
| TC | XGBoost | A | Reduced grid | 5x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini, EMP approximate | Baseline/core | Trung bình | Must |
| AC | LR/RF/XGB/MLP-1/3/5 | A | Reduced grid | 10x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini, EMP approximate | Core partial | Trung bình | Must |
| GC | LR/RF/XGB/MLP-1/3/5 | A | Reduced grid | 10x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini, EMP approximate | Core partial | Trung bình | Must |
| TC | LR/RF/XGB/MLP-1/3/5 | A | Reduced grid | 5x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini, EMP approximate | Core partial | Trung bình-cao | Must |
| GMC | LR/RF/XGB/MLP-1/3/5 | A | Reduced grid/fixed budget | 5x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini, EMP approximate | Core partial | Cao | Must |
| TH02 | Core models | A | Reduced grid | 10x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini, EMP approximate | Core partial | Trung bình | Must |
| HMEQ full | Core models | A | Reduced grid | 5x2 + inner 5-fold | 1 | AUC, Brier, Partial Gini | Core partial | Thấp | Must |
| Public pass datasets | CatBoost | A | Fixed budget/Optuna | Same outer/inner | 1-3 | AUC, Brier, Partial Gini, EMP approximate | Modern A | Trung bình | Should |
| Public pass datasets | CatBoost | B | Native categorical | Same outer/inner | 1-3 | AUC, Brier, Partial Gini, EMP approximate | Modern B | Trung bình | Should |
| TC/GMC | TabNet | A/B tách riêng | Fixed budget | Same outer/inner hoặc nested-lite | 3 | AUC, Brier, Partial Gini | Modern deep | Cao | Optional |
| TC/GMC | FT-Transformer | A/B tách riêng | Fixed budget | Same outer/inner hoặc nested-lite | 3 | AUC, Brier, Partial Gini | Modern deep | Cao | Optional |

## 11. Câu hỏi nghiên cứu và khả năng trả lời

| Câu hỏi | Dữ liệu cần có | Model cần chạy | Metric chính | Kiểm định | Giới hạn diễn giải |
|---|---|---|---|---|---|
| Kết quả chính của paper có tái xuất hiện trên dataset công khai không? | AC, GC, TC, GMC tối thiểu; TH02/HMEQ nếu pass | LR, DT, RF, XGB, MLP-1/3/5 | AUC, Brier, Partial Gini, EMP | Friedman, Rom, Bayesian signed-rank | Không có 4 dataset độc quyền; HMEQ/TH02 chưa chắc khớp. |
| MLP nhiều lớp có tiếp tục không tốt hơn MLP một lớp không? | Dataset public pass, fold-level metrics | MLP-1, MLP-3, MLP-5 | AUC, Brier, Partial Gini, EMP | Nemenyi và Bayesian signed-rank với ROPE | Nhạy seed; nếu giảm grid thì là deviation. |
| CatBoost/TabNet/FT-Transformer có thay đổi kết luận ưu tiên XGBoost không? | Dataset public pass, đặc biệt TC/GMC | XGBoost, CatBoost, TabNet, FT-Transformer | AUC, Brier, Partial Gini, EMP nếu đủ | Friedman/Bayesian trong từng protocol | Không trộn Protocol A và B; deep tabular có thể thiếu compute/seed. |

## 12. Deviation so với bài báo gốc

Các deviation hiện đã biết hoặc rất có khả năng cần chấp nhận:

- Chỉ partial replication vì thiếu Bene1, Bene2, Bene3 và UK.
- HMEQ 604 dòng cũ không dùng cho core; core phải dùng `data/raw/hmeq/hmeq_full.csv`.
- HMEQ full pass shape/schema/class counts nhưng checksum local khác checksum SAS kỳ vọng trong yêu cầu; đây là source-artifact deviation cần ghi khi công bố.
- TH02 cần bước conversion/read bằng `xlrd` hoặc LibreOffice trước khi pipeline tiêu thụ CSV.
- Không chạy DBN trong core partial replication.
- Decision Tree có thể phải dùng CART approximation nếu không có C4.5 tương đương.
- Có thể giảm grid hoặc dùng fixed budget/Optuna thay vì nguyên grid paper.
- EMP có thể chỉ approximate nếu không bổ sung tham số/implementation chính xác theo Verbraken et al. (2014).
- Version thư viện, seed và fold split sẽ khác paper nếu không có code gốc.
- Protocol B model-native cho CatBoost/TabNet/FT-Transformer là modern reassessment, không phải replication trực tiếp.

## 13. Kết luận khả thi

**Cập nhật sau Phase 6:** Phase 6 đã đóng ở boundary infrastructure/hardening: MLP-1/3/5 đã có nested-CV leakage-safe, early stopping deterministic, artifacts, resume/retry và reduced engineering validation trên GC/TC. Các artifact này không phải kết quả khoa học hay bằng chứng ranking; Phase 7 là bước kế tiếp để chạy core replication và tạo fold-level scientific evidence. TC cho thấy CPU local khả thi cho checkpoint bị chặn, nhưng full scope cần profiling/budget trước khi phê duyệt; peak-memory telemetry chưa có nên không được suy diễn thành capacity claim.

**Kết luận cập nhật sau Phase 0:** **Khả thi có điều kiện** cho partial replication trên 6 dataset công khai; vẫn không khả thi cho full replication vì thiếu Bene1, Bene2, Bene3 và UK.

Các blocker Phase 4 về metric validation đã được chốt trong Phase 4, và Phase 5–6 đã hoàn thành infrastructure/hardening. Theo source of truth hiện hành, Phase 0–6 là Completed, Phase 7 là In Progress: P7A đã completed và khóa protocol/manifest candidate; P7B đã completed với P7B.2 CART engineering-feasibility 60/60 fit, 0 failure, artifact validation pass và final CART-A 12-candidate manifest lock. Đây chỉ là evidence kỹ thuật non-publishable, không phải candidate ranking hay scientific result. P7C là checkpoint tiếp theo nhưng chưa chạy; search space/budget của RF, XGBoost, MLP-1/3/5, CatBoost, TabNet và FT-Transformer chưa được khóa bởi quyết định CART. Phase 0 acceptance có registry tập trung, checksum portable, target mapping đủ 6 dataset, data cards, dependency pin, reusable TH02 conversion và verifier chạy độc lập current working directory. HMEQ full và TH02 đã pass verification; không có scientific result nào được suy diễn từ smoke/reduced/dry-run/P7B artifact.

Dataset nên dùng cho smoke test đầu tiên: **GC** cho dataset nhỏ vì schema và mapping nhãn rõ, sau đó **TC** cho dataset vừa/lớn vì khớp paper và có ID cần loại đúng. Model nên triển khai đầu tiên: **Logistic Regression** và **XGBoost**.

Nên giữ **CatBoost** trong modern reassessment. **TabNet** và **FT-Transformer** nên giữ ở mức Optional/Phase 3, chỉ chạy sau khi baseline và CatBoost ổn định, vì chi phí compute và seed variance cao. Không nên chạy lại **DBN** trong scope hiện tại, trừ khi mục tiêu đổi thành full historical replication và có budget riêng.

Nên giữ nguyên `N x 2` CV của paper cho dataset Must nếu compute cho phép, nhưng dùng reduced grid/fixed budget là deviation có kiểm soát. Với EMP, ở trạng thái repo hiện tại chỉ nên xem là **approximate/chưa xác định exact**, vì thiếu tham số cost/benefit và implementation cụ thể. Phạm vi core có thể gồm AC, GC, HMEQ full, TH02, TC và GMC.

### Tóm tắt terminal

- Kết luận feasibility: Khả thi có điều kiện cho partial replication trên 6 dataset công khai; Phase 0 data verification pass với caveat provenance HMEQ.
- Số dataset tìm thấy: 6/10 dataset của paper có file raw trong repo.
- Số dataset đủ điều kiện ngay: 6 dataset công khai sau remediation; HMEQ dùng `hmeq_full.csv`, TH02 dùng conversion artifact.
- Blocker chính: thiếu 4 dataset độc quyền, HMEQ full chưa khớp checksum SAS kỳ vọng, C4.5/EMP/WOE/VIF cần implementation chống leakage.
- Protocol đề xuất: Protocol A làm chính, Protocol B làm sensitivity/modern reassessment phụ.
- Phase tiếp theo: Phase 7 core replication run; Phase 6 đã Completed ở mức MLP infrastructure/hardening và mọi smoke/reduced artifact vẫn non-publishable.
- File báo cáo đã tạo: `docs/EXPERIMENT_FEASIBILITY_ASSESSMENT.md`.
