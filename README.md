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
│   ├── datasets.yaml
│   ├── processed/
│   └── raw/
├── docs/
│   └── data-cards/
├── paper/
├── results/
├── scripts/
├── tests/
└── website/
```

## Quy ước dữ liệu

- `1 = bad/default`
- `0 = good/non-default`
- Không chỉnh sửa trực tiếp file trong `data/raw/`.
- Dữ liệu sau xử lý phải ghi vào `data/processed/`.

## Yêu cầu môi trường

- Python: đã kiểm tra với Python `3.11.0`.
- Runtime dependencies: `requirements.txt`.
- Test/development dependencies: `requirements-dev.txt`.

Thiết lập môi trường sạch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements-dev.txt
```

## Website local P1A

Website P1A nằm trong `website/`, dùng Next.js với TypeScript theo hướng static-first. Website giới thiệu đề tài, paper gốc, phạm vi dữ liệu công khai, câu hỏi nghiên cứu, phương pháp dự kiến, tiến độ phase và kết quả placeholder.

P1A không triển khai Docker, CI/CD, backend API, database, authentication, Google Cloud VM, domain hoặc HTTPS. Website không công bố kết quả thực nghiệm và không chứa raw data, processed data, secret hoặc path nội bộ không cần thiết.

Cài dependencies website:

```powershell
cd website
npm install
```

Chạy local:

```powershell
npm run dev
```

Kiểm tra website:

```powershell
npm run lint
npm run type-check
npm run validate:content
npm run build
```

Cấu trúc nội dung chính:

- `website/content/project.yaml`: thông tin đề tài, mục tiêu và câu hỏi nghiên cứu.
- Navigation công khai hiện gồm 6 mục: Trang chủ, Giới thiệu, Dữ liệu, Phương pháp, Tiến độ và Kết quả.
- Menu và route `Tái lập` không còn nằm trong website công khai P1A. Reproducibility vẫn là yêu cầu nội bộ của pipeline thực nghiệm và thuộc Phase 11, không bị loại khỏi kế hoạch nghiên cứu.
- `website/content/internship.yaml`: thông tin giới thiệu đề tài thực tập; trường chưa có giữ `Chưa cập nhật`.
- `website/content/paper.yaml`: thông tin paper gốc, thiết kế thực nghiệm và kết quả chính do paper báo cáo.
- `website/content/background.md`: bối cảnh và mục đích của đề tài thực tập.
- `website/content/limitations.md`: hạn chế của paper và hạn chế khi tái thực hiện.
- `website/content/progress.yaml`: trạng thái phase; Phase 0 là `Completed`, Phase 1 là `In progress`, Phase 2 đến Phase 11 là `Planned`.
- `website/content/methods.md`: phương pháp dự kiến.
- `website/content/deviations.md`: deviation và giới hạn hiện tại.
- `data/datasets.yaml`: source of truth cho metadata dataset hiển thị trên website.

### Docker và CI P1B

P1B Docker hóa website và bổ sung CI kiểm tra website, Docker image và các test Phase 0 phù hợp. P1B không push image, không deploy VM, không cấu hình GHCR, Nginx production, domain hoặc HTTPS.

Yêu cầu:

- Node.js/npm: CI dùng Node `24`.
- Docker: dùng để build và smoke test image local.
- Python: CI dùng Python `3.11` cho Phase 0 tests không cần raw data.

Build metadata:

- Website đọc `NEXT_PUBLIC_BUILD_SHA` tại build time.
- Nếu không truyền giá trị, website hiển thị `local`.
- Docker build nhận `BUILD_SHA` và truyền vào `NEXT_PUBLIC_BUILD_SHA`.
- Build version xem được ở footer và route `/version/`.

Đồng bộ derivative metadata dataset công khai từ `data/datasets.yaml`:

```powershell
cd website
npm run sync:datasets
```

Build Docker image:

```powershell
cd ..
docker build --build-arg BUILD_SHA=$(git rev-parse --short HEAD) -t credit-scoring-replication-website:local website
```

Chạy container local:

```powershell
docker run --rm -p 8080:8080 credit-scoring-replication-website:local
```

Kiểm tra health và version:

```powershell
curl http://127.0.0.1:8080/health/
curl http://127.0.0.1:8080/version/
```

CI hiện kiểm tra:

- `npm ci`;
- sync public dataset metadata;
- lint;
- TypeScript type check;
- content validation;
- security scan;
- production build;
- Phase 0 Python tests không cần raw data: `python -m pytest tests/test_verify_credit_datasets.py -m "not raw_data"`;
- Docker image build;
- container smoke test;
- health/version response;
- kiểm tra image không chứa raw data, processed data, `.env`, secret hoặc private key.

GitHub Actions workflow chỉ chạy trên `pull_request` và `push` vào `main`. P1B không dùng production secret, không login registry và không deploy.

Trạng thái nội bộ:

- P1A: Completed.
- P1B: Completed khi các acceptance checks ở trên pass.
- P1C: Planned.

Phần còn lại cho P1C:

- P1C: CI/CD, Google Cloud VM, health check, rollback và deployment runbook.

### Production deployment P1C

P1C bổ sung luồng continuous deployment cho website production:

```text
GitHub Actions
-> build Docker image
-> tag bằng full Git commit SHA
-> push lên GHCR
-> SSH vào Google Cloud VM
-> docker pull đúng image SHA
-> docker compose up -d
-> kiểm tra /, /health/ và /version/
-> rollback tự động nếu deploy lỗi
```

Các file chính:

- `.github/workflows/deploy-production.yml`: build, push GHCR và deploy/rollback qua SSH; không chạy trên Pull Request.
- `docker-compose.prod.yml`: chạy website image qua biến `WEBSITE_IMAGE`, map host `80` sang container `8080`.
- `scripts/deploy-production.sh`: deploy immutable image, ghi state/log, kiểm tra health/version và rollback khi lỗi.
- `scripts/rollback-production.sh`: rollback thủ công về previous successful image.
- `docs/DEPLOYMENT_GOOGLE_CLOUD.md`: runbook production, GHCR authentication, secrets, logs, retention và acceptance checklist.

Image production có dạng:

```text
ghcr.io/<owner>/<repository>-website:<full-sha>
```

GitHub Secrets production đang dùng:

- `PROD_HOST`
- `PROD_USER`
- `PROD_SSH_PORT`
- `PROD_SSH_PRIVATE_KEY`
- `PROD_KNOWN_HOSTS`

Manual deploy dùng workflow `Deploy Production` với `action=deploy`. Manual rollback dùng cùng workflow với `action=rollback`.

P1C chưa được đánh dấu hoàn thành chính thức nếu chưa có evidence chạy thật trên GitHub Actions và Google Cloud VM: GHCR push, VM pull đúng image immutable, website public IP trả `/`, `/health/`, `/version/` khớp SHA, rollback thật pass và failed deployment tự rollback được. Domain và HTTPS vẫn Optional trong checkpoint này.

## Metadata dataset

`data/datasets.yaml` là registry chính cho Phase 0. File này khai báo cho 6 dataset công khai:

- file active;
- target mapping;
- numeric/categorical/identifier columns;
- expected shape, class distribution và default rate;
- source/access condition;
- raw preprocessing caveat;
- deviation so với paper.

Data cards chi tiết nằm trong `docs/data-cards/`.

## Download dữ liệu

Raw data đang bị `.gitignore` loại khỏi commit. Sau khi clone repo mới, đặt hoặc tải dữ liệu vào đúng vị trí sau:

| Dataset | File kỳ vọng | Nguồn/điều kiện |
|---|---|---|
| AC | `data/raw/ac/australian.dat`, `data/raw/ac/australian.doc`, `data/raw/ac/Index` | UCI Australian Credit Approval. |
| GC | `data/raw/gc/german.data`, `data/raw/gc/german.doc`, `data/raw/gc/Index` | UCI German Credit Data. |
| HMEQ | `data/raw/hmeq/hmeq_full.csv` | Có thể tải bằng script bên dưới; artifact hiện tại có SHA-256 `DFDBC2B7CDF728A15B53E323CDE6127995715DFA6B178BD3C1E3D9916D0367AA`. |
| TH02 | `data/raw/th02/public.xls`, `data/raw/th02/publicdict.xls` | Supplementary material của *Credit Scoring and Its Applications*; kiểm tra điều kiện truy cập nguồn. |
| TC | `data/raw/tc/default of credit card clients.xls` | UCI Default of Credit Card Clients. |
| GMC | `data/raw/gmc/cs-training.csv`, `data/raw/gmc/Data Dictionary.xls` | Kaggle Give Me Some Credit; có thể cần Kaggle login/credential và chấp nhận điều khoản. |

Tải HMEQ full nếu file chưa có:

```powershell
.\.venv\Scripts\python.exe scripts\download_hmeq.py
```

Script không ghi đè file hiện có nếu không truyền `--force`.

## Prepare dữ liệu

Phase 0 chỉ thực hiện format conversion cần thiết, không làm experimental preprocessing. Không chạy imputation, WOE, VIF, encoding, scaling, class balancing hoặc train/test split.

TH02 raw là Excel BIFF2 legacy. Tạo lại CSV kiểm chứng được bằng:

```powershell
.\.venv\Scripts\python.exe scripts\convert_th02.py --input data/raw/th02/public.xls --output data/processed/th02/th02.csv
```

Conversion giữ nguyên 1.225 dòng, 15 cột, `BAD=0: 902`, `BAD=1: 323` và 23 duplicate rows. Không sửa hoặc ghi đè `data/raw/th02/public.xls`.

## Verify dữ liệu

Verify một dataset:

```powershell
.\.venv\Scripts\python.exe scripts\verify_credit_datasets.py --dataset gc
```

Verify toàn bộ dataset:

```powershell
.\.venv\Scripts\python.exe scripts\verify_credit_datasets.py --dataset all
```

Verify toàn bộ checksum:

```powershell
.\.venv\Scripts\python.exe scripts\verify_credit_datasets.py --dataset all --checksums-only
```

Script tự xác định repository root theo vị trí script, nên có thể chạy từ root hoặc thư mục khác. Kết quả `pass: true` nghĩa là file tồn tại, checksum khớp, schema/target/class count/default rate/metadata feature set đều khớp registry.

## Target mapping

| Dataset | Target | Mapping dùng trong pipeline |
|---|---|---|
| AC | `target` | `0 = good/non-default`, `1 = bad/default`; mapping suy ra từ `australian.doc` và default rate paper. |
| GC | `target` | Raw `1 = Good`, `2 = Bad`; pipeline map `1 -> 0`, `2 -> 1`. |
| HMEQ | `BAD` | `0 = good/non-default`, `1 = bad/default`. |
| TH02 | `BAD` | `0 = good/non-default`, `1 = bad/default`. |
| TC | `default payment next month` | `0 = non-default`, `1 = default`; `ID` không phải input. |
| GMC | `SeriousDlqin2yrs` | `0 = non-default`, `1 = default`; `Unnamed: 0` không phải input. |

Numeric/categorical/identifier metadata đầy đủ nằm trong `data/datasets.yaml` và `docs/data-cards/*.md`.

## Chính sách Git

- Không commit raw data.
- Không commit processed data.
- Không commit PDF của paper.
- Không commit Kaggle token hoặc credential.
- Chỉ commit source code, tài liệu, checksum và script tải hoặc kiểm tra dữ liệu.
- Raw file không được chỉnh sửa trực tiếp.
- Processed artifact phải có script tái tạo; hiện tại `data/processed/th02/th02.csv` được tái tạo bởi `scripts/convert_th02.py`.
- `data/checksums-sha256.csv` dùng đường dẫn tương đối từ repository root, dấu `/`, không dùng drive letter hoặc path tuyệt đối.

## Clean environment verification

Sau khi clone và đặt raw data đúng vị trí:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts\convert_th02.py --input data/raw/th02/public.xls --output data/processed/th02/th02.csv
.\.venv\Scripts\python.exe scripts\verify_credit_datasets.py --dataset all --checksums-only
.\.venv\Scripts\python.exe scripts\verify_credit_datasets.py --dataset all
.\.venv\Scripts\python.exe -m pytest tests/test_verify_credit_datasets.py
```

## Trạng thái hiện tại

- Đã tải paper.
- Đã tải đủ 6 dataset công khai.
- Đã bổ sung HMEQ full tại `data/raw/hmeq/hmeq_full.csv`; file HMEQ 604 dòng cũ được giữ nguyên nhưng không dùng cho core replication.
- Đã chuyển đổi TH02 từ `data/raw/th02/public.xls` sang `data/processed/th02/th02.csv` bằng `scripts/convert_th02.py`.
- Đã chuyển `data/checksums-sha256.csv` sang relative portable paths.
- Đã tạo `data/datasets.yaml` làm registry metadata chính.
- Đã có script kiểm tra dữ liệu `scripts/verify_credit_datasets.py`.
- Đang triển khai P1A website local và nội dung ban đầu.
- Bước tiếp theo sau P1A là Docker hóa/CI-CD theo phạm vi P1B/P1C; chưa chạy model và chưa có pipeline smoke experiment.
