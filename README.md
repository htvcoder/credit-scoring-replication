# Tái lập và đánh giá lại mô hình tính điểm tín dụng

## Tổng quan

Repository `credit-scoring-replication` chuẩn bị cho đề tài **Tái lập và đánh giá lại mô hình tính điểm tín dụng**. Dự án tái lập có kiểm soát nghiên cứu **Deep Learning for Credit Scoring: Do or Don't?** trên các bộ dữ liệu công khai đã được xác minh, đồng thời đánh giá lại các kết luận của nghiên cứu bằng các mô hình học máy truyền thống, mạng nơ-ron nhiều lớp và một số mô hình tabular hiện đại.

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
├── src/
│   └── creditrep/
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
- Runtime dependencies: khai báo trong `pyproject.toml` và đồng bộ trong `requirements.txt`.
- Test/development dependencies: optional extra `test` trong `pyproject.toml`.

Cài package chỉ để sử dụng runtime:

```powershell
python -m pip install -e .
```

Cài package để phát triển và chạy test:

```powershell
python -m pip install -e ".[test]"
```

`pip install -e .` chỉ cài runtime dependencies. `pip install -e ".[test]"` cài thêm dependency dùng để chạy test, nên không cần chạy `pip install pytest` thủ công sau đó.

Thiết lập môi trường phát triển:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

## Website local P1A

Website nằm trong `website/`, dùng Next.js với TypeScript theo hướng static-first. Website giới thiệu đề tài, paper gốc, phạm vi dữ liệu công khai, câu hỏi nghiên cứu, phương pháp dự kiến, tiến độ phase và kết quả placeholder.

P1A không triển khai Docker, CI/CD, backend API, database, authentication, Google Cloud VM, domain hoặc HTTPS. Website không công bố kết quả thực nghiệm và không chứa raw data, processed data, secret hoặc path nội bộ không cần thiết.

### Nhận diện website

Website dùng phương án logo số 2 làm nhận diện chính thức cho dự án: biểu tượng khiên đen, ba cột biểu đồ tăng dần màu đỏ và dấu kiểm trắng trong vòng tròn đỏ. Bộ asset nằm trong `website/public/brand/`:

- `csr-mark.svg`: biểu tượng khiên dùng cho favicon, header và các vị trí nhỏ.
- `csr-logo-horizontal.svg`: lockup ngang gồm biểu tượng, `CSR` và `CREDIT SCORING REPLICATION`.
- `csr-logo-full.svg`: phiên bản đầy đủ gồm biểu tượng, `CSR`, tên tiếng Anh và tên tiếng Việt `Tái lập và đánh giá lại mô hình tính điểm tín dụng`.
- `csr-icon-192.png`, `csr-icon-512.png`, `apple-touch-icon.png` và `favicon.ico`: app icon và favicon chỉ dùng biểu tượng khiên, không dùng full wordmark để tránh chữ bị mất ở kích thước nhỏ.

Hệ màu thương hiệu của website là đỏ - đen - trắng. Mã màu chính: primary red `#E30613`, dark red `#B80916`, soft red background `#FFF1F2`, main black `#111111`, charcoal `#1F2933`, muted text `#5F6368`, border `#E5E7EB`, main background `#FFFFFF`, secondary background `#FAFAFA`. Khi thay đổi logo trong tương lai, cập nhật các asset trong `website/public/brand/`, giữ favicon là biểu tượng khiên riêng và đồng bộ metadata trong `website/app/layout.tsx` cùng `website/public/site.webmanifest`.

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

- `website/content/project.yaml`: tên đề tài công khai, mô tả ngắn, mục tiêu nghiên cứu, vai trò website và câu hỏi nghiên cứu.
- Navigation công khai hiện gồm 6 mục: Trang chủ, Giới thiệu, Dữ liệu, Phương pháp, Tiến độ và Kết quả.
- Menu và route `Tái lập` không còn nằm trong website công khai P1A. Reproducibility vẫn là yêu cầu nội bộ của pipeline thực nghiệm và thuộc Phase 11, không bị loại khỏi kế hoạch nghiên cứu.
- `website/content/internship.yaml`: thông tin giới thiệu đề tài, mục tiêu dự án, phạm vi công việc và thông tin thực hiện; trường chưa có giữ `Chưa cập nhật`.
- `website/content/paper.yaml`: thông tin paper gốc, thiết kế thực nghiệm và kết quả chính do paper báo cáo.
- `website/content/background.md`: bối cảnh và mục đích của đề tài thực tập.
- `website/content/limitations.md`: hạn chế của paper và hạn chế khi tái thực hiện.
- `website/content/progress.yaml`: source of truth cho trạng thái phase; Phase 0 đến Phase 6 là `Completed`, Phase 7 là `In Progress`, Phase 8 đến Phase 11 là `Planned`. Tag Phase 6 đang là đề xuất, chưa được tạo trong Git.
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
- P1B: Completed; CI đã chạy thành công trên Pull Request và `main`.
- P1C: Production deployment Completed; rollback verification PASS.

Production hiện tại:

- Workflow `Test VM SSH` đã pass, xác nhận GitHub Actions SSH được vào Google Cloud VM và user production dùng được Docker/Docker Compose.
- Workflow production deployment đã chạy tự động thành công từ `main`.
- Docker image website đã được build, push lên GHCR và deploy lên VM bằng Docker Compose.
- Website production truy cập được tại `http://34.142.206.15`.
- Health/version validation trong deployment workflow đã pass vì workflow production báo thành công.
- Website hiện dùng HTTP/public IP; domain và HTTPS vẫn Optional trong phạm vi hiện tại.
- Website chỉ có nội dung công khai, không có đăng nhập hoặc chức năng truyền dữ liệu nhạy cảm.
- Manual rollback production và automatic failed-deployment rollback đã được xác nhận PASS, nên Phase 1 đã Completed.

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

Các evidence đã có: GHCR push, VM pull đúng image immutable, website public IP trả `/`, `/health/`, `/version/` khớp SHA trong workflow deployment thành công; manual rollback production PASS; automatic failed-deployment rollback PASS bằng forced post-deploy failure mode. Domain và HTTPS vẫn Optional trong checkpoint này.

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

## Dataset loader P2A

Phase 2A bổ sung package Python `creditrep` trong `src/` để các phase thực nghiệm sau dùng chung một interface load dataset:

```python
from creditrep.datasets import load_dataset

loaded = load_dataset("GC")
loaded.dataset_id
loaded.features
loaded.target
loaded.metadata
```

`data/datasets.yaml` tiếp tục là source of truth. Mỗi dataset khai báo `active_file`, `reader`, `target.mapping_to_binary`, `identifier_columns`, `ignored_columns`, `numeric_columns` và `categorical_columns`. Loader đọc file active, validate schema, chuẩn hóa target về `0 = good/non-default` và `1 = bad/default`, rồi loại target cùng identifier/ignored columns khỏi `features`.

Kiểm tra nhanh một dataset mà không in raw records:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_dataset.py --dataset GC
.\.venv\Scripts\python.exe scripts\inspect_dataset.py --dataset TC
.\.venv\Scripts\python.exe scripts\inspect_dataset.py --dataset GMC
```

P2A không thực hiện imputation, encoding, scaling, WOE, VIF, train/test split, cross-validation, training model, metrics hoặc ghi experiment artifact. Các phần đó thuộc các bước P2B/P2C và phase sau.

## Split và artifact P2B

Phase 2B bổ sung deterministic stratified holdout split và experiment artifact contract. Config mẫu nằm trong:

- `configs/experiments/split_gc.yaml`
- `configs/experiments/split_tc.yaml`

Tạo split artifact:

```powershell
.\.venv\Scripts\python.exe scripts\create_split_artifact.py --config configs\experiments\split_gc.yaml
.\.venv\Scripts\python.exe scripts\create_split_artifact.py --config configs\experiments\split_tc.yaml
```

CLI load config, load dataset bằng P2A, kiểm tra checksum từ `data/checksums-sha256.csv`, tạo split deterministic và ghi artifact vào `artifacts/experiments/<experiment_id>/`. Artifact gồm `manifest.json`, `config.yaml`, `split.json` và `split.csv`. `split.csv` chỉ lưu `row_position,partition`, không lưu feature values hoặc target values.

`config_hash` là SHA-256 trên config đã parse/normalize. `split_hash` là SHA-256 trên payload canonical gồm dataset ID, source file portable, checksum active file, strategy, test size, seed, train indices và test indices. Cùng dataset/config/seed tạo cùng `split_hash`; timestamp chỉ ảnh hưởng `experiment_id`, không ảnh hưởng `split_hash`.

Chi tiết contract nằm trong `docs/EXPERIMENT_ARTIFACT_CONTRACT.md`.

P2B không train model, không tính metrics, không sinh predictions và không chạy smoke experiment. Các phần đó thuộc P2C và phase sau.

## Smoke experiment runner P2C

Phase 2C bổ sung runner end-to-end tối thiểu để xác nhận pipeline có thể load dataset, kiểm tra checksum, tái dùng deterministic split P2B, fit preprocessing train-only, train Logistic Regression hoặc XGBoost, sinh probability test set, tính metrics tối thiểu và ghi artifact.

Ba config bắt buộc:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\experiments\smoke_gc_lr.yaml
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\experiments\smoke_gc_xgb.yaml
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\experiments\smoke_tc_lr.yaml
```

Smoke artifacts mở rộng P2B với `metrics.json`, `predictions.csv` và `model_metadata.json`. `predictions.csv` chỉ chứa `row_position,partition,y_true,y_score,y_pred`, không chứa raw features. Mọi smoke run có `publishable: false` và `result_scope: smoke_validation`; các metric này chỉ dùng để xác minh pipeline, không phải kết quả nghiên cứu và không đưa lên website như kết quả khoa học.

Chi tiết runner nằm trong `docs/SMOKE_EXPERIMENT_RUNNER.md`.

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
py -3.11 -m venv .venv-clean
.\.venv-clean\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[test]"

python scripts\convert_th02.py --input data/raw/th02/public.xls --output data/processed/th02/th02.csv
python scripts\verify_credit_datasets.py --dataset all --checksums-only
python scripts\verify_credit_datasets.py --dataset all
python scripts\inspect_dataset.py --dataset GC
python scripts\create_split_artifact.py --config configs\experiments\split_gc.yaml
python scripts\run_experiment.py --config configs\experiments\smoke_gc_lr.yaml
python -m pytest
python -m pip check

deactivate
Remove-Item -Recurse -Force .venv-clean
```

`.venv-clean` chỉ là môi trường kiểm tra tạm thời và không được commit. Số lượng test có thể tăng theo thời gian; tiêu chí dài hạn là toàn bộ test suite hiện hành phải pass.

## Trạng thái hiện tại

- Đã tải paper.
- Đã tải đủ 6 dataset công khai.
- Đã bổ sung HMEQ full tại `data/raw/hmeq/hmeq_full.csv`; file HMEQ 604 dòng cũ được giữ nguyên nhưng không dùng cho core replication.
- Đã chuyển đổi TH02 từ `data/raw/th02/public.xls` sang `data/processed/th02/th02.csv` bằng `scripts/convert_th02.py`.
- Đã chuyển `data/checksums-sha256.csv` sang relative portable paths.
- Đã tạo `data/datasets.yaml` làm registry metadata chính.
- Đã có script kiểm tra dữ liệu `scripts/verify_credit_datasets.py`.
- Đã triển khai P2A dataset loader trong `src/creditrep/datasets/`, kèm CLI `scripts/inspect_dataset.py` và unit test fixture độc lập raw data.
- Đã triển khai P2B deterministic split và artifact contract trong `src/creditrep/config/`, `src/creditrep/splitting/`, `src/creditrep/artifacts/`, kèm CLI `scripts/create_split_artifact.py`.
- Đã triển khai P2C smoke experiment runner trong `src/creditrep/preprocessing/`, `src/creditrep/models/`, `src/creditrep/evaluation/`, `src/creditrep/experiments/`, kèm CLI `scripts/run_experiment.py`.
- Đã hoàn tất Phase 4 metric validation: ROC AUC, Brier Score và Partial Gini có production implementation/reference validation; EMP được chốt ở trạng thái `unsupported` có provenance rõ ràng; metric config registry, nested-CV integration và metric-validation artifacts đã có ở mức non-publishable.

<!-- PROJECT_STATUS:BEGIN -->
Generated from website/content/progress.yaml. Do not edit manually.

- Last completed phase: Phase 6
- Current phase: Phase 7
- Next phase: Phase 7 - Core replication run
- Updated at: 2026-08-15

| Phase | Status | Milestone tag |
| --- | --- | --- |
| Phase 0 - Data verification baseline | Completed | p0-data-verification-baseline |
| Phase 1 - Website production foundation | Completed | p1-website-production-complete |
| Phase 2 - Experiment foundation | Completed | p2-experiment-foundation-complete |
| Phase 3 - Leakage-safe preprocessing | Completed | p3-leakage-safe-preprocessing-complete |
| Phase 4 - Metric validation | Completed | p4-metric-validation-complete |
| Phase 5 - Classical and ensemble models | Completed | p5-classical-replication-complete |
| Phase 6 - MLP depth replication | Completed | p6-mlp-depth-replication-complete |
| Phase 7 - Core replication run | In Progress | - |
| Phase 8 - Modern reassessment | Planned | - |
| Phase 9 - So sánh thống kê | Planned | - |
| Phase 10 - Robustness | Planned | - |
| Phase 11 - Báo cáo cuối và website kết quả | Planned | - |

Phase 3 checkpoints:
- P3A: Completed - Preprocessing contract và train-only imputation.
- P3B: Completed - WOE, iterative VIF và train-only scaling.
- P3C: Completed - Nested CV, fold persistence, per-fold preprocessing và tuning isolation.

Phase 4 checkpoints:
- P4A: Completed - Metric specification và metric contract.
- P4B: Completed - ROC AUC, Brier Score và Partial Gini implementation/reference validation.
- P4C: Completed - EMP unsupported có provenance, metric config registry, nested-CV integration và metric-validation harness.

Phase 5 checkpoints:
- P5A: Completed - Model contract and configuration foundation.
- P5B: Completed - Classical model implementations.
- P5C: Completed - Nested-CV model-validation harness với atomic per-fold artifacts, resume/retry, failure stages và deterministic reconciliation.

Phase 6 checkpoints:
- P6A: Completed - Typed PyTorch MLP contract, logits-only shared builder, CPU-first tiny trainer, device/seed policy and in-memory early stopping foundation. Artifacts remain non-publishable.
- P6B: Completed - Stable MLP-1/3/5 IDs, shared PyTorch wrapper, fair training-budget specification, model configs and provenance metadata. Artifacts remain non-publishable.
- P6C: Completed - Neural nested-CV integration/hardening, deterministic early stopping, artifacts, retry/resume, GC reduced validation và TC resource checkpoint completed; non-publishable engineering evidence only.

Phase 7 checkpoints:
- P7A: Completed - Protocol bất biến, candidate manifest P7B, provenance Table 2/3, CART-A Grid 2 approved cho pilot non-publishable, validator và CLI đã hoàn thành; final scientific search space chờ P7B closeout.
- P7B: Completed - P7B.1 runner/hardening và P7B.2 CART engineering-feasibility đã completed: run-002 tại Git 989997a9dd8cb792636d99f5e2b243b5775807ed có 60/60 fit, 0 failed, 0 pending, artifact validated. Decision record đã khóa final CART-A grid 12 candidates; runtime/RSS vẫn chỉ là bằng chứng kỹ thuật non-publishable.
- P7C: In Progress - P7C vẫn đang thực hiện. P7C.3 đã completed với canonical `vm-run-003`; P7C.4–P7C.7 còn chưa hoàn tất và scientific execution chưa bắt đầu. `vm-run-001` là historical/non-canonical, `vm-run-002` là historical invalid; không dùng hai artifact này cho workload projection hoặc kết luận khoa học.
- P7C.1: Completed - Protocol inventory, readiness matrix, decision register và validator đã completed; không chạy training hoặc tạo scientific result.
- P7C.2: Completed - RF/XGBoost engineering-feasibility và protocol decision đã completed: immutable 60-fit pilot 60/60 completed, artifact validated; final full P7A/Table-2 grids RF 30 và XGBoost 108 candidates đã khóa.
- P7C.2.1: Completed - Immutable 60-fit plan, runner, CLI, atomic artifacts, telemetry, resume/retry và artifact validator đã hoàn thành và qua acceptance tests.
- P7C.2.2: Completed - Research pilot đã completed: 60/60 fit, 0 failed, 0 missing; artifact validation PASS. Engineering evidence non-publishable theo immutable plan digest.
- P7C.2.3: Completed - Analysis và final decision completed: full P7A/Table-2 RF 30-candidate và XGBoost 108-candidate grids locked; compute worksheet chỉ dùng cho planning, không authorize execution.
- P7C.3: Completed - Completed — canonical feasibility pilot accepted. Canonical `vm-run-003` tại Git `84c71266d0eb375effc317601602fb9deb67d7d2` có artifact validator PASS, 60/60 fit completed, 0 failed/missing, CPU feasibility PASS, memory feasibility PASS và execution stability PASS. GPU không bắt buộc cho correctness/feasibility MLP, nhưng quyết định GPU để tối ưu thời gian còn chờ workload projection và phê duyệt. Artifact engineering này non-publishable, không phải kết quả hiệu năng dự báo hoặc kết quả khoa học cuối cùng.
- P7C.4A: Completed - Completed — benchmark plan ready for human review: decision study, ba candidate-budget scenario, threshold đề xuất, benchmark matrix, telemetry/artifact contract và digest đã được kiểm tra. P7C.4B.2a sau đó đã ghi nhận human decision riêng; không chạy benchmark.
- P7C.4B: In Progress - P7C.4B.1 harness/readiness và P7C.4B.2a–P7C.4B.2e contracts đã completed. Fresh target canary `p7c4b2d-target-rerun-01` đã operationally accepted (4/4 `target_preflight` task ở `cpu_parallel_2`); đây không authorize target compute/outer-refit preflight hay canonical/GPU/multi-VM scientific execution.
- P7C.4B.1: Completed - Completed — B1a CPU sequential, B1b validation/resume/corruption, B1c CPU parallel-2 và B1d independent operational readiness review passed. Engineering smoke remains non-publishable; no canonical benchmark ran.
- P7C.4B.1d: Completed - Completed — independent operational readiness review, artifact read-back, full B1 regression, operator checklist and go/no-go policy completed.
- P7C.4B.2: In Progress - Fresh target canary đã operationally accepted. Outer P1 projection-preflight trên source `aac4504...` đã fail-closed sau 9 AC task do environment chỉ bind AC/GMC; failed evidence được giữ nguyên, không có valid outer projection result và không được resume/promote. Hotfix closed-world six-dataset/typed receipt cần fresh post-merge common-SHA B2b P1/P2 và outer evidence. Scientific projection và canonical execution vẫn pending/NO-GO.
- P7C.4B.2a: Completed - Completed — balanced MLP-1/3/5 scientific scope 24/48/48 generated deterministically from seed 42, validated and digest-locked; workload 54,270 fits, execution guard and static multi-VM sharding readiness contract added. No execution occurred.
- P7C.4B.2b: Completed - Final-reviewed executable bounded-preflight harness ready — deterministic plan/CLI, distinct typed profile/environment → proposal → effective-authorization chain per P1/P2, append-only runtime generations, per-task attempt history, bounded process-tree cleanup, fail-closed RSS sampling, unique temporary attempts, pre-promotion/finalization guards, canonical control/path revalidation và typed launch/submission records pass controlled regression. Existing B2b safety limits remain unchanged; boolean-only authorization is rejected. No target workload ran; target preflight remains pending.
- P7C.4B.2c: Completed - Implementation completed / valid target result pending — historical outer P1 attempt failed after 9 AC tasks because its environment bound only AC/GMC. Failed evidence is retained and terminal-invalid; there is no valid projection result. Closed-world six-dataset binding and transient-unit receipt recovery hotfix is under review; fresh post-merge evidence is required and canonical execution remains NO-GO.
- P7C.4B.2d: Completed - Fresh target environment, proposal và effective authorization cho canary `p7c4b2d-target-rerun-01` đã cross-validate với Git/source, AC/GMC input identity, locked runtime inputs, mode, resource scope và bốn task. Authorization đã hết scope tại canary và không authorize canonical scientific execution; canonical execution vẫn NO-GO.
- P7C.4B.2e: Completed - Fresh controlled target canary `p7c4b2d-target-rerun-01` completed and was operationally accepted: immutable launch record, one-time submission receipt, exact 4/4 authorized tasks, telemetry, completion marker và public validation đều hợp lệ. Đây không phải scientific execution/result và không làm scientific projection hay canonical execution eligible.

Current scope limits:
- Core replication has not run.
- Smoke, reduced, fake, preprocessing-validation, and metric-validation artifacts remain non-publishable validation artifacts.
- Website still must not present validation artifacts as scientific results.
- Phase 6 completed the non-publishable MLP infrastructure/hardening validation; Phase 7 is In Progress.
<!-- PROJECT_STATUS:END -->
