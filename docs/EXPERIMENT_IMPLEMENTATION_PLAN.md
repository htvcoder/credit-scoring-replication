# Kế hoạch triển khai tổng thể đề tài thực tập `credit-scoring-replication`

## Mục lục

1. Mục tiêu và định vị đề tài
2. Baseline hiện tại của repository
3. Phạm vi và ngoài phạm vi
4. Kiến trúc tổng thể
5. Luồng dữ liệu và luồng công bố kết quả
6. Sơ đồ phụ thuộc giữa các phase
7. Roadmap chi tiết Phase 0 đến Phase 11
8. Kiến trúc module website
9. Kiến trúc module thực nghiệm
10. Kiến trúc CI/CD
11. Kiến trúc triển khai Google Cloud VM
12. Hợp đồng artifact và hợp đồng kết quả công khai
13. Chiến lược kiểm thử
14. Ma trận thực nghiệm sơ bộ
15. Kế hoạch tài nguyên
16. Đối chiếu câu hỏi nghiên cứu
17. Nhật ký quyết định
18. Danh mục rủi ro
19. Milestone, commit và tag
20. Tiêu chí hoàn thành toàn dự án
21. Thứ tự triển khai khuyến nghị
22. Giai đoạn tiếp theo
23. Quyết định cần người dùng chốt

## 1. Mục tiêu và định vị đề tài

Tài liệu này là kế hoạch triển khai tổng thể cho đề tài thực tập `credit-scoring-replication`. Đề tài không chỉ là một chuỗi thực nghiệm mô hình, mà gồm ba luồng công việc liên kết với nhau:

1. **Luồng website và công bố thông tin**: xây dựng website công khai để giới thiệu đề tài, trình bày mục tiêu, câu hỏi nghiên cứu, phạm vi dữ liệu, phương pháp, tiến độ và kết quả đã được kiểm chứng.
2. **Luồng hạ tầng và DevOps**: triển khai website trên máy ảo Google Cloud (Google Cloud VM), Docker hóa, thiết lập tích hợp và triển khai liên tục (CI/CD), health check, rollback, logging và bảo mật. Domain riêng và HTTPS là hạng mục Optional/Should trong phạm vi website báo cáo thực tập hiện tại.
3. **Luồng nghiên cứu và thực nghiệm**: thực hiện partial replication bài báo “Deep Learning for Credit Scoring: Do or Don’t?”, modern reassessment với CatBoost, TabNet, FT-Transformer, phân tích thống kê, robustness và reproducibility package.

**Định hướng bắt buộc:** website và nền tảng CI/CD phải được xây dựng, kiểm thử và triển khai trước khi bắt đầu coding pipeline thực nghiệm. Website không chỉ là sản phẩm cuối; từ Phase 1 trở đi, website là nơi cập nhật tiến độ và sau này công bố kết quả đã qua kiểm chứng.

## 2. Baseline hiện tại của repository

### 2.1. Trạng thái Git và audit repository

| Hạng mục | Trạng thái tại thời điểm lập kế hoạch |
|---|---|
| Branch | `main`, tracking `origin/main` |
| HEAD | `73729c5071caf14f17b6cf2d4ba2ca07e11663ec` |
| Working tree trước khi viết lại plan | Chỉ có `docs/EXPERIMENT_IMPLEMENTATION_PLAN.md` đang untracked |
| Tag Phase 0 | `p0-data-verification-baseline` tồn tại |
| `AGENTS.md` | Không tồn tại |
| `.agents/` | Tồn tại nhưng không có file con |
| `.codex/` | Tồn tại nhưng không có file con |
| Website hiện có | Chưa có `website/` |
| CI/CD hiện có | Chưa có `.github/workflows/` |
| Docker/deployment hiện có | Chưa có Dockerfile, `.dockerignore`, `docker-compose*.yml`, `deploy/` hoặc Nginx config |
| Paper PDF | Có trên đĩa nhưng bị ignore theo chính sách |

### 2.2. Baseline dữ liệu Phase 0

**Fact từ repository:** Phase 0 đã hoàn thành ở trạng thái **PASS WITH CAVEAT**.

- 6 dataset công khai đã verify: AC, GC, HMEQ, TH02, TC, GMC.
- Registry chính: `data/datasets.yaml`.
- Checksum portable: `data/checksums-sha256.csv`.
- Data cards: `docs/data-cards/`.
- Scripts hiện có: `scripts/verify_credit_datasets.py`, `scripts/convert_th02.py`, `scripts/download_hmeq.py`.
- Tests hiện có: `tests/test_verify_credit_datasets.py`.
- Raw và processed data không được commit theo `.gitignore`.
- HMEQ vẫn có caveat provenance: shape, schema và class distribution khớp, nhưng checksum không phải checksum SAS artifact kỳ vọng.

### 2.3. Baseline tài liệu và paper

**Fact từ bài báo:**

- Paper gốc dùng 10 dataset; repository chỉ có 6 dataset public, nên dự án chỉ có thể là partial replication.
- Paper gốc dùng LR, Decision Tree C4.5, RF, XGBoost, MLP-1, MLP-3, MLP-5, DBN-1, DBN-3, DBN-5.
- Dự án hiện tại không đưa DBN vào core scope; đây là deviation cần giữ trong report.
- Preprocessing paper: mean/mode imputation, Weight of Evidence (WOE), Variance Inflation Factor (VIF) với ngưỡng `VIF <= 10`, không dùng class balancing.
- CV paper: outer `N x 2` CV và inner five-fold CV.
- Metric paper: AUC, Brier Score, Partial Gini với `b = 0.4`, Expected Maximum Profit (EMP).
- So sánh thống kê paper: average rank, Friedman, Rom, Nemenyi, Bayesian signed-rank test, ROPE.

### 2.4. Vấn đề tham chiếu phase hiện có

`docs/EXPERIMENT_FEASIBILITY_ASSESSMENT.md` vẫn tham chiếu Phase 1 là smoke experiment. Tài liệu này không được sửa trong nhiệm vụ hiện tại. Kế hoạch mới sẽ ghi rõ phase mới và nếu cần ở phase sau có thể cập nhật feasibility report để đồng bộ thuật ngữ.

## 3. Phạm vi và ngoài phạm vi

### 3.1. Phạm vi của kế hoạch này

- Viết lại toàn bộ `docs/EXPERIMENT_IMPLEMENTATION_PLAN.md` bằng tiếng Việt có dấu.
- Định vị lại kế hoạch thành kế hoạch tổng thể cho đề tài thực tập.
- Đưa website, Google Cloud VM và CI/CD lên trước pipeline thực nghiệm.
- Đánh số lại roadmap từ Phase 0 đến Phase 11.
- Giữ các nội dung khoa học quan trọng của plan cũ: loader, split, WOE, VIF, nested CV, metrics, artifact contract, statistical comparison, robustness, Must/Should/Optional, danh mục rủi ro, nhật ký quyết định và tiêu chí hoàn thành.

### 3.2. Ngoài phạm vi của nhiệm vụ hiện tại

- Không tạo website.
- Không tạo source code.
- Không tạo Dockerfile.
- Không tạo workflow CI/CD.
- Không tạo deployment script.
- Không sửa README, scripts, tests, data registry, data cards hoặc dependencies.
- Không chạy model, không chạy thực nghiệm, không tạo result.
- Không commit, push hoặc tạo tag.

### 3.3. Ghi chú về tên file

Tên `EXPERIMENT_IMPLEMENTATION_PLAN.md` vẫn sử dụng được vì phần thực nghiệm vẫn là lõi nghiên cứu. Tuy nhiên, nếu muốn phản ánh đầy đủ hơn phạm vi website và DevOps, có thể cân nhắc đổi tên trong một nhiệm vụ riêng, ví dụ `PROJECT_IMPLEMENTATION_PLAN.md`. Không đổi tên file trong nhiệm vụ hiện tại.

## 4. Kiến trúc tổng thể

### 4.1. Ba lớp kiến trúc

| Lớp | Trách nhiệm | Ghi chú |
|---|---|---|
| Website công khai | Giới thiệu đề tài, hiển thị tiến độ, công bố kết quả đã kiểm chứng | Phase 1 phải triển khai trước pipeline thực nghiệm |
| Hạ tầng DevOps | CI/CD, Docker, Google Cloud VM, reverse proxy nếu cần, health check, rollback | Không dùng VM website để chạy training nặng; HTTPS bổ sung khi có domain/hostname phù hợp |
| Pipeline thực nghiệm | Load dữ liệu, split, preprocessing, model, metric, artifact, thống kê, báo cáo | Chạy local hoặc compute environment riêng |

### 4.2. Cấu trúc thư mục dự kiến

```text
.
├── website/
│   ├── app/ hoặc src/
│   ├── content/
│   ├── public/
│   ├── tests/
│   ├── Dockerfile
│   └── .dockerignore
├── deploy/
│   └── nginx/
├── .github/
│   └── workflows/
├── configs/
│   ├── experiments/
│   ├── models/
│   └── protocols/
├── src/
│   └── creditrep/
├── scripts/
├── tests/
├── results/
│   ├── raw/
│   ├── aggregated/
│   ├── figures/
│   └── reports/
└── docs/
```

**Đề xuất:** website nằm trong `website/`, tách khỏi package Python thực nghiệm. `results/raw/` và artifact nội bộ không được đưa vào Docker build context của website.

### 4.3. Công nghệ website đề xuất

| Thành phần | Đề xuất mặc định | Lý do |
|---|---|---|
| Framework | Next.js |
| Ngôn ngữ | TypeScript |
| Rendering | Static-first hoặc server-side rendering tối thiểu |
| Backend | Không cần backend API riêng trong Phase 1 |
| Database | Không cần database trong Phase 1 |
| Nội dung | Markdown, YAML hoặc JSON source-controlled |
| Đóng gói | Docker |
| Triển khai | Google Cloud VM + Nginx reverse proxy nếu cần; HTTP/public IP đủ cho Phase 1, HTTPS là Optional/Should |

Nếu chọn static site generator khác, cần so sánh chi phí triển khai, mức độ quen thuộc, CI/CD, Docker hóa và khả năng cập nhật nội dung.

## 5. Luồng dữ liệu và luồng công bố kết quả

### 5.1. Luồng dữ liệu nghiên cứu

1. Dữ liệu raw nằm trong `data/raw/`, không commit.
2. Dữ liệu format-converted cần thiết nằm trong `data/processed/`, không commit.
3. Registry và checksum xác định dataset active.
4. Pipeline thực nghiệm sinh fold artifacts, predictions và metrics nội bộ.
5. Aggregation tạo artifact đã tổng hợp.
6. Validation kiểm tra artifact tổng hợp.
7. Chỉ artifact public/sanitized mới được đưa sang website.

### 5.2. Hợp đồng nội dung công khai

Nguồn nội dung website dự kiến:

- `website/content/project.yaml`
- `website/content/internship.yaml`
- `website/content/progress.yaml`
- `website/content/datasets.json` nếu cần, nhưng chỉ là generated artifact hoặc checked derivative từ `data/datasets.yaml`
- `website/content/methods.md`
- `website/content/deviations.md`
- `website/public/results/`

Phân loại:

| Loại nội dung | Có được đưa lên website không? | Ghi chú |
|---|---|---|
| Nội dung giới thiệu source-controlled | Có | Markdown/YAML/JSON đã review |
| Tiến độ phase và milestone | Có | Cập nhật sau milestone hoặc tag |
| Bảng metric aggregated | Có, sau validation | Không chứa prediction cấp bản ghi |
| Biểu đồ aggregated | Có, sau validation | Không chứa dữ liệu nhạy cảm |
| Raw datasets | Không | Tuyệt đối không công khai qua website |
| Train/test indices chi tiết | Không mặc định | Chỉ công bố nếu có lý do khoa học và đã review |
| Prediction từng bản ghi | Không | Tránh rủi ro dữ liệu nhạy cảm |
| Secret, local path, deployment log nhạy cảm | Không | Phải scan trước publish |

### 5.3. Source of truth cho nội dung dataset trên website

Để tránh trùng lặp và lệch dữ liệu giữa registry, data cards và website, kế hoạch áp dụng nguyên tắc source of truth như sau:

1. `data/datasets.yaml` là nguồn metadata có cấu trúc chính cho dataset.
2. `docs/data-cards/` là tài liệu diễn giải chi tiết dành cho con người.
3. Nội dung dataset trên website không được nhập tay độc lập nếu có thể tránh.
4. Nếu có `website/content/datasets.json`, file này phải là generated artifact hoặc checked derivative, không phải nguồn dữ liệu độc lập.
5. CI phải fail nếu thông tin công khai quan trọng không khớp registry, tối thiểu gồm dataset ID, shape, target, default rate, caveat và trạng thái usable.
6. Website không được đưa trường nội bộ, đường dẫn local, raw path nhạy cảm hoặc thông tin không cần công khai.

### 5.4. Luồng xuất bản kết quả công khai

Không copy trực tiếp toàn bộ `results/` vào website. Kết quả công khai phải đi qua một bước export riêng:

```text
internal result
  -> aggregate
  -> validate
  -> sanitize
  -> public artifact
  -> website
```

Public artifact chỉ được chứa aggregated metrics, rankings, statistical tables, validated figures, summarized runtime, deviation và limitation. Public artifact không được chứa row-level predictions, raw data, train/test indices, internal exception logs, local paths, environment secrets hoặc model checkpoint chưa được duyệt.

## 6. Sơ đồ phụ thuộc giữa các phase

```text
Phase 0 - Xác minh dữ liệu nền tảng
  -> Phase 1 - Website và CI/CD
    -> Phase 2 - Nền tảng thực nghiệm và smoke test
      -> Phase 3 - Preprocessing chống leakage
        -> Phase 4 - Metric và đánh giá kinh doanh
          -> Phase 5 - Mô hình truyền thống và ensemble
            -> Phase 6 - MLP depth replication
              -> Phase 7 - Core replication run
            -> Phase 8 - Modern reassessment

Phase 7 + Phase 8 nếu có modern results
  -> Phase 9 - So sánh thống kê
    -> Phase 10 - Robustness
      -> Phase 11 - Báo cáo cuối và website kết quả
```

Ràng buộc phụ thuộc cần giữ nhất quán trong toàn bộ roadmap:

- Phase 5 phụ thuộc Phase 3 và Phase 4.
- Phase 6 phụ thuộc Phase 3, Phase 4 và các pattern từ Phase 5.
- Phase 7 phụ thuộc Phase 3, Phase 4, Phase 5 và Phase 6.
- Phase 8 phụ thuộc tối thiểu Phase 3, Phase 4 và Phase 5, vì cần baseline XGBoost trước khi đánh giá modern models.
- Phase 9 chỉ bắt đầu khi có kết quả Phase 7; nếu đánh giá modern models thì Phase 9 đồng thời cần kết quả Phase 8.
- Phase 10 phụ thuộc Phase 9.
- Phase 11 phụ thuộc Phase 10.

Điểm sửa quan trọng so với plan cũ: **Phase 4 - Metric validation đứng trước Phase 7 - Core replication run**. Không chạy core experiment quy mô lớn khi Partial Gini và EMP chưa được xác định rõ exact/approximate.

## 7. Roadmap chi tiết Phase 0 đến Phase 11

### Phase 0 - Xác minh dữ liệu nền tảng

| Mục | Nội dung |
|---|---|
| Phân loại | Must, đã hoàn thành |
| Mục tiêu | Xác lập baseline dữ liệu verified cho 6 dataset công khai. |
| Phạm vi | Mô tả evidence hiện có; không lặp lại công việc Phase 0. |
| Đầu vào | Raw/processed local, `data/datasets.yaml`, checksum, data cards, verifier, tests. |
| Đầu ra | Baseline tagged `p0-data-verification-baseline`. |
| Nhiệm vụ chi tiết | Giữ nguyên registry, checksum, scripts và tests; propagate HMEQ caveat sang các phase sau. |
| File dự kiến tạo/sửa | Không tạo/sửa trong Phase 0 hiện tại; các file đã có gồm `data/datasets.yaml`, `docs/data-cards/*.md`, `scripts/*.py`, `tests/test_verify_credit_datasets.py`. |
| Test | `python scripts/verify_credit_datasets.py --dataset all`, `python scripts/verify_credit_datasets.py --dataset all --checksums-only`, `python -m pytest tests/test_verify_credit_datasets.py`. |
| Acceptance criteria | 6 dataset pass; checksum portable; metadata target/numeric/categorical/identifier pass; HMEQ caveat documented. |
| Dependency | Không. |
| Rủi ro | HMEQ provenance, license/access chưa xác định, raw data không nằm trong repository. |
| Go/no-go | Go cho Phase 1. |
| Ngoài phạm vi | Website, CI/CD, training pipeline, model result. |
| Commit đề xuất | `chore(data): establish phase 0 verification baseline` |
| Tag đề xuất | `p0-data-verification-baseline` |
| Website cần cập nhật | Phase 0 hiển thị là Completed, kèm caveat HMEQ và link data cards. |

### Phase 1 - Website giới thiệu đề tài và nền tảng CI/CD

| Mục | Nội dung |
|---|---|
| Phân loại | Must |
| Mục tiêu | Tạo website công khai, Docker hóa, triển khai lên Google Cloud VM và thiết lập CI/CD đầy đủ từ đầu. |
| Phạm vi | Website nội dung ban đầu, trang hoặc khu vực “Giới thiệu đề tài thực tập”, content contract, Docker, CI workflow, production deployment workflow, Google Cloud VM, reverse proxy nếu cần, health check, rollback, logging, deployment runbook. Domain riêng, subdomain riêng, HTTPS, Let’s Encrypt và chứng chỉ SSL/TLS là Optional/Should. |
| Đầu vào | README, feasibility report, data cards, `data/datasets.yaml`, plan này, paper Markdown. |
| Đầu ra | Website production chạy ổn định trên Google Cloud VM, truy cập được từ Internet bằng public IP, hostname hoặc domain; CI/CD hoạt động; health check pass; rollback đã kiểm thử; website hiển thị Git commit SHA hoặc version đang deploy; trang tiến độ hiển thị Phase 0 completed và Phase 1 completed sau tag; kiểm tra dữ liệu và bảo mật pass. |
| Nhiệm vụ chi tiết | Tạo `website/`; xây trang chủ, giới thiệu đề tài thực tập, giới thiệu nghiên cứu, RQ, dữ liệu, phương pháp, tiến độ, kết quả placeholder, tái lập; tạo content schema; Docker hóa website; tạo CI; tạo deploy workflow; cấu hình VM; cấu hình reverse proxy nếu cần; thiết lập health check, rollback và logging; cho phép truy cập bằng public IP, hostname hoặc domain; có thể bổ sung HTTPS nếu có domain/hostname phù hợp; viết runbook; kiểm tra no raw data và no secret. |
| File dự kiến tạo/sửa | `website/`, `website/Dockerfile`, `website/.dockerignore`, `docker-compose.prod.yml`, `.github/workflows/ci.yml`, `.github/workflows/deploy-production.yml`, `deploy/nginx/`, `scripts/deploy-production.sh`, `scripts/rollback-production.sh`, `docs/DEPLOYMENT_GOOGLE_CLOUD.md`. |
| Test | Website lint, TypeScript type check, unit/component test nếu có, production build, Docker build, Python Phase 0 tests, content source validation, secret/raw-data scan, no forbidden files in image, health check, deployment smoke test, rollback test, responsive smoke, accessibility smoke. |
| Acceptance criteria | Clean clone có thể cài dependencies, chạy test, build website và build Docker image; website được deploy lên Google Cloud VM; website truy cập được từ Internet bằng public IP, hostname hoặc domain; website responsive, tiếng Việt có dấu, có trang hoặc khu vực giới thiệu đề tài thực tập, không có kết quả giả; CI chạy thành công; production deployment workflow chạy thành công; health check pass; rollback được kiểm thử thực tế ít nhất một lần; website hiển thị Git commit SHA hoặc version đang deploy; Docker image không chứa `data/raw/`, `data/processed/`, row-level prediction, secret hoặc `.env` production; trang tiến độ hiển thị đúng Phase 0 và Phase 1; domain và HTTPS là Optional/Should, không phải acceptance criteria bắt buộc; có thể tạo tag `p1-website-cicd-foundation` khi các tiêu chí bắt buộc đã pass, kể cả website đang truy cập qua HTTP và public IP. |
| Dependency | Phase 0. |
| Rủi ro | Raw data lọt vào Docker context, secret bị commit, deploy lỗi gây downtime, image không rollback được, workflow PR không tin cậy truy cập secret, URL public IP khó nhớ, website chỉ dùng HTTP nếu chưa có domain/hostname phù hợp. |
| Go/no-go | Go Phase 2 khi website đã deploy được trên Google Cloud VM, truy cập được từ Internet, CI/CD hoạt động, health check pass, rollback pass và không có vi phạm dữ liệu hoặc bảo mật. Không để Phase 1 ở trạng thái Blocked chỉ vì chưa mua domain, chưa cấu hình DNS, chưa có HTTPS hoặc chưa cài Let’s Encrypt. |
| Ngoài phạm vi | Training job, backend API, database, result thật, admin dashboard. |
| Commit đề xuất | `feat(website): add project website and production delivery foundation` |
| Tag đề xuất | `p1-website-cicd-foundation` |
| Website cần cập nhật | Website là sản phẩm chính của phase; trạng thái Phase 1 chuyển từ In progress sang Completed sau tag. |

Checkpoint nội bộ của Phase 1:

| Checkpoint | Mục tiêu | Acceptance criteria |
|---|---|---|
| P1A - Website local và nội dung ban đầu | Khởi tạo Next.js, xây các trang nội dung, content schema, responsive, accessibility cơ bản, placeholder kết quả đúng quy định, hiển thị Phase 0 Completed | Local build pass; lint/type check pass; không có kết quả giả; content validation pass |
| P1B - Docker và CI | Docker hóa website, `.dockerignore`, CI workflow, production build, Docker build, no raw data, no secret, version metadata | Docker image build pass; CI pass; image không chứa dữ liệu bị cấm; clean clone build pass |
| P1C - Google Cloud VM và triển khai liên tục | Cấu hình Google Cloud VM; cài Docker Engine và Docker Compose; triển khai website; cấu hình reverse proxy nếu cần; thiết lập production deployment workflow, health check, rollback và logging; cho phép truy cập bằng public IP, hostname hoặc domain; có thể bổ sung HTTPS nếu có domain/hostname phù hợp | Website production truy cập được từ Internet bằng public IP, hostname hoặc domain; deployment workflow chạy thành công; health check pass; rollback đã kiểm thử; website hiển thị đúng Git commit SHA hoặc version; logging hoạt động; production image không có raw data hoặc secret; domain và HTTPS không bắt buộc; có thể đánh dấu P1C và Phase 1 Completed, đồng thời tạo tag `p1-website-cicd-foundation`, khi website đang chạy qua HTTP/public IP |

Không tạo tag riêng cho P1A/P1B/P1C, trừ khi sau này cần internal tags. Tag chính của Phase 1 vẫn là `p1-website-cicd-foundation`.

### Phase 2 - Nền tảng thực nghiệm và smoke test

| Mục | Nội dung |
|---|---|
| Phân loại | Must |
| Mục tiêu | Xây nền tảng config, loader, split, artifact contract và smoke test tối thiểu. |
| Phạm vi | Dataset loader, target normalization, identifier removal, split deterministic, artifact writer, Logistic Regression, XGBoost, smoke test GC và TC. |
| Đầu vào | Phase 1 CI/CD, Phase 0 registry/data cards/verifier. |
| Đầu ra | Skeleton `src/creditrep/`, configs smoke, runner CLI, smoke artifacts nội bộ. |
| Nhiệm vụ chi tiết | Implement registry loader; map target về 0/1; loại `ID`/`Unnamed: 0`; tạo stratified split; lưu split hash; tạo artifact JSON/CSV; chạy GC LR/XGBoost smoke và TC LR smoke. |
| File dự kiến tạo/sửa | `src/creditrep/datasets/`, `src/creditrep/splitting/`, `src/creditrep/artifacts/`, `src/creditrep/runners/`, `configs/experiments/smoke_*.yaml`, `scripts/run_experiment.py`, tests tương ứng. |
| Test | Unit target mapping, loader, identifier removal, split determinism, artifact serialization; integration smoke fixture; CI không cần raw data hoặc skip rõ. |
| Acceptance criteria | Smoke GC và TC tạo predictions/metrics hợp lệ; artifact có Git commit, config hash, dataset checksum, seed; không có leakage cơ bản. |
| Dependency | Phase 1. |
| Rủi ro | XGBoost dependency nặng, artifact contract quá phức tạp, raw data thiếu trong môi trường CI. |
| Go/no-go | Go Phase 3 khi smoke runner deterministic và artifact schema ổn định. |
| Ngoài phạm vi | WOE/VIF đầy đủ, full nested CV, MLP, statistical analysis. |
| Commit đề xuất | `feat(experiments): add foundation and smoke-test runner` |
| Tag đề xuất | `p2-experiment-foundation` |
| Website cần cập nhật | Phase 2 Completed/In progress; không công bố metrics smoke như kết quả khoa học. |

### Phase 3 - Pipeline tiền xử lý chống leakage

| Mục | Nội dung |
|---|---|
| Phân loại | Must |
| Mục tiêu | Triển khai preprocessing gần paper nhưng fit chỉ trên training fold. |
| Phạm vi | Mean/mode imputation, WOE, VIF, unseen category, scaling, nested CV, fold persistence. |
| Đầu vào | Phase 2 loader/split/artifact. |
| Đầu ra | Protocol A leakage-safe và nested CV engine. |
| Nhiệm vụ chi tiết | Implement imputer train-only; WOE smoothing; unseen-category mapping; iterative VIF removal; scaling train-only cho MLP; persist fold definitions; log selected features. |
| File dự kiến tạo/sửa | `src/creditrep/preprocessing/`, `src/creditrep/splitting/nested.py`, `configs/protocols/protocol_a.yaml`, tests leakage. |
| Test | WOE toy cases, unseen category, VIF removal order, no target leakage, no train/test overlap, AC categorical metadata. |
| Acceptance criteria | Mọi transformer fit chỉ trên training fold; selected features lưu theo fold; test leakage pass. |
| Dependency | Phase 2. |
| Rủi ro | WOE smoothing và VIF order không được paper mô tả đủ; AC đã source-imputed; VIF có thể không ổn định. |
| Go/no-go | Go Phase 4/5 khi Protocol A frozen. |
| Ngoài phạm vi | Model grid đầy đủ, metric business, modern Protocol B. |
| Commit đề xuất | `feat(preprocessing): add leakage-safe replication protocol` |
| Tag đề xuất | `p3-leakage-safe-preprocessing` |
| Website cần cập nhật | Cập nhật phương pháp preprocessing và trạng thái Phase 3. |

### Phase 4 - Xác minh metric và đánh giá kinh doanh

| Mục | Nội dung |
|---|---|
| Phân loại | Must cho AUC, Brier Score, Partial Gini; Should cho EMP nếu exact chưa chốt |
| Mục tiêu | Xác minh metric trước khi chạy core experiment quy mô lớn. |
| Phạm vi | AUC, Brier Score, Partial Gini, EMP exact hoặc approximate, unit tests và reference tests. |
| Đầu vào | Paper, Lessmann/Verbraken references cần đọc ở phase triển khai, toy predictions, artifact contract. |
| Đầu ra | Metrics module đã test, metric docs, quyết định exact/approximate cho EMP. |
| Nhiệm vụ chi tiết | Implement AUC/Brier; xác định công thức Partial Gini với `b=0.4`; xác định EMP exact/approximate; viết toy/reference tests; ghi rõ sign convention và tham số business. |
| File dự kiến tạo/sửa | `src/creditrep/metrics/`, `tests/test_metrics_*.py`, docs metric nếu cần. |
| Test | Perfect/random/worst classifier toy, Brier hand-computed, Partial Gini reference, EMP toy/reference, threshold không chọn trên test fold. |
| Acceptance criteria | Metric deterministic, finite, documented; Partial Gini pass reference; EMP không bị gọi là exact nếu chỉ approximate. |
| Dependency | Phase 2, có thể song song sau Phase 3 nhưng phải hoàn thành trước Phase 7. |
| Rủi ro | Partial Gini formula ambiguous, EMP thiếu tham số, metric profit nhạy với imbalance. |
| Go/no-go | Go Phase 7 khi AUC/Brier/Partial Gini pass; EMP được label rõ. |
| Ngoài phạm vi | Chạy full experiment, statistical comparison. |
| Commit đề xuất | `feat(metrics): validate replication and business metrics` |
| Tag đề xuất | `p4-metric-validation` |
| Website cần cập nhật | Trang phương pháp ghi metric đã xác minh và EMP exact/approximate status. |

### Phase 5 - Mô hình truyền thống và ensemble

| Mục | Nội dung |
|---|---|
| Phân loại | Must |
| Mục tiêu | Triển khai LR, Decision Tree, RF và XGBoost cho replication core. |
| Phạm vi | Model factory, tuning strategy, probability output, C4.5/CART decision, runtime logging. |
| Đầu vào | Phase 3 preprocessing, Phase 4 metric module. |
| Đầu ra | Model wrappers và configs reduced/paper-reference. |
| Nhiệm vụ chi tiết | Implement model factory; LR/RF/XGBoost/DT wrappers; map grid paper và reduced grid; enforce `predict_proba`; log hyperparameters; chốt C4.5 thật hay CART approximation. |
| File dự kiến tạo/sửa | `src/creditrep/models/`, `src/creditrep/tuning/`, `configs/models/*.yaml`, tests. |
| Test | Model factory, invalid config, probability shape, seed determinism, reduced grid count, tuning không chạm test fold. |
| Acceptance criteria | GC/TC nested-lite pass với LR/RF/XGBoost/DT decision; nếu CART thì không gọi là C4.5. |
| Dependency | Phase 3 và Phase 4. |
| Rủi ro | C4.5 library không ổn định, XGBoost version drift, RF/XGBoost tốn compute trên GMC. |
| Go/no-go | Go Phase 7 classical khi reduced runs pass. |
| Ngoài phạm vi | MLP, CatBoost, TabNet, FT-Transformer. |
| Commit đề xuất | `feat(models): add classical and ensemble replication models` |
| Tag đề xuất | `p5-classical-replication` |
| Website cần cập nhật | Cập nhật mô hình core đã triển khai; không công bố kết quả nếu chưa qua Phase 7. |

### Phase 6 - Tái lập ảnh hưởng độ sâu của MLP

| Mục | Nội dung |
|---|---|
| Phân loại | Must cho RQ2, budget có thể reduced |
| Mục tiêu | Kiểm tra MLP-3 và MLP-5 có cải thiện so với MLP-1 hay không. |
| Phạm vi | MLP-1, MLP-3, MLP-5, framework, seed, scaling, early stopping, tuning budget. |
| Đầu vào | Phase 3 preprocessing, Phase 4 metrics, Phase 5 model patterns. |
| Đầu ra | MLP wrappers, configs, training logs, prediction artifacts. |
| Nhiệm vụ chi tiết | Chọn PyTorch hoặc framework khác; implement depth builder; train loop; early stopping; deterministic settings; scaling train-only; checkpoint policy. |
| File dự kiến tạo/sửa | `src/creditrep/models/mlp.py`, `src/creditrep/models/torch_utils.py`, `configs/models/mlp_*.yaml`, tests. |
| Test | Tiny deterministic run, probability output, CPU fallback, scaling train-only, early stopping artifact. |
| Acceptance criteria | MLP-1/3/5 smoke pass trên GC; reduced run trên GC/TC pass; runtime được log. |
| Dependency | Phase 3, Phase 4 và Phase 5; Phase 6 phụ thuộc Phase 5 để tái sử dụng model factory, tuning contract, artifact contract và các quy ước logging đã được ổn định cho nhóm mô hình core. Phase 8 phụ thuộc riêng vào Phase 5 vì cần XGBoost baseline. |
| Rủi ro | Seed variance, overfitting dataset nhỏ, GPU/CUDA determinism, compute tăng mạnh. |
| Go/no-go | Go Phase 7 MLP khi runtime acceptable. |
| Ngoài phạm vi | DBN, TabNet, FT-Transformer. |
| Commit đề xuất | `feat(models): add mlp depth replication models` |
| Tag đề xuất | `p6-mlp-depth-replication` |
| Website cần cập nhật | Cập nhật trạng thái MLP depth implementation. |

### Phase 7 - Chạy thực nghiệm replication cốt lõi

| Mục | Nội dung |
|---|---|
| Phân loại | Must |
| Mục tiêu | Chạy core replication trên 6 dataset public đủ điều kiện. |
| Phạm vi | AC, GC, HMEQ, TH02, TC, GMC; Protocol A; LR, DT/CART, RF, XGBoost, MLP-1/3/5; fold-level artifacts; resume; timeout. |
| Đầu vào | Phase 3, Phase 4, Phase 5, Phase 6. |
| Đầu ra | Fold-level predictions, metrics, configs, runtime, failure logs. |
| Nhiệm vụ chi tiết | Chạy dry run; smoke run; reduced-budget run; full approved run nếu đủ resource; validate incremental result; resume failed folds; chạy GMC sau cùng. |
| File dự kiến tạo/sửa | `configs/experiments/core_*.yaml`, result artifacts trong `results/`, aggregation scripts nếu cần. |
| Test | Dry-run test, resume fixture, failure artifact, metric finite, split hash consistency, no duplicate fold. |
| Acceptance criteria | Mỗi dataset-model Must có artifact hợp lệ hoặc failure status rõ; no result fake; HMEQ caveat đi kèm mọi report. |
| Dependency | Phase 7 phụ thuộc Phase 3, Phase 4, Phase 5 và Phase 6. Không bắt đầu core replication nếu preprocessing, metrics, classical models hoặc MLP depth models chưa đạt acceptance criteria. |
| Rủi ro | Compute lớn, disk lớn, GMC timeout, một fold lỗi làm aggregation sai. |
| Go/no-go | Go Phase 9 nếu core reduced/full approved artifacts pass validation. |
| Ngoài phạm vi | Modern reassessment, robustness mở rộng, final public report. |
| Commit đề xuất | `results(core): add approved core replication artifacts` |
| Tag đề xuất | `p7-core-experiment-results` |
| Website cần cập nhật | Chỉ công bố kết quả aggregated đã validate và đi qua public result exporter; cập nhật tiến độ core run. |

### Phase 8 - Đánh giá lại bằng mô hình hiện đại

| Mục | Nội dung |
|---|---|
| Phân loại | CatBoost là Must; TabNet là Should hoặc Optional theo resource checkpoint; FT-Transformer là Optional theo resource checkpoint |
| Mục tiêu | Kiểm tra model tabular hiện đại có thay đổi kết luận ưu tiên XGBoost hay không. |
| Phạm vi | CatBoost, TabNet, FT-Transformer; Protocol A và Protocol B tách biệt. |
| Đầu vào | Phase 3 preprocessing, Phase 4 metrics, Phase 5 XGBoost baseline. |
| Đầu ra | Modern reassessment artifacts và go/no-go theo model. |
| Nhiệm vụ chi tiết | Implement CatBoost trước và xem đây là minimum evidence cho RQ3; tách Protocol A/B; benchmark runtime; quyết định TabNet; quyết định FT-Transformer; ưu tiên TC/GMC cho deep tabular; nếu TabNet hoặc FT-Transformer no-go thì ghi rõ lý do và không coi là lỗi core scope. |
| File dự kiến tạo/sửa | `src/creditrep/models/catboost_model.py`, `tabnet_model.py`, `ft_transformer.py`, configs Protocol B, tests. |
| Test | Native categorical routing, protocol separation, probability output, GPU/CPU config validation. |
| Acceptance criteria | CatBoost có kết quả hợp lệ trên cùng dataset, cùng metric và cùng protocol so sánh phù hợp với XGBoost baseline; kết quả Protocol A/B tách riêng; TabNet/FT chỉ chạy nếu checkpoint resource pass; no-go của TabNet/FT có lý do rõ; không trộn modern với replication core. |
| Dependency | Phase 3, Phase 4, Phase 5. |
| Rủi ro | Dependency nặng, GPU thiếu, overfitting dataset nhỏ, so sánh không công bằng nếu protocol bị trộn. |
| Go/no-go | Go CatBoost sau baseline XGBoost; go TabNet/FT sau resource checkpoint. |
| Ngoài phạm vi | Thay đổi core replication conclusions trước statistical analysis. |
| Commit đề xuất | `feat(models): add modern tabular reassessment models` |
| Tag đề xuất | `p8-modern-reassessment` |
| Website cần cập nhật | Cập nhật trang phương pháp và kết quả modern chỉ sau validation. |

### Phase 9 - So sánh thống kê

| Mục | Nội dung |
|---|---|
| Phân loại | Must |
| Mục tiêu | So sánh mô hình đúng đơn vị thống kê, không pseudo-replicate fold như dataset độc lập. |
| Phạm vi | Average rank, Friedman, Rom, Nemenyi, Bayesian signed-rank, ROPE, posterior probability, posterior odds. |
| Đầu vào | Aggregated metrics từ Phase 7 và Phase 8. |
| Đầu ra | Tables tương ứng Table 4, Table 5, Table 6 adaptation; statistical report. |
| Nhiệm vụ chi tiết | Aggregate per dataset/model; compute rank; implement Friedman/Rom; Nemenyi cho MLP depth; Bayesian signed-rank với ROPE; report 6-dataset limitation. |
| File dự kiến tạo/sửa | `src/creditrep/stats/`, `src/creditrep/aggregation/`, report scripts, tests. |
| Test | Toy rank table, known Friedman/Rom example, Bayesian deterministic seed, no fold pseudo-replication default. |
| Acceptance criteria | Statistical tables reproducible từ artifacts; replication và modern reassessment tách riêng; limitation power ghi rõ. |
| Dependency | Phase 7 là bắt buộc; Phase 7 và Phase 8 đều bắt buộc. Phase 9 phải tạo các phân tích riêng cho replication core và modern reassessment. |
| Rủi ro | Chỉ có 6 dataset, implementation Rom/Bayesian sai, ties/ranks ambiguity. |
| Go/no-go | Go Phase 10 khi tables regenerate được. |
| Ngoài phạm vi | Chạy thêm model để “cứu” kết quả sau khi thấy thống kê. |
| Commit đề xuất | `feat(stats): add replication statistical comparison` |
| Tag đề xuất | `p9-statistical-comparison` |
| Website cần cập nhật | Công bố bảng thống kê đã validate và diễn giải giới hạn. |

### Phase 10 - Robustness và sensitivity analysis

| Mục | Nội dung |
|---|---|
| Phân loại | Should, một số kiểm tra là Must nếu ảnh hưởng kết luận |
| Mục tiêu | Kiểm tra độ ổn định của kết luận với seed, dataset caveat, preprocessing và protocol. |
| Phạm vi | Nhiều seed, HMEQ include/exclude, GMC influence, Protocol A/B, có/không VIF, class imbalance, calibration, fold stability. |
| Đầu vào | Phase 7, Phase 8, Phase 9. |
| Đầu ra | Robustness tables, sensitivity appendix, updated deviation register. |
| Nhiệm vụ chi tiết | Chạy hoặc aggregate sensitivity Must; so sánh with/without HMEQ; kiểm tra GMC influence; seed sensitivity; Protocol A/B; calibration/Brier sanity. |
| File dự kiến tạo/sửa | `configs/experiments/robustness_*.yaml`, `src/creditrep/aggregation/sensitivity.py`, reports. |
| Test | Sensitivity fixture, paired differences, missing-run handling, stable aggregation. |
| Acceptance criteria | HMEQ/GMC/seed sensitivity có kết luận hoặc no-go documented; không làm mờ câu hỏi nghiên cứu chính. |
| Dependency | Phase 9. |
| Rủi ro | Compute tăng mạnh, quá nhiều sensitivity gây scope creep. |
| Go/no-go | Go Phase 11 nếu sensitivity Must không phát hiện blocker chưa xử lý. |
| Ngoài phạm vi | Thêm model mới ngoài scope. |
| Commit đề xuất | `test(results): add robustness and sensitivity analyses` |
| Tag đề xuất | `p10-robustness-analysis` |
| Website cần cập nhật | Cập nhật robustness summary và caveat kết luận. |

### Phase 11 - Báo cáo cuối, website kết quả và reproducibility package

| Mục | Nội dung |
|---|---|
| Phân loại | Must |
| Mục tiêu | Đóng gói kết quả cuối, cập nhật website từ trạng thái đang thực hiện sang kết quả hoàn chỉnh. |
| Phạm vi | Final report, tables, figures, deviation register, reproducibility guide, public sanitized artifacts, website result pages. |
| Đầu vào | Phase 7 đến Phase 10 artifacts đã validate. |
| Đầu ra | Final report, website kết quả, reproducibility package, deployment production cuối. |
| Nhiệm vụ chi tiết | Generate tables/figures; viết final report; viết deviation register; viết reproducibility guide; chạy public result exporter để validate/sanitize artifacts; publish sanitized artifacts; update website; deploy qua CI/CD; verify clean clone. |
| File dự kiến tạo/sửa | `docs/FINAL_REPLICATION_REPORT.md`, `docs/DEVIATION_REGISTER.md`, `docs/REPRODUCIBILITY.md`, `results/aggregated/`, `results/figures/`, `website/content/`, `website/public/results/`. |
| Test | Report generation smoke, public artifact schema, forbidden fields, no row-level prediction, manifest integrity, website build, Docker build, deployment smoke, no raw data, no secret, broken-link check. |
| Acceptance criteria | Final report và website trả lời RQ1/RQ2/RQ3; no full replication claim; no overclaim; website version traceable to commit/tag; clean clone rebuild được. |
| Dependency | Phase 10. |
| Rủi ro | Website công bố artifact nhạy cảm, report trộn Protocol A/B, result không traceable. |
| Go/no-go | Done nếu tiêu chí hoàn thành toàn dự án pass. |
| Ngoài phạm vi | Thêm experiment mới sau khi final report frozen. |
| Commit đề xuất | `docs(results): add final reproducibility package` |
| Tag đề xuất | `p11-final-reproducibility-package` |
| Website cần cập nhật | Công bố kết quả cuối đã kiểm chứng, tables, figures, limitations và reproducibility guide. |

## 8. Kiến trúc module website

| Module | Trách nhiệm | Input | Output | Test |
|---|---|---|---|---|
| Content loader | Đọc Markdown/YAML/JSON source-controlled | `website/content/*` | Typed content objects | Schema validation |
| Trang chủ | Giới thiệu đề tài, trạng thái hiện tại | Project metadata | Public page | Responsive smoke |
| Trang hoặc khu vực giới thiệu đề tài thực tập | Trình bày tên đề tài, mô tả ngắn, mục tiêu thực tập, phạm vi, sinh viên thực hiện, người hướng dẫn, đơn vị/chương trình, thời gian, repository và vai trò của website | `website/content/project.yaml`; thông tin chưa có phải là `Chưa cập nhật` hoặc `Sẽ bổ sung` | Public page hoặc section trong trang giới thiệu | Content schema, no private information |
| Trang giới thiệu | Bối cảnh, paper gốc, partial replication | README/paper summary curated | Public page | Content review |
| Trang câu hỏi nghiên cứu | RQ1, RQ2, RQ3 | Plan content | Public page | Snapshot/unit |
| Trang dữ liệu | 6 dataset public, shape, class, caveat | Generated hoặc checked derivative từ `data/datasets.yaml`, có link tới `docs/data-cards/` | Public page | Consistency check với registry, no raw data check |
| Trang phương pháp | Models, Protocol A/B, metrics, stats | Plan/method content | Public page | Content schema |
| Trang tiến độ | Phase status, milestone, tag, update date | `progress.yaml` | Public page | Progress validation |
| Trang kết quả | Placeholder ban đầu, sau này tables/figures | Sanitized artifacts | Public page | No fake result check |
| Trang tái lập | Environment, dataset preparation, deviations | Docs curated | Public page | Link check |
| Build/version banner | Hiển thị commit/tag đang deploy | CI metadata | Public version | E2E smoke |
| Public result exporter | Validate, sanitize và copy kết quả public hợp lệ sang `website/public/results/`; tạo manifest; fail nếu phát hiện trường bị cấm | Aggregated internal results | Public artifacts + manifest | Forbidden fields, no row-level prediction, manifest integrity |

## 9. Kiến trúc module thực nghiệm

| Module | Trách nhiệm | Input | Output | Test |
|---|---|---|---|---|
| Dataset loading | Đọc registry và active file | `data/datasets.yaml` | DataFrame + metadata | Loader tests |
| Schema validation | Verify schema/checksum trước run | DataFrame, registry | Validation report | Bad schema tests |
| Target normalization | Map target về 0/1 | Series, mapping | Binary target | Mapping tests |
| Split generation | Holdout/k-fold/Nx2/nested-lite | Target, seed | Fold definitions | Determinism tests |
| Preprocessing | Impute, WOE, VIF, scaling train-only | Fold data | Transformed data | Leakage tests |
| Model factory | Tạo estimator theo config | Model config | Estimator wrapper | `predict_proba` tests |
| Hyperparameter search | Inner CV hoặc fixed/reduced grid | Model, grid | Best params | No test-fold access |
| Metrics | AUC, Brier, Partial Gini, EMP | y_true, y_prob | Metrics dict | Toy/reference tests |
| Artifact storage | Lưu config, split, prediction, metrics | Run objects | JSON/CSV/Parquet | Schema tests |
| Aggregation | Fold-level sang dataset-level | Raw metrics | Aggregated tables | Fixture tests |
| Statistical analysis | Rank, Friedman, Rom, Bayesian | Aggregated metrics | Statistical tables | Known examples |
| Reporting | Sinh Markdown/HTML reports | Aggregated outputs | Reports | Regeneration smoke |

## 10. Kiến trúc CI/CD

### 10.1. CI workflow

Trigger:

- Pull request.
- Push vào `main`.

Kiểm tra dự kiến:

- Python Phase 0 tests và registry/config validation không cần raw data hoặc skip rõ.
- Website lint.
- TypeScript type check.
- Website unit/component tests nếu có.
- Website production build.
- Docker build.
- Secret scan.
- No raw data committed.
- No raw data in Docker context/image.
- Markdown/link check nếu phù hợp.

### 10.2. Production deployment workflow

**Đề xuất mặc định:** CI chạy trên mọi pull request và mọi push vào `main`, nhưng production deployment chỉ tự động chạy khi push vào `main` có thay đổi liên quan trực tiếp tới website hoặc hạ tầng công bố. Cách này tránh deploy website không cần thiết khi commit chỉ thay đổi `src/creditrep/**`, `configs/**`, `tests/**` hoặc `scripts/**` phục vụ pipeline thực nghiệm.

Path filter production deployment tối thiểu:

- `website/**`
- `deploy/**`
- `docker-compose.prod.yml`
- `.github/workflows/deploy-production.yml`
- `website/public/results/**`
- các file content website được khai báo rõ, ví dụ `website/content/**`

Workflow vẫn phải hỗ trợ `workflow_dispatch` để redeploy và rollback thủ công. Tag milestone dùng để truy vết release quan trọng, nhưng không bắt buộc mọi tag đều tự deploy nếu không có thay đổi public content.

Luồng đề xuất:

1. GitHub Actions build image.
2. Tag image bằng Git commit SHA; `latest` hoặc `stable` chỉ là alias.
3. Push image lên GitHub Container Registry (GHCR) mặc định.
4. SSH tới Google Cloud VM bằng GitHub Actions Secrets.
5. Pull image theo SHA.
6. `docker compose up -d`.
7. Health check.
8. Rollback về image trước nếu health check fail.
9. Ghi deployment log.
10. Concurrency control để không deploy song song.

**GHCR vs Google Artifact Registry:** GHCR đơn giản hơn vì tích hợp trực tiếp GitHub Actions và phù hợp phase đầu. Google Artifact Registry hợp lý hơn nếu muốn chuẩn hóa toàn bộ artifact trong Google Cloud, nhưng cần thêm IAM/service account và chi phí vận hành. Mặc định chọn GHCR cho Phase 1.

## 11. Kiến trúc triển khai Google Cloud VM

| Thành phần | Đề xuất |
|---|---|
| VM | Ubuntu trên Google Cloud VM |
| Runtime | Docker Engine và Docker Compose |
| Truy cập website | Public IP, hostname hoặc domain |
| Giao thức bắt buộc | HTTP đủ cho phạm vi đề tài thực tập |
| HTTPS | Optional; bổ sung nếu có domain hoặc hostname phù hợp |
| Chứng chỉ | Let’s Encrypt nếu triển khai HTTPS |
| Reverse proxy | Nginx được khuyến nghị nhưng không phụ thuộc HTTPS |
| Network | Chỉ mở SSH, HTTP và HTTPS khi cần |
| Secret | Lưu trong GitHub Actions Secrets và file production env trên VM, không commit |
| Health check | Endpoint HTTP đơn giản của website hoặc Nginx route |
| Logging | Docker logs + Nginx logs, có log rotation |
| Rollback | Giữ image version trước, rollback bằng compose |
| Backup | Backup Nginx config, compose file, env production không chứa trong repo |

Ràng buộc vận hành:

- Website production không dùng để chạy training job nặng.
- Pipeline thực nghiệm chạy local hoặc compute environment riêng.
- Không mount raw dataset vào container website.
- Không copy raw dataset vào Docker build context.
- Public artifact phải là sanitized aggregated artifact.

Điều kiện truy cập, domain và HTTPS:

- Website có thể sử dụng public IP và HTTP trong toàn bộ thời gian thực tập.
- Website production chỉ cần truy cập được từ Internet bằng public IP, hostname tạm, subdomain hoặc domain riêng để hoàn thành Phase 1.
- Domain riêng, subdomain riêng, HTTPS, Let’s Encrypt và chứng chỉ SSL/TLS là Optional/Should, triển khai khi có sẵn domain/hostname phù hợp hoặc khi còn thời gian.
- Let’s Encrypt chỉ áp dụng nếu có domain hoặc hostname phù hợp trỏ về VM.
- Không có domain hoặc HTTPS không ảnh hưởng tới tiến độ thực nghiệm, không làm P1C bị Blocked và không ngăn tạo tag `p1-website-cicd-foundation`.
- Nếu sau này website có authentication, admin page, form nhập mật khẩu hoặc chức năng truyền dữ liệu nhạy cảm thì HTTPS phải được nâng thành Must trước khi đưa chức năng đó vào sử dụng.

Lưu ý bảo mật khi dùng HTTP/public IP:

- Website Phase 1 là website công khai dạng static-first.
- Website không có đăng nhập, form nhập mật khẩu hoặc chức năng truyền dữ liệu nhạy cảm.
- Không gửi secret, credential hoặc dữ liệu cá nhân qua HTTP.
- Không đưa raw data, processed data, train/test indices chi tiết hoặc prediction cấp bản ghi lên website.
- Trong phạm vi website chỉ đọc của đề tài thực tập, HTTP/public IP được chấp nhận.

## 12. Hợp đồng artifact và hợp đồng kết quả công khai

### 12.1. Artifact thực nghiệm nội bộ

| Artifact | Định dạng | Nội dung |
|---|---|---|
| Run metadata | JSON | experiment ID, run ID, timestamp, Git commit/tag, dirty state, command |
| Dataset metadata | JSON | dataset ID, checksum, target mapping, caveat |
| Fold definition | JSON/CSV | fold ID, split seed, split hash, train/test summary |
| Predictions | CSV/Parquet | y_true, y_prob, fold ID; không public mặc định |
| Metrics | CSV/JSON | fold metrics và summary |
| Runtime | JSON | duration, device, n_jobs, warnings |
| Environment | JSON | Python, package versions, OS, CPU/GPU |

### 12.2. Artifact công khai

| Artifact public | Điều kiện công bố |
|---|---|
| Metric summary | Aggregated, validated, không có row-level prediction |
| Rank/statistical tables | Reproducible từ artifacts nội bộ |
| Figures | Không chứa dữ liệu nhạy cảm |
| Deviation register | Có review |
| Reproducibility guide | Không chứa secret/local path |

Không lưu mọi trained model nếu không cần. Nếu cần lưu selected model, phải có config `save_model: true` và không công bố public mặc định.

### 12.3. Public result exporter

`public result exporter` là module bắt buộc trước khi đưa kết quả lên website.

Trách nhiệm:

- Đọc aggregated internal results.
- Validate schema.
- Sanitize các trường không được công bố.
- Copy output hợp lệ sang `website/public/results/`.
- Tạo manifest public artifact.
- Fail nếu phát hiện raw data, row-level prediction, train/test index, local path, secret, internal exception log hoặc model checkpoint chưa được duyệt.

Test bắt buộc:

- Public artifact schema.
- Forbidden fields.
- No row-level prediction.
- Manifest integrity.

## 13. Chiến lược kiểm thử

| Nhóm test | Nội dung | Nơi chạy |
|---|---|---|
| Python unit tests | Target mapping, registry, WOE, VIF, metrics, split, config, artifact serialization | CI |
| Python integration tests | Fixture nhỏ, preprocessing + model + prediction, nested CV mini, resume/restart | CI hoặc local |
| Data integration tests | Verify raw data khi local có file | Local, skip rõ trong CI |
| Scientific regression tests | Metric toy cases, stable split hash, deterministic fake model, rank/stat fixture | CI |
| Website tests | Lint, TypeScript, unit/component, content schema, progress validation, internship project content validation | CI |
| Website build tests | Production build, Docker build, no raw data in image | CI |
| Security checks | No secret, no raw data, no local path public | CI |
| Deployment tests | Health check, deployment smoke, rollback test | Deployment workflow/production |
| UX smoke | Accessibility smoke, responsive smoke, broken-link check | CI hoặc local |
| Dataset content consistency | Website dataset derivative khớp `data/datasets.yaml` về dataset ID, shape, target, default rate, caveat và usable status | CI |
| Public result exporter tests | Public artifact schema, forbidden fields, no row-level prediction, manifest integrity | CI |

## 14. Ma trận thực nghiệm sơ bộ

| Scope | Dataset | Model | Protocol | CV | Seed | Tuning | Priority |
|---|---|---|---|---|---:|---|---|
| Smoke | GC | LR | Minimal/A | 3-fold stratified | 1 | none | Must |
| Smoke | GC | XGBoost | Minimal/A | 3-fold stratified | 1 | tiny grid | Must |
| Smoke | TC | LR | Minimal/A | holdout hoặc 3-fold reduced | 1 | none | Must |
| Preprocessing | GC/fixture | LR/XGBoost | A | nested-lite | 1 | tiny | Must |
| Classical core | AC, GC, HMEQ, TH02, TC, GMC | LR, DT/CART, RF, XGBoost | A | paper `N x 2` hoặc approved reduced | 1 | reduced/paper reference | Must |
| MLP depth | 6 public nếu đủ compute | MLP-1, MLP-3, MLP-5 | A | approved | 1 | reduced | Must |
| Modern tối thiểu | 6 public hoặc cùng tập dataset với XGBoost core đã chốt | CatBoost | A/B separate | approved | 1-3 | fixed/Optuna | Must |
| Modern deep | TC/GMC ưu tiên | TabNet | A/B separate | reduced | 3 | fixed | Should hoặc Optional theo resource checkpoint |
| Modern deep | TC/GMC ưu tiên | FT-Transformer | A/B separate | reduced | 3 | fixed | Optional theo resource checkpoint |
| Robustness | Selected | Core + CatBoost | sensitivity | selected | 3+ | fixed | Should |

## 15. Kế hoạch tài nguyên

### 15.1. Website production

| Tài nguyên | Kế hoạch |
|---|---|
| CPU/RAM | VM nhỏ đến trung bình, đủ phục vụ website static/SSR nhẹ |
| Disk | Dành cho Docker images, logs, Nginx, certs; cần log rotation |
| Network | HTTP public là đủ; HTTPS optional; SSH hạn chế |
| Docker image | Image website nhỏ, không chứa raw data |
| HTTPS | Optional; dùng Let’s Encrypt auto-renewal nếu triển khai HTTPS |
| Chi phí | Duy trì VM, disk, outbound traffic; cần theo dõi |

### 15.2. Experiment compute

| Tài nguyên | Kế hoạch |
|---|---|
| CPU | Bắt buộc cho classical, XGBoost, CatBoost CPU |
| RAM | Phụ thuộc GMC và nested CV |
| GPU | Không bắt buộc cho core; khuyến nghị cho MLP/TabNet/FT-Transformer |
| Runtime | Benchmark theo checkpoint, không ước lượng giờ giả |
| Artifact storage | Có thể lớn; predictions/metrics ưu tiên hơn trained models |

Website VM không được giả định là nơi chạy toàn bộ training workload.

## 16. Đối chiếu câu hỏi nghiên cứu

| RQ | Dataset | Model | Protocol | Metric | Statistical test | Minimum evidence | Giới hạn diễn giải |
|---|---|---|---|---|---|---|---|
| RQ1: Kết quả chính của paper có tái xuất hiện trên 6 dataset public không? | AC, GC, HMEQ, TH02, TC, GMC | LR, DT/CART, RF, XGBoost, MLP-1/3/5 | A | AUC, Brier, Partial Gini, EMP nếu validated | Average rank, Friedman/Rom, Bayesian signed-rank | Fold-level artifacts và dataset aggregates | Chỉ 6/10 dataset; HMEQ caveat; không full replication |
| RQ2: MLP nhiều lớp có tiếp tục không tốt hơn MLP một lớp không? | 6 public nếu đủ compute | MLP-1, MLP-3, MLP-5 | A | AUC, Brier, Partial Gini, EMP nếu có | Nemenyi, Bayesian signed-rank | Cùng preprocessing, cùng budget policy | Seed variance, reduced grid là deviation |
| RQ3: Các mô hình tabular hiện đại, tối thiểu là CatBoost và tùy điều kiện tài nguyên gồm TabNet hoặc FT-Transformer, có làm thay đổi kết luận ưu tiên XGBoost không? | CatBoost: cùng dataset với XGBoost core đã chốt, ưu tiên 6 public; TabNet/FT: TC/GMC ưu tiên nếu resource pass | XGBoost và CatBoost là minimum evidence; TabNet/FT là mở rộng có điều kiện | A và B tách riêng | AUC, Brier, Partial Gini, EMP nếu có | Rank/Bayesian trong từng protocol | XGBoost, CatBoost, cùng dataset, cùng metric, cùng protocol so sánh phù hợp | Protocol B không được trộn với replication; TabNet/FT no-go có lý do không phải lỗi core scope |

## 17. Nhật ký quyết định

| Quyết định | Lựa chọn | Ưu điểm | Nhược điểm | Đề xuất mặc định | Chốt ở phase | Phase ảnh hưởng |
|---|---|---|---|---|---|---|
| Next.js hay static site generator khác? | Next.js / Astro / Hugo | Next.js phổ biến, TypeScript tốt; Astro nhẹ; Hugo rất nhanh | Next.js phức tạp hơn static thuần | Next.js static-first | P1 | P1, P11 |
| Static-first hay backend? | Static-first / backend API | Đơn giản, ít vận hành | Ít dynamic | Static-first, không backend P1 | P1 | P1 |
| Thư mục website | `website/` / khác | Tách rõ khỏi Python | Thêm workspace | `website/` | P1 | P1 |
| GHCR hay Google Artifact Registry? | GHCR / GAR | GHCR đơn giản với GitHub Actions; GAR tích hợp GCP | GAR cần IAM thêm | GHCR ở P1 | P1 | P1 |
| Chính sách deploy production | Main có path filter / tag / manual | Path filter tránh deploy website khi chỉ đổi code thực nghiệm; tag ổn định; manual an toàn | Tag/manual chậm hơn | Main sau CI pass nhưng chỉ khi path filter public/deploy match; có workflow_dispatch redeploy/rollback | P1 | P1 |
| Nginx hay Caddy? | Nginx / Caddy | Nginx quen thuộc; Caddy có HTTPS tự động nếu có domain | Nginx cần thêm certbot nếu triển khai HTTPS; Caddy là thêm lựa chọn vận hành | Nginx được khuyến nghị; HTTPS không bắt buộc cho Phase 1 | P1 | P1 |
| Domain và HTTPS | Production truy cập bằng public IP, hostname tạm, subdomain hoặc domain; domain riêng Optional; HTTPS Optional/Should; Let’s Encrypt chỉ khi có domain/hostname phù hợp | Public IP + HTTP triển khai nhanh, không cần mua hoặc cấu hình domain, đủ cho website báo cáo thực tập, không làm chậm Phase 2; domain + HTTPS chuyên nghiệp hơn, URL dễ nhớ và kết nối được mã hóa | Public IP + HTTP có URL khó nhớ, trình duyệt không hiển thị kết nối bảo mật, không phù hợp nếu sau này có authentication hoặc dữ liệu nhạy cảm; domain + HTTPS cần DNS/certificate và tăng vận hành | Dùng public IP hoặc hostname sẵn có để hoàn thành Phase 1; bổ sung domain/HTTPS sau nếu thuận tiện; điều kiện hoàn thành Phase 1 và tag P1 không phụ thuộc domain hoặc HTTPS | P1C | P1, P11 |
| Một VM hay tách compute? | Một VM website / compute riêng | Website VM đơn giản | Không chạy training nặng | Website VM riêng cho web, compute thực nghiệm riêng | P1 | P1-P11 |
| Publish sanitized artifacts | Copy vào `website/public/results/` / fetch remote | Đơn giản | Cần review thủ công/CI | Public sanitized artifacts source-controlled hoặc release artifacts | P11 | P7-P11 |
| Source of truth dataset website | Sinh từ `data/datasets.yaml` / nhập tay / checked derivative | Sinh từ registry tránh lệch metadata; checked derivative linh hoạt hơn | Generated flow cần script/validation | `data/datasets.yaml` là source of truth; website derivative phải được CI kiểm tra | P1/P11 | P1, P7-P11 |
| Public result exporter | Bắt buộc / thủ công / bỏ qua | Bắt buộc giúp tránh lộ dữ liệu nhạy cảm | Thêm bước validation/export | Bắt buộc trước khi publish kết quả | P7-P11 | P7-P11 |
| Authentication admin? | Không / có admin | Không admin đơn giản | Cập nhật qua repo | Không admin, content source-controlled | P1 | P1 |
| Staging environment | Không / có | Có staging an toàn hơn | Tăng vận hành | Optional, chưa Must | P1 | P1 |
| Rollback policy | Giữ 1 / 3 / nhiều image | 3 image đủ an toàn | Tốn disk | Giữ tối thiểu 3 image | P1 | P1 |
| C4.5 hay CART | C4.5 / CART approximation | C4.5 gần paper | CART ổn định nhưng deviation | Chưa xác định | P5 | P5, P7, P9 |
| EMP exact hay approximate | Exact / approximate | Exact gần paper | Exact khó hơn | Chưa xác định; label rõ | P4 | P4, P7, P9 |
| Partial Gini formula | Lessmann/reference | Đúng metric core | Cần đối chiếu | Chưa xác định, `b=0.4` | P4 | P4, P7 |
| Framework MLP | PyTorch / sklearn / Keras | PyTorch linh hoạt | Thêm dependency | PyTorch nếu chấp nhận | P6 | P6, P7 |
| CatBoost priority | Must / Should | CatBoost là mở rộng hiện đại tối thiểu để trả lời RQ3 | Thêm dependency và runtime | Must | P8 | P8, P9, P11 |
| TabNet priority | Should / Optional | Có thêm deep tabular evidence | Phụ thuộc GPU, seed variance, tuning budget | Should hoặc Optional theo resource checkpoint | P8 | P8 |
| FT-Transformer priority | Optional / Should | Kiểm tra Transformer tabular hiện đại | Compute cao, tuning nhạy | Optional theo resource checkpoint | P8 | P8 |

Chi tiết quyết định về domain và HTTPS:

| Quyết định | Lựa chọn |
|---|---|
| Cách truy cập production | Public IP, hostname tạm, subdomain hoặc domain |
| Đề xuất mặc định | Public IP hoặc hostname sẵn có |
| Domain riêng | Optional |
| HTTPS | Optional/Should |
| Let’s Encrypt | Chỉ triển khai nếu có domain hoặc hostname phù hợp |
| Điều kiện hoàn thành Phase 1 | Không phụ thuộc domain hoặc HTTPS |
| Điều kiện tạo tag P1 | Không phụ thuộc domain hoặc HTTPS |

Public IP + HTTP:

- Ưu điểm: triển khai nhanh; không cần mua hoặc cấu hình domain; đủ cho website báo cáo thực tập; không làm chậm Phase 2.
- Nhược điểm: URL khó nhớ; trình duyệt không hiển thị kết nối bảo mật; không phù hợp nếu sau này có authentication hoặc dữ liệu nhạy cảm.

Domain + HTTPS:

- Ưu điểm: chuyên nghiệp hơn; URL dễ nhớ; kết nối được mã hóa.
- Nhược điểm: cần domain, DNS và cấu hình certificate; tăng thêm công việc vận hành; không cần thiết cho scope website hiện tại.

Đề xuất mặc định: dùng public IP hoặc hostname sẵn có để hoàn thành Phase 1; bổ sung domain/HTTPS sau nếu thuận tiện.

## 18. Danh mục rủi ro

| Rủi ro | Mức độ | Ảnh hưởng | Giảm thiểu | Residual |
|---|---|---|---|---|
| Raw data lọt vào Docker context | Cao | Lộ dữ liệu, image lớn | `website/.dockerignore`, no-raw-data-in-image test | Thấp |
| Raw data bị commit | Cao | Vi phạm chính sách/license | `.gitignore`, CI scan, status check | Thấp |
| Secret bị commit | Cao | Lộ production | GitHub Secrets, secret scan, không commit env prod | Thấp |
| SSH key bị lộ | Cao | Mất quyền VM | GitHub Secrets, rotate key, least privilege | Trung bình |
| Website hiển thị local path | Trung bình | Rò thông tin máy cá nhân | Content scan | Thấp |
| Public artifact chứa prediction cấp bản ghi | Cao | Rủi ro dữ liệu nhạy cảm | Public artifact contract, review trước publish | Thấp |
| Deployment lỗi gây downtime | Trung bình | Website mất truy cập | Health check, rollback, concurrency | Thấp-trung bình |
| Không rollback được image | Trung bình | Downtime kéo dài | Immutable SHA tags, giữ 3 image | Thấp |
| VM hết dung lượng | Trung bình | Deploy fail | Log rotation, image cleanup, disk monitoring | Trung bình |
| HTTPS hết hạn nếu đã triển khai | Thấp | Website mất mã hóa hoặc cảnh báo trình duyệt, nhưng không làm hỏng phạm vi website chỉ đọc nếu HTTP fallback được chấp nhận | Auto-renewal, renewal check; nếu chưa cần HTTPS thì dùng public IP/HTTP | Thấp |
| Dependency vulnerability | Trung bình | Security risk | Dependency scanning phù hợp | Trung bình |
| Workflow PR không tin cậy truy cập secret | Cao | Lộ secret | Không expose production secrets cho PR từ fork | Thấp |
| Production deploy không có path filter | Trung bình | Website deploy không cần thiết khi chỉ đổi code thực nghiệm | Dùng path filter cho `website/**`, `deploy/**`, workflow deploy, compose và public results | Thấp |
| Không có domain riêng | Thấp | URL khó nhớ, giao diện công bố kém chuyên nghiệp hơn | Dùng public IP hoặc hostname tạm | Thấp |
| Website chỉ dùng HTTP | Thấp trong phạm vi hiện tại | Kết nối không được mã hóa | Website chỉ đọc, không có authentication hoặc dữ liệu nhạy cảm; bổ sung HTTPS nếu scope thay đổi hoặc có domain phù hợp | Thấp |
| Scope website thay đổi và có dữ liệu nhạy cảm | Cao | HTTP không còn phù hợp | Nâng HTTPS thành Must trước khi triển khai authentication, admin page, form mật khẩu hoặc chức năng truyền dữ liệu nhạy cảm | Thấp nếu kiểm soát scope |
| Nội dung dataset website lệch registry | Trung bình | Công bố sai shape/default rate/caveat | `data/datasets.yaml` là source of truth; CI consistency check | Thấp |
| Public result exporter bỏ sót trường bị cấm | Cao | Lộ prediction, path hoặc log nhạy cảm | Forbidden-field tests, manifest validation, review trước publish | Thấp |
| HMEQ provenance caveat | Trung bình | Ảnh hưởng diễn giải | Include caveat, robustness exclude-HMEQ | Trung bình |
| WOE/VIF leakage | Cao | Metric inflated | Train-only tests | Thấp |
| EMP ambiguous | Cao | Business metric sai | Exact/approx label, reference tests | Trung bình |
| 6 dataset thay vì 10 | Cao | Statistical power thấp | Không overclaim, limitation rõ | Cao |
| CatBoost dependency hoặc runtime gây chậm Phase 8 | Trung bình | RQ3 minimum evidence bị chậm | Benchmark sớm, fixed budget, CPU fallback, ghi blocker nếu dependency không cài được | Trung bình |

## 19. Milestone, commit và tag

| Phase | Milestone | Commit message đề xuất | Tag đề xuất |
|---|---|---|---|
| P0 | Xác minh dữ liệu nền tảng | `chore(data): establish phase 0 verification baseline` | `p0-data-verification-baseline` |
| P1 | Website và CI/CD | `feat(website): add project website and production delivery foundation` | `p1-website-cicd-foundation` |
| P2 | Nền tảng thực nghiệm | `feat(experiments): add foundation and smoke-test runner` | `p2-experiment-foundation` |
| P3 | Preprocessing chống leakage | `feat(preprocessing): add leakage-safe replication protocol` | `p3-leakage-safe-preprocessing` |
| P4 | Metric validation | `feat(metrics): validate replication and business metrics` | `p4-metric-validation` |
| P5 | Classical replication | `feat(models): add classical and ensemble replication models` | `p5-classical-replication` |
| P6 | MLP depth replication | `feat(models): add mlp depth replication models` | `p6-mlp-depth-replication` |
| P7 | Core experiment results | `results(core): add approved core replication artifacts` | `p7-core-experiment-results` |
| P8 | Modern reassessment | `feat(models): add modern tabular reassessment models` | `p8-modern-reassessment` |
| P9 | Statistical comparison | `feat(stats): add replication statistical comparison` | `p9-statistical-comparison` |
| P10 | Robustness analysis | `test(results): add robustness and sensitivity analyses` | `p10-robustness-analysis` |
| P11 | Final reproducibility package | `docs(results): add final reproducibility package` | `p11-final-reproducibility-package` |

## 20. Tiêu chí hoàn thành toàn dự án

- Website được deploy công khai trên Google Cloud VM.
- Website truy cập được từ Internet bằng public IP, hostname hoặc domain.
- CI/CD hoạt động từ giai đoạn đầu.
- HTTPS được triển khai nếu có domain hoặc khi phạm vi website phát sinh chức năng cần bảo mật đường truyền.
- Domain và HTTPS không phải điều kiện bắt buộc trong phạm vi website báo cáo thực tập hiện tại.
- Có health check và rollback.
- Website thể hiện tiến độ theo milestone.
- Website không công khai raw data hoặc artifact nhạy cảm.
- Kết quả trên website sinh từ sanitized aggregated artifacts.
- Version website truy vết được tới Git commit/tag.
- 6 dataset public verify pass.
- Pipeline không leakage.
- Split và seed tái tạo được.
- Metric chính được xác minh.
- Core models chạy thành công theo scope đã chốt.
- MLP depth comparison hoàn thành.
- Statistical comparison hoàn thành.
- Modern reassessment hoàn thành tối thiểu với CatBoost; TabNet và FT-Transformer hoàn thành nếu resource checkpoint pass, hoặc có no-go hợp lệ và được ghi rõ.
- Robustness Must checks hoàn thành hoặc no-go documented.
- Final report và website trả lời RQ1, RQ2, RQ3.
- Không tuyên bố full replication.
- Không overclaim từ 6 public datasets.
- Website build và deploy lại được từ clean clone.
- Không phụ thuộc đường dẫn local của một máy cụ thể.

## 21. Thứ tự triển khai khuyến nghị

1. Hoàn tất Phase 1 website và CI/CD.
   - Điều kiện chuyển tiếp: website production truy cập được từ Internet, CI/CD hoạt động, health check pass, rollback pass, no raw data/secret pass.
2. Xây nền tảng thực nghiệm và smoke test.
   - Điều kiện chuyển tiếp: GC/TC smoke artifacts hợp lệ.
3. Xây preprocessing chống leakage.
   - Điều kiện chuyển tiếp: WOE/VIF train-only tests pass.
4. Xác minh metric.
   - Điều kiện chuyển tiếp: AUC/Brier/Partial Gini pass; EMP exact/approx chốt rõ.
5. Triển khai classical và ensemble models.
   - Điều kiện chuyển tiếp: LR/RF/XGBoost/DT reduced pass.
6. Triển khai MLP depth.
   - Điều kiện chuyển tiếp: MLP-1/3/5 smoke pass.
7. Chạy core replication.
   - Điều kiện chuyển tiếp: reduced/full approved artifacts pass validation.
8. Chạy CatBoost như phần Must của modern reassessment.
   - Điều kiện chuyển tiếp: CatBoost có kết quả hợp lệ trên cùng dataset, cùng metric và cùng protocol so sánh phù hợp với XGBoost baseline.
9. Quyết định TabNet và FT-Transformer theo resource checkpoint.
   - Điều kiện chuyển tiếp: GPU/runtime đủ để chạy, hoặc ghi no-go có lý do; no-go của TabNet/FT không làm hỏng core scope nếu CatBoost đã hoàn thành.
10. So sánh thống kê.
    - Điều kiện chuyển tiếp: tables regenerate được.
11. Robustness.
    - Điều kiện chuyển tiếp: sensitivity Must không tạo blocker chưa xử lý.
12. Final report và website kết quả.
    - Điều kiện chuyển tiếp: tiêu chí hoàn thành pass.

## 22. Giai đoạn tiếp theo

**Giai đoạn tiếp theo được đề xuất:** **Phase 1 - Website giới thiệu đề tài và nền tảng CI/CD**.

Các quyết định cần chốt trước Phase 1:

- Dùng Next.js static-first trong `website/`.
- Dùng GHCR hay Google Artifact Registry.
- Deploy production theo path filter sau push vào `main`, theo tag, hay manual-only.
- Dùng Nginx hay Caddy.
- Có cần staging environment trong phạm vi thực tập không.
- Rollback giữ bao nhiêu image version.

Quyết định Optional, không chặn Phase 1:

- Có muốn sử dụng domain riêng hay không?
- Có muốn cấu hình HTTPS nếu có domain hoặc hostname phù hợp hay không?

## 23. Quyết định cần người dùng chốt

1. Có chấp nhận Phase 1 là website + CI/CD trước mọi coding pipeline thực nghiệm không?
2. Website dùng Next.js static-first trong `website/` có phù hợp không?
3. Production deployment dùng GHCR + SSH vào Google Cloud VM hay muốn Google Artifact Registry?
4. Deploy production khi push vào `main` hay chỉ deploy khi tạo tag milestone?
5. Dùng Nginx hay Caddy cho reverse proxy nếu cần?
6. Phase 1 sử dụng public IP để truy cập production có được chấp nhận không?
7. Domain riêng có được giữ ở mức Optional không?
8. HTTPS có được giữ ở mức Optional và chỉ triển khai nếu có domain phù hợp không?
9. Có cần staging environment không?
10. Có chấp nhận website không có backend/database trong Phase 1 không?
11. Public artifacts sẽ source-control trong `website/public/results/` hay dùng release artifacts?
12. Decision Tree dùng C4.5 thật hay CART approximation?
13. EMP exact hay approximate?
14. Framework MLP có dùng PyTorch không?
15. TabNet giữ Should hoặc Optional theo resource checkpoint, và FT-Transformer giữ Optional theo resource checkpoint có phù hợp không?
