# P7C.4A — Nghiên cứu quyết định MLP và kế hoạch benchmark compute

## Trạng thái và ranh giới

P7C.4A đã **Completed — benchmark plan ready for human review**. Tài liệu này chuẩn bị bằng chứng cho
`DR-P7C-03` (budget MLP-1/3) và `DR-P7C-04` (phạm vi MLP-5), nhưng không phê
duyệt hai quyết định, không khóa manifest final và không chạy benchmark hay
thực nghiệm khoa học.

Mapping canonical: `DR-P7C-03` là **Final MLP-1/3 candidate budget and search
strategy** — chọn một scenario đã định trước cho MLP-1/3, không dùng pilot
metric. `DR-P7C-04` là **MLP-5 core-scope inclusion decision** — giữ hoặc loại
MLP-5 theo scope/compute evidence, không theo predictive performance. Cả hai
đều `Open — pending human approval`, thuộc handoff P7C.4B và chưa tạo manifest.

Evidence canonical duy nhất của P7C.3 là `vm-run-003`: 60/60 fit hoàn tất,
CPU/memory/stability PASS và peak process-tree RSS **916.6640625 MiB** (xấp xỉ
916.66 MiB, không phải 916,664 MiB). Đây chỉ là
evidence engineering non-publishable; không có predictive metric nào được dùng
để chọn candidate, architecture, backend hoặc giảm grid.

## Audit protocol và workload tham chiếu

P7A (`configs/protocols/p7a/p7a_candidate_manifest.yaml`) khai báo reference
space theo Table 2: `mlp_1=144`, `mlp_3=720`, `mlp_5=2016`. Các số này là số
candidate **cho từng architecture**, không phải final budget và không nhân đôi
theo `batch_normalization` (manifest đã ghi count inconsistency). P7A có 90
outer partitions trên sáu dataset và 5 inner folds. Vì vậy full reference
workload là `(144 + 720 + 2016) × 90 × 5 = 1,296,000` inner fits, cộng `3 ×
90 = 270` outer refits; chưa gồm prediction, evaluation, I/O hay aggregation.

| Khái niệm | Ý nghĩa trong P7C.4A |
| --- | --- |
| Reference search space | Paper/P7A: 144/720/2.016; chỉ làm baseline fidelity. |
| Pilot search space | Hai candidate `low_stress`/`high_stress` mỗi depth của P7C.3; chỉ coverage engineering. |
| Proposed final scientific space | Ba scenario ở dưới; cần human review trước khi thành manifest. |
| Final approved space | Chưa tồn tại; chỉ xuất hiện sau DR-P7C-03/04 và manifest có lock. |

## So sánh chiến lược tìm kiếm

| Chiến lược | Fidelity/fairness/reproducibility | Bias và nested CV | Compute/diễn giải | Kết luận |
| --- | --- | --- | --- | --- |
| Exhaustive grid | Fidelity cao nhất; đối xử đúng reference count nhưng chi phí giữa depth không cân bằng. Deterministic. | Không sampling bias nhưng có selection noise chuẩn của inner CV. | 1.296.000 fit, không khả thi theo P7C.3; dễ giải thích. | Không chọn. |
| Curated deterministic subset | Reproducible và dễ audit; fairness phụ thuộc coverage được quy định trước. | Rủi ro curator bias và bỏ sót tổ hợp, nhưng không dùng pilot metric. | Rẻ, dễ báo cáo nếu nêu rõ rule. | Không chọn đơn lẻ: thiếu exploration có kiểm soát. |
| Seeded random search | Reproducible khi lock seed; có thể phân bổ cùng budget theo architecture. | Sampling variance/coverage miss; cần stratification và seed audit. | Chi phí bị chặn, diễn giải vừa phải. | Không chọn đơn lẻ: random thuần không bảo đảm candidate replication bắt buộc. |
| Successive halving/early-stopping search | Có thể giảm compute, nhưng fairness nhạy với resource schedule và early-stop semantics. | Loại sớm có thể ưu tiên candidate hội tụ nhanh; nested CV phức tạp hơn. | Tiết kiệm khi có harness riêng, khó giải thích như replication. | Không chọn cho final hiện tại; có thể nghiên cứu sau approval riêng. |
| Hybrid: mandatory + seeded stratified exploration | Giữ candidate replication bắt buộc; cùng rule coverage/seed cho ba depth. | Giảm curator bias so với subset thuần; vẫn có sampling bias được công bố trước. | Budget chặn, tương thích nested CV và dễ audit bằng manifest. | **Đề xuất để review**, chưa approved. |

Hybrid luôn chọn candidate từ reference grid bằng rule đã định trước: mandatory
coverage gồm biên width/regularization/learning-rate và cấu hình trung tâm cho
mỗi depth; phần còn lại sampled seeded, stratified theo các trục hyperparameter.
Không dùng kết quả P7C.3 hoặc bất kỳ performance metric nào.

## Candidate-budget scenarios

Mọi scenario dùng `inner_fits = (C1 + C3 + C5) × 90 × 5`, `outer_refits =
270`, seed gốc 42 và seed con xác định từ `(scenario_id, model_id, candidate
index)`. Candidate duy nhất, thuộc reference grid; coverage table, seed,
canonical digest và danh sách cuối phải được lock trước execution.

| Scenario | C1 / C3 / C5 | Inner fits | Outer refits | Lựa chọn, coverage và trade-off |
| --- | ---: | ---: | ---: | --- |
| Minimum viable | 12 / 24 / 24 | 27.000 | 270 | Mandatory biên + trung tâm, phần còn lại seeded stratified. Mỗi depth phải bao phủ low/high width, dropout 0 và >0, L2 0 và >0, learning-rate thấp/cao, batch-normalization nếu representation cho phép. Rủi ro coverage thưa, nhất là MLP-5. |
| Balanced / recommended | 24 / 48 / 48 | 54.000 | 270 | Cùng rule, tăng exploration trong từng stratum và yêu cầu ít nhất hai candidate khác nhau trên từng tổ hợp coverage bắt buộc. Cân bằng compute, fairness và auditability; **đề xuất chờ human review**. |
| High-fidelity | 48 / 96 / 96 | 108.000 | 270 | Cùng rule, tăng mật độ sample nhưng vẫn không phải exhaustive grid. Rủi ro vượt thời hạn nếu CPU/GPU benchmark không đạt; fidelity cao hơn nhưng MLP-5 vẫn có tỷ lệ cover reference nhỏ. |

Assumption chung: 90 outer partitions, 5 inner folds, one selected refit/model/
outer partition; retry chỉ theo policy final. Các scenario không cam kết runtime,
không cho phép suy luận GPU speedup, và không tự khóa candidate budget.

### Cấu trúc candidate có thể kiểm tra

| Scenario | Mandatory MLP-1 / 3 / 5 | Seeded exploration MLP-1 / 3 / 5 |
| --- | ---: | ---: |
| Minimum viable | 8 / 12 / 12 | 4 / 12 / 12 |
| Balanced / recommended | 12 / 18 / 18 | 12 / 30 / 30 |
| High-fidelity | 16 / 24 / 24 | 32 / 72 / 72 |

Mandatory candidate được chọn trước bằng các biên thấp/cao và điểm trung tâm
về width, dropout (`0` và `>0`), L2 (`0` và `>0`) và learning rate (thấp/cao)
trong không gian hợp lệ của từng depth. Phần exploration sampling phân tầng theo
bốn trục đó, với seed gốc `42` và seed con SHA-256 của
`(scenario_id, model_id, candidate_index)`. Candidate dùng canonical parameter
tuple, mandatory bị loại khỏi population exploration và validator phải kiểm tra
unique tuple, hai tập rời nhau, thuộc P7A reference space, đủ stratum bắt buộc,
replay đúng seed và không có metric đầu vào.

`batch_normalization` không là trục sampling: declared count P7A không nhân đôi
theo trường này; candidate list final phải giữ đúng convention đó và chỉ ghi nó
như metadata runtime. MLP-3 và MLP-5 có cùng budget để kiểm soát cơ hội search
giữa hai depth sâu, không ngầm coi reference space của chúng bằng nhau.

## Ma trận benchmark bounded

Plan machine-readable là `configs/protocols/p7c/p7c4a_mlp_compute_benchmark_plan.yaml`.
Nó không phải execution result. TC và GMC đại diện cho hai tải dữ liệu; candidate
`low_stress`/`high_stress` đã định trước cung cấp coverage cho mỗi `mlp_1`,
`mlp_3`, `mlp_5`, không phải lựa chọn theo metric. Mỗi cell chạy 3 repetitions
sau một warm-up được ghi tách biệt/không tính evidence; một fit là unit telemetry.

| Backend/mode | Phạm vi | Điều kiện |
| --- | --- | --- |
| CPU, sequential | TC/GMC × ba MLP × hai candidate × 3 repetitions | Bắt buộc; một fit tại một thời điểm. |
| CPU, parallel-2 | Cùng ma trận, ghép pair đã định trước | Bắt buộc sau sequential; không dùng nếu hard RAM/exit violation. |
| GPU, sequential | Cùng ma trận | Bắt buộc nếu GPU khả dụng; không giả định speedup. |
| GPU, parallel-2 | Cùng ma trận | Conditional: chỉ đề xuất sau preflight VRAM, có headroom và sequential ổn định. |

36 logical fits là **measured workload**, không gồm warm-up:
`2 dataset × 3 architecture × 2 candidate × 3 repetition`. Số invocation thực
tế được cố định như sau; warm-up được log tách riêng và không được tính vào
runtime hay cost comparison.

| Mode | Measured logical fits | Measured executions | Warm-up executions | VRAM preflight probes | Tổng executions thực tế |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU sequential | 36 | 36 (1 worker) | 12 | 0 | 48 |
| CPU parallel-2 | 36 | 18 paired (2 workers) | 6 paired | 0 | 24 |
| GPU sequential, chỉ khi GPU khả dụng | 36 | 36 (1 worker) | 12 | 0 | 48 |
| GPU parallel-2, conditional | 36 | 18 paired (2 workers) | 6 paired | 6 | 30 |

Parallel-2 vẫn chỉ có 36 measured logical fits; hai fit được ghép trong một
execution nên không tăng gấp đôi workload. Một mode không enabled không có
measured execution và không làm benchmark tổng thể fail. GPU parallel-2 chỉ
enabled sau 6 VRAM preflight probe pass và GPU sequential pass hard criteria.

Telemetry tối thiểu: wall-clock, CPU utilization/process CPU time, host peak RAM,
GPU utilization, peak VRAM, timeout/process exit, stability, preprocessing và
training time khi tách được, estimated cost/fit, environment/provenance. Artifact
chỉ là engineering evidence non-publishable; cấm prediction, scientific metric,
raw rows, model weights và local absolute path. Layout đề xuất:
`artifacts/p7c4a-mlp-compute-benchmark/<plan_id>/<run_id>/` (Git-ignored), gồm
`plan_snapshot.yaml`, `environment.json`, `fits/<stable_id>/result.json`,
`summary.json`, `completion.json`. Resume chỉ nhận identity completed hợp lệ,
cùng digest/provenance; failed/timeout giữ lại và retry chỉ theo policy đã lock.

## Tiêu chí quyết định định trước

| Loại | Tiêu chí |
| --- | --- |
| Hard acceptance — đề xuất, chưa approved | Với mỗi **enabled mode**, 36/36 measured logical fit hoàn tất; 0 timeout, non-zero exit hay OOM; mỗi fit có cap `1.800 s` và measured fit không được chạm timeout; peak process-tree host RAM `≤12.348.030.976 bytes` (`11,5 GiB`, lấy từ `p7c3_mlp_feasibility_plan.yaml: compute_policy.rss_hard_bytes`); VRAM `≤70%` single-fit hoặc `≤85%` parallel-2 của VRAM total; tối đa một retry chỉ cho transient infrastructure failure; final identity không missing/invalid. |
| Hard acceptance — determinism | Digest, provenance và stable identity khớp; exact coverage; không duplicate/corrupt/unexpected artifact; không có predictive metric làm input. Tất cả vẫn chờ human approval. |
| Preferred — đề xuất, chưa approved | CPU parallel-2 throughput `≥1,25×` sequential; GPU median runtime/fit `≤0,80×` CPU matched; direct infrastructure cost/completed fit GPU `≤` CPU dưới price assumption được ghi; scenario phải nằm trong compute window được human phê duyệt. |
| Informational | CPU/GPU utilization, runtime variance, preprocessing/training split và point-estimate cost; không đủ một mình để quyết định backend/budget. |

CPU parallelism final chỉ được chấp nhận ở mức cao nhất đã pass hard criteria. GPU chỉ
được chọn nếu pass hard criteria và có lợi runtime hoặc chi phí theo preferred
criteria; nếu không, record phải nêu CPU hoặc "inconclusive". Scenario chỉ có thể
được đề xuất cho DR-P7C-03/04 khi estimated end-to-end workload, gồm contingency
và overhead policy, nằm trong thời gian dự án do human xác nhận. Không có ngưỡng
speedup nào được coi là kết quả quan sát trước benchmark.

### Cost và compute window

`direct infrastructure cost` = giá theo giờ được ghi tại thời điểm benchmark ×
wall-clock đo được + storage/transfer bắt buộc. `wall-clock completion time` là
elapsed time từ measured execution đầu tiên đến measured completion hợp lệ cuối
cùng, gồm scheduler wait. So sánh chỉ dùng cùng dataset/architecture/candidate/
repetition set. CPU cá nhân phải có nguồn chi phí ghi riêng; không được mặc định
cost-to-completion tương đương cloud CPU.

Sprint 2 theo `reports/Draft_Sprint_Plan_v2.md` kết thúc 15/08/2026; Sprint 3
(17/08–29/08) dành cho Phase 9–11 và phụ thuộc Phase 7–8. Vì vậy **21 ngày
không phải hard project deadline hiện hữu**. Proposal pending human approval là
14 ngày calendar active compute + 7 ngày buffer (= 21 ngày); chỉ dùng nếu human
rebaseline schedule vì nó vượt mốc Sprint 2. Nếu không rebaseline, scenario phải
fit phần thời gian Sprint 2 còn lại; không scenario nào được tự approved từ
proposal này.

## Ngoài phạm vi và approval cần thiết

Kết quả MLP không tự áp dụng cho TabNet hay FT-Transformer. TabNet cần GPU
benchmark riêng ở P7C.5/DR-P7C-06; FT-Transformer cần scope approval và GPU
benchmark riêng ở P7C.6/DR-P7C-07. P7C.4A không triển khai các benchmark đó.

P7C.4A hoàn tất planning; P7C.4B bị **blocked awaiting human approval**. Trước
P7C.4B, human phải phê duyệt: (1) giữ/loại MLP-5 theo DR-P7C-04; (2)
scenario candidate budget theo DR-P7C-03; (3) hard RAM/VRAM, timeout, cost và
deadline thresholds; (4) policy GPU availability/parallel-2; (5) seed,
candidate-list và retry/retention policy final. Sau approval mới có thể tạo final
manifest hoặc runner; P7C.4A cố ý không tạo runner.
