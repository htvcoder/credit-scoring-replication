# P7C.3 — Closeout MLP feasibility pilot

## Trạng thái

**Completed — canonical feasibility pilot accepted.** Evidence duy nhất được chấp nhận là `artifacts/p7c3-mlp-feasibility/vm-run-003`. `vm-run-001` là historical/non-canonical; `vm-run-002` là historical invalid (đã phát hiện lỗi WOE propagation và `plan_mismatch`). Hai artifact này được giữ lại nhưng không dùng để tính workload projection hay đưa ra kết luận khoa học.

## Evidence đã xác nhận

Local `validate-artifacts` trả về `valid: true`, `completion_status: completed`, `expected: 60`, `completed: 60`, `failed: 0`, `missing: 0`, `resumable: false` và `errors: []`. Có đúng 60 `result.json`: 12 nhóm coverage (2 dataset × 3 model × 2 candidate), mỗi nhóm có 5 fit hoàn tất. Cả 60 fit có `timeout: false`, `process_exit_code: 0`, `rss_threshold_state: ok`, không có lỗi timestamp hay record không hoàn tất.

Plan digest là `fdfa543c82b159840aa85664f11f349f06750838c7d868ca4392ace8de57b749`; provenance của environment và từng fit ghi Git HEAD `84c71266d0eb375effc317601602fb9deb67d7d2` với working tree `clean`. Pilot bắt đầu `2026-08-07T08:21:19.850793+00:00`, hoàn tất `2026-08-07T09:16:19.142870+00:00`, wall-clock `3299.292077` giây; tổng fit time `3299.2772443050026` giây (`0.916465901195834` giờ) và đỉnh process-tree RSS `916.6640625 MiB`.

Kết luận theo acceptance criteria engineering: **CPU feasibility PASS; memory feasibility PASS; execution stability PASS.** GPU không bắt buộc cho correctness/feasibility của MLP. Pilot không có predictive metric, ranking, outer refit hoặc lựa chọn candidate; tuyệt đối không diễn giải nó như kết quả hiệu năng dự báo hoặc kết quả khoa học cuối cùng.

## Workload projection tái lập

Nguồn protocol là `configs/protocols/p7a/p7a_candidate_manifest.yaml`: `reference_search_spaces.mlp.mlp_1.declared_configurations: 144`, `mlp_3: 720`, `mlp_5: 2016`. Đây là số candidate **trên mỗi architecture**, không phải tổng ba architecture và không phải Cartesian product cần nhân thêm. Manifest đã ghi `count_inconsistency`: các số Table 2 khớp công thức không gồm `batch_normalization`; vì vậy projection dùng đúng `declared_configurations`, không tự nhân đôi.

`cross_validation.outer_splits: 2` cùng `outer_repeats` cho AC/GC/TH02 là 10 và HMEQ/TC/GMC là 5 cho ra: AC `2×10=20`, GC `20`, TH02 `20`, HMEQ `10`, TC `10`, GMC `10`; tổng `90` outer partitions đã bao gồm cả sáu dataset. `cross_validation.inner_splits: 5`. Công thức cũ `6 × 90 × 5 × Σ(C_m × t_m)` sai vì nhân lại sáu dataset sau khi `90` đã tổng hợp chúng.

Inner candidate-evaluation fits phải tính tách theo dataset và architecture: `Σ_d Σ_m [C_m × O_d × 5] = (144 + 720 + 2016) × (20+20+20+10+10+10) × 5 = 1,296,000`. Outer refit sau chọn candidate là `Σ_d Σ_m O_d = 3 × 90 = 270`; không trộn 270 refit này vào inner search. Protocol chưa quy định fit bổ sung nào khác; retry chỉ khi lỗi transient. Prediction/evaluation, I/O và aggregation có thể phát sinh overhead nhưng không được pilot đo riêng, nên không được cộng thành thời gian fit.

Nguồn quan sát chỉ là `vm-run-003`, CPU tuần tự một fit, hai thread. Mean theo candidate-inner-fit: TC MLP-1 `11.5581 s`, MLP-3 `16.8882 s`, MLP-5 `28.9361 s`; GMC MLP-1 `56.9784 s`, MLP-3 `99.3005 s`, MLP-5 `116.2665 s`. Với đúng 10 outer partitions của mỗi dataset benchmark, projection theo architecture là TC: `23.12/168.88/810.21 giờ` (tổng `1,002.21 giờ`); GMC: `113.96/993.00/3,255.46 giờ` (tổng `4,362.42 giờ`).

`low_stress` và `high_stress` không phải cặp biên runtime hợp lệ: trong canonical run, `low_stress` thường lâu hơn do early stopping/training dynamics. Chúng chỉ là hai candidate engineering đã định trước; mean của chúng được dùng làm point scenario của từng dataset/architecture, không là dự báo cho toàn grid. TC/GMC tạo observed scenarios cho bốn dataset chưa benchmark.

| Kịch bản | Cách tính | Kết quả projected | Giới hạn |
| --- | --- | ---: | --- |
| CPU sequential | `90 × 5 × Σ(C_m × t_m)`; toàn bộ outer partition lần lượt dùng mean TC hoặc GMC | TC scenario `9,019.87 giờ` (`375.83 ngày`); GMC scenario `39,261.82 giờ` (`1,635.91 ngày`); midpoint tham chiếu `24,140.84 giờ` | Ngoại suy TC/GMC sang AC, GC, TH02, HMEQ; không phải dự báo chính xác. Chưa gồm 270 refit hay overhead. |
| CPU parallel 2 fit | Sequential ÷ `(2 × 0.8)`; giả định hiệu suất 80% | TC `5,637.42 giờ` (`234.89 ngày`); GMC `24,538.63 giờ` (`1,022.44 ngày`); midpoint `15,088.02 giờ` | Hai fit đồng thời chưa được benchmark; RSS đơn-fit không chứng minh RSS/throughput song song. |
| GPU | `T_GPU = T_CPU / S_GPU` | Chưa định lượng | GPU speedup chưa được quan sát; cần benchmark riêng trước quyết định. |

Các assumption: candidate count là reference count chưa khóa final; 90 partitions đã gồm sáu dataset; duration là mean hai candidate trên một outer partition; duration/refit/prediction evaluation của bốn dataset còn lại chưa quan sát. Early stopping, kích thước dataset, I/O, retry, scheduling, VRAM và outer-refit có thể làm workload thực khác đáng kể. CPU full grid vì vậy không có cơ sở để cam kết trong target/ceiling hiện tại, nhưng đây không phải benchmark GPU.

## Đề xuất quyết định

- **Giữ `mlp_5` trong core scope**: evidence đủ để xác nhận feasibility, nhưng pilot không có metric nên không được dùng để loại kiến trúc. Việc giữ/bỏ cuối cùng vẫn cần phê duyệt con người theo `DR-P7C-04` (MLP-5 core-scope inclusion decision).
- **Final MLP candidate search space**: chưa khóa và không giảm grid từ pilot; giữ reference space MLP-1 `144`, MLP-3 `720`, MLP-5 `2016` cho đến khi có quyết định/manifest được phê duyệt.
- **CPU parallelism**: chọn đề xuất vận hành ban đầu tối đa 2 fit CPU, giữ 2 thread mỗi fit và theo dõi RSS/throughput; cần benchmark đồng thời có kiểm soát trước khi coi đây là budget final.
- **CPU hay GPU cho final MLP**: GPU không cần cho correctness/feasibility, nhưng full CPU grid có range quá lớn; đề xuất GPU benchmark riêng trước khi phê duyệt final backend/budget. Chưa được tự khóa CPU hay GPU.
- **TabNet/FT-Transformer**: chưa có evidence riêng; cần GPU feasibility/budget riêng, không suy ra từ MLP pilot. TabNet vẫn ở P7C.5; FT-Transformer vẫn là extension chờ scope approval ở P7C.6.

Đủ evidence: closeout P7C.3, CPU/memory/stability feasibility và việc không dùng pilot để ranking/exclusion. Còn chờ phê duyệt: `DR-P7C-03` final MLP-1/3 candidate budget and search strategy, `DR-P7C-04` MLP-5 core-scope inclusion decision, backend/budget final và final manifest; `DR-P7C-06/07` của TabNet/FT-Transformer không được đóng bởi P7C.3.
