# P7C.4B.1c — xác minh CPU parallel-2 có giới hạn

## Vấn đề và câu hỏi con

Xác minh scheduler CPU dùng tối đa hai fit worker cô lập bằng process trên workload fixture nhỏ; đây không phải benchmark canonical hay kết quả khoa học. Câu hỏi là liệu contract artifact/resume/validator của B1b còn giữ được khi hai fit có thể hoàn tất khác thứ tự hay không.

## Phương pháp và contract

`cpu_parallel_2` dùng Windows-safe `spawn`. Parent cấp logical/attempt identity theo thứ tự plan, giữ `max_workers=2`, ghi artifact atomically và tự tạo summary/validation/marker. Worker chỉ chạy một fit, có giới hạn OpenMP/MKL/BLAS là 2 (execution setting, không phải hyperparameter). Manifest mang `execution_mode`, `max_workers`, provenance, config digest và nhãn `engineering smoke evidence — non-publishable`.

RSS process-tree được sampler parent đo bằng `psutil`: mỗi sample cộng parent và descendant đang sống, tránh double-count PID; record ghi parent/worker PID, interval, peak byte và active-worker cap. Ngưỡng cứng là `rss_hard_bytes` trong `configs/protocols/p7c/p7c3_mlp_feasibility_plan.yaml`.

## Bounded evidence

Audit hardening phát hiện phép so sánh trước đó không cùng timing boundary (sequential cộng per-fit duration, parallel dùng duration khác) và sequential chưa có process-tree peak cùng schema. Hai run mới `smoke-audit-seq-v3` và `smoke-audit-par-v3` dùng cùng workload digest/4 logical IDs, cùng fixture/seed/partition/candidate/architecture và cùng boundary `run_wall_clock_seconds`; validator đọc lại từ disk đều PASS:

| Metric | CPU sequential | CPU parallel-2 |
| --- | ---: | ---: |
| Expected / completed / failed | 4 / 4 / 0 | 4 / 4 / 0 |
| Fixture wall-clock evidence (s) | 20.609 | 13.984 |
| Peak process-tree RSS (bytes) | 515,325,952 | 831,856,640 |
| Max active workers | 1 | 2 |
| Validation / marker | PASS / present | PASS / present |

Fixture speedup evidence là xấp xỉ `1.474`; memory ratio là `1.614`. Cả hai đều thấp hơn RSS hard threshold `12,348,030,976` bytes từ plan P7C.3. Speedup cũ `2.96×` bị loại vì không apples-to-apples. Đây chỉ là artifact engineering non-publishable: không chứng minh throughput hay memory của TC/GMC canonical.

## Failure, interruption và resume

Regression fixture xác minh completion được giữ khi interruption, resume skip fit đã complete và thực thi fit còn thiếu; corrupt artifact bị validator từ chối và quarantine giữ reason manifest. Corruption là terminal: resume không ghi đè evidence hoặc reset retry budget.

Audit hardening bổ sung completion-order inversion có kiểm soát, mixed success/failure (failure mang `deterministic_fit_failure`, không có run marker) và CLI subprocess Windows `spawn`: parallel run/validator/resume trả `0`; missing run và unsupported GPU mode trả non-zero. Summary lưu terminal logical IDs theo plan order, độc lập với completion order. Run-level RSS parallel là maximum của các sample parent + unique live descendants cùng thời điểm, không phải tổng các peak rời rạc.

## Acceptance và kết luận

B1c pass trong phạm vi bounded fixture: cap hai worker, process isolation/spawn, identity deterministic, telemetry parallel, failure/corruption isolation, resume và validator đều được kiểm thử. Chưa được phép suy luận parallel-2 là lựa chọn canonical/final hoặc thử parallelism lớn hơn hai. Khuyến nghị B1d: chỉ xem xét operator/readiness gate sau review độc lập của artifact/telemetry và policy canonical.

Full acceptance suite B1a+B1b+B1c/protocol/status kết thúc tự nhiên với `44 passed` trong `303.59s` (exit 0). Đây là điều kiện hardening; không thay đổi trạng thái scientific hay cho phép chạy canonical.
