# P7B.1 — Runner feasibility CART-A

P7B.1 cung cấp runner riêng cho feasibility engineering CART-A. Runner chỉ thực hiện từng inner fit đã được khóa trong P7A; không có candidate selection, winner/best-candidate, predictive ranking, outer-test evaluation hay outer selected-model refit. Mọi artifact mang `phase=P7B`, `purpose=engineering_feasibility`, `publishable=false` và không phải kết quả khoa học.

Runner tái sử dụng dataset loader, checksum, nested split, seed derivation, preprocessing train-only, model factory/registry, canonical hashing và P7A manifest lock. P5C không bị sửa đổi.

Artifact root dành riêng cho P7B là `artifacts/p7b-cart-feasibility/` và toàn bộ runtime output dưới root này bị Git ignore. Root này không chứa manifest/protocol/source/test cần version-control. Một plan-only dry-run gồm `plan.json`, `config_snapshot.json`, `environment.json` và `validator.json`; P7B.2 chỉ tạo thêm `fits/<fit-id>/result.json` hoặc `failure.json` cùng `engineering_summary.json`. Các path lưu trong plan là relative, dùng dấu `/`, và không được phép escape ra ngoài run root.

Telemetry của mỗi fit ghi thời gian, attempts, rows/features, leaf fraction/cận nguyên `ceil`, artifact bytes và RSS process bằng `psutil.Process.memory_info().rss`. Sampler nền lấy mẫu mỗi 0.05 giây, chỉ tính process hiện tại (không cộng child process), lưu `process_rss_start_bytes`, `process_rss_peak_bytes` và `process_rss_delta_peak_bytes`. Đây là RSS process-local, không phải system-wide used memory; native allocation trong chính process được phản ánh ở mức RSS, nhưng child process không được tính. `python_tracemalloc_peak_bytes` vẫn được lưu riêng để chẩn đoán allocation Python, không phải peak RAM thực tế và không dùng để ngoại suy workload. CART chạy tuần tự theo mặc định. Retry tối đa một lần sau lần chạy đầu, chỉ cho `OSError`/`TimeoutError`; resume chỉ bỏ qua result `completed` cùng fit ID và run/config hash.

## Provenance và validation

`run` và `resume` bắt buộc phải chụp Git provenance **trước khi load dữ liệu hoặc fit đầu tiên**. `git_head` phải là SHA đầy đủ 40 ký tự; runner cũng lưu `working_tree` (`clean`/`dirty`) và porcelain có cấu trúc. Nếu Git executable thiếu, root không phải Git repository, ownership/safe-directory bị Git từ chối, hoặc HEAD không resolve được, lệnh fail-fast với hướng dẫn khắc phục và không bắt đầu training. Runner gọi Git với repository root tường minh, không tự sửa `safe.directory` hay Git config. Detached HEAD và shallow clone vẫn hợp lệ khi HEAD resolve được SHA đầy đủ.

`plan` là dry-run plan-only, không fit model. Nó có thể tạo artifact non-training/non-publishable khi provenance không lấy được; record phải phản ánh điều này và không làm suy yếu fail-fast của `run`/`resume`. `validate-artifacts` tái đọc plan, snapshot, environment, summary và từng fit artifact; kiểm tra hash/config/identity, paths, provenance, summary, retry, telemetry RSS và trạng thái completed/failed. `validator.json` chỉ là record của một lần validation, không phải nguồn sự thật; validator không tin giá trị `valid=true` cũ. Plan-only dry-run được báo là `training_artifacts_validated=false`, không bao giờ được mô tả là completed training run.

Lệnh dry-run P7B.1 (không fit model):

```powershell
.\.venv\Scripts\python.exe -m creditrep.experiments.p7b_cli plan --output-dir artifacts\p7b-cart-feasibility\p7b1-dry-run
```

Lệnh dự kiến cho P7B.2 (chỉ dùng khi được yêu cầu riêng):

```powershell
.\.venv\Scripts\python.exe -m creditrep.experiments.p7b_cli run --output-dir artifacts\p7b-cart-feasibility\p7b2-run-001
```

Kiểm tra output sau run (hoặc kiểm tra dry-run ở chế độ plan-only):

```powershell
.\.venv\Scripts\python.exe -m creditrep.experiments.p7b_cli validate-artifacts --output-dir artifacts\p7b-cart-feasibility\p7b2-run-001
```

P7B.1 hiện implementation-ready; P7B.2 chưa chạy. P7B.1 không khóa final scientific search space của P7C và không dùng metric dự đoán để quyết định bao gồm mô hình.
