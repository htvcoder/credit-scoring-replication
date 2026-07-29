# Nền tảng huấn luyện MLP (P6A)

P6A xây dựng hợp đồng và trainer PyTorch tối thiểu để P6B có thể chốt MLP-1/3/5, còn P6C tích hợp chúng vào nested CV. Đây không phải core replication, không tạo kết quả khoa học, và mọi metadata/history P6A là `publishable: false`, với `result_scope: mlp_training_validation`.

## Phạm vi và dependency

PyTorch được chọn vì cho phép định nghĩa một builder dùng chung, logits rõ ràng và kiểm soát runtime/seed trực tiếp. Nó là optional extra để classical stack vẫn import/chạy được khi neural functionality không được dùng:

```bash
python -m pip install -e ".[neural,test]"
```

Không pin CUDA-specific wheel, không thêm dependency GPU-only. Khi PyTorch chưa cài mà gọi neural API, hệ thống ném `NeuralDependencyError` với hướng dẫn cài đặt thay vì lộ `ImportError` thô.

P6A gồm typed `MLPConfig`, runtime/device utilities, builder, early stopping, history/summary JSON-safe và tiny CPU trainer. P6A không gồm MLP production IDs, registry integration, preprocessing/scaling, tuning, nested CV, artifact production, DBN, CatBoost, TabNet hay FT-Transformer.

## Hợp đồng mô hình và probability

Builder nhận danh sách `hidden_layers`; các fixture `[64]`, `[64, 64, 64]`, `[64, 64, 64, 64, 64]` chỉ là ví dụ/test, không phải kiến trúc paper. Network trả logits, không có sigmoid nội bộ. Trainer dùng `BCEWithLogitsLoss`; chỉ `predict_probabilities` áp dụng sigmoid ở prediction boundary và trả shape `(n_samples,)` là `P(class 1) = P(bad/default)`, luôn finite trong `[0, 1]`.

Scaling và preprocessing không thuộc trainer: P6C phải fit chúng chỉ trên training partition. Outer test fold tuyệt đối không được dùng để fit preprocessing/scaling, early stopping, tuning, model selection hay threshold selection.

## Cấu hình, runtime và training lifecycle

`MLPConfig` kiểm tra hidden layers, binary output dimension, ReLU, Adam/AdamW, dropout, learning rate, weight decay, batch size, epochs, workers, seed, device policy và JSON serialization. Checkpoint file bị vô hiệu trong P6A; best weights chỉ snapshot `state_dict` trong memory.

Device policy:

- `auto`: CUDA nếu thực sự available, nếu không CPU.
- `cpu`: luôn CPU.
- `cuda`: fail-fast nếu CUDA unavailable; không fallback im lặng.

Mỗi training run mới cấu hình seed Python, NumPy, PyTorch CPU và CUDA khi có; import module không tạo global side effect. Metadata lưu requested/resolved device, CUDA availability/count, deterministic requested/enabled và limitation: cùng seed là best-effort trong stack hiện tại, không cam kết bitwise-identical giữa mọi OS/hardware.

Early stopping monitor validation loss, mode `min`, `patience` và `min_delta`. Giá trị NaN/Inf là lỗi dừng rõ ràng, không phải improvement. Improvement snapshot weights; sau training trainer restore best weights và ghi kết quả đó trong summary. History chỉ chứa epoch/loss/LR/duration/flags; summary chứa runtime, architecture, parameter count, optimizer, loss, stop reason, warnings và provenance JSON-serializable — không ghi tensor hoặc PyTorch object.

## Kiểm thử và các quyết định còn lại

Chạy test P6A bằng `python -m pytest tests/test_p6a_mlp_training_foundation.py -q`; test dùng dữ liệu toy CPU, không internet, raw dataset, result artifact hoặc checkpoint.

P6B cần chốt hidden widths, kiến trúc theo paper và deviation khi thiếu chi tiết, tuning grid, batch size/max epochs/patience, number of seeds, checkpoint retention, stable model IDs/config files và production wrapper. P6C cần quyết định inner-validation policy, nested-CV integration, resume/retry neural training và per-fold training artifacts. Các validation artifacts trước Phase 7 vẫn non-publishable và không được đưa metrics lên website như scientific results.
