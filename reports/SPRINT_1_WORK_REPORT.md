# BÁO CÁO CÔNG VIỆC – SPRINT 1

- Người báo cáo: Hoàng Trọng Vĩnh
- Người hướng dẫn: Trần Công Phú Khánh
- Thời gian: 16/07/2026 - 24/07/2026
- Đơn vị: VinSmart Future - Fintech
- Dự án: Credit Scoring Replication

| STT | Mô tả | Trạng thái | Ghi chú |
| --: | ----- | ---------- | ------- |
| 1 | Phase 0 – Lựa chọn bài báo gốc và xác định phạm vi tái lập | Hoàn thành có lưu ý | Chọn paper Deep Learning for Credit Scoring; phạm vi là partial replication do chỉ có 6/10 dataset công khai. |
| 2 | Phase 0 – Thu thập paper và 6 bộ dữ liệu công khai | Hoàn thành | Paper lưu trong `paper/`; dữ liệu công khai gồm AC, GC, HMEQ, TH02, TC và GMC. |
| 3 | Phase 0 – Xây dựng registry dữ liệu chuẩn | Hoàn thành | `data/datasets.yaml` là source of truth cho active file, target mapping, schema, feature groups và caveat. |
| 4 | Phase 0 – Tạo và kiểm tra SHA-256 checksum | Hoàn thành có lưu ý | `data/checksums-sha256.csv`, `scripts/verify_credit_datasets.py`; HMEQ full không khớp checksum SAS artifact kỳ vọng. |
| 5 | Phase 0 – Xây dựng data cards và metadata dataset | Hoàn thành | `docs/data-cards/AC.md`, `GC.md`, `HMEQ.md`, `TH02.md`, `TC.md`, `GMC.md`. |
| 6 | Phase 0 – Xác minh target mapping, class distribution, kiểu biến và identifier columns | Hoàn thành | Mapping chuẩn `0 = good/non-default`, `1 = bad/default`; loại ID như TC `ID` và GMC `Unnamed: 0`. |
| 7 | Phase 0 – Khắc phục HMEQ và khả năng đọc TH02 | Hoàn thành có lưu ý | Dùng `hmeq_full.csv`; TH02 có conversion artifact từ BIFF2; HMEQ giữ caveat provenance. |
| 8 | Phase 0 – Đánh giá tính khả thi thực nghiệm | Hoàn thành có lưu ý | `docs/EXPERIMENT_FEASIBILITY_ASSESSMENT.md`; kết luận PASS WITH CAVEAT cho 6 dataset công khai. |
| 9 | Phase 1 – Xây dựng website giới thiệu và theo dõi tiến độ | Hoàn thành | Website Next.js trong `website/`, gồm trang giới thiệu, dữ liệu, phương pháp, tiến độ, kết quả placeholder. |
| 10 | Phase 1 – Xây dựng nhận diện đỏ – đen – trắng và logo dự án | Hoàn thành | Brand assets trong `website/public/brand/`; màu chính đỏ, đen, trắng. |
| 11 | Phase 1 – Đồng bộ dataset, tài liệu và trạng thái phase lên website | Hoàn thành | `website/content/progress.yaml`, `methods.md`, `deviations.md`, `datasets.public.json`; script sync/validate content. |
| 12 | Phase 1 – Container hóa và triển khai production lên Google Cloud VM | Hoàn thành | `website/Dockerfile`, `docker-compose.prod.yml`, GHCR image và deployment qua SSH lên VM. |
| 13 | Phase 1 – Thiết lập production health check và trạng thái container | Hoàn thành | `/`, `/health/`, `/version/`; production chạy HTTP/public IP theo `docs/DEPLOYMENT_GOOGLE_CLOUD.md`. |
| 14 | Phase 1 – Kiểm thử rollback production | Hoàn thành | Manual rollback PASS; automatic failed-deployment rollback PASS; deploy lại production từ `main`. |
| 15 | Phase 1 – Đồng bộ tài liệu, website và tag milestone | Hoàn thành | Tag `p1-website-production-complete`; README và deployment docs ghi Phase 1 Completed. |
| 16 | Phase 2 – P2A: Xây dựng Python package, config và dataset loader | Hoàn thành | Package `src/creditrep`; loader trong `src/creditrep/datasets/`; CLI `scripts/inspect_dataset.py`. |
| 17 | Phase 2 – P2A: Xây dựng script kiểm tra dataset | Hoàn thành | `scripts/verify_credit_datasets.py`; test `tests/test_verify_credit_datasets.py`. |
| 18 | Phase 2 – P2B: Xây dựng deterministic split và artifact contract | Hoàn thành | `src/creditrep/splitting/`, `src/creditrep/artifacts/`, `scripts/create_split_artifact.py`; artifact gồm manifest/config/split. |
| 19 | Phase 2 – P2B: Hash, metadata và kiểm tra split hợp lệ | Hoàn thành | Canonical SHA-256 `config_hash`, `split_hash`; kiểm tra overlap, coverage, checksum và portable paths. |
| 20 | Phase 2 – P2C: Xây dựng smoke experiment runner | Hoàn thành có lưu ý | `scripts/run_experiment.py`; smoke artifacts có `publishable: false`, `result_scope: smoke_validation`. |
| 21 | Phase 2 – P2C: Hỗ trợ Logistic Regression, XGBoost và preprocessing train-only | Hoàn thành có lưu ý | Smoke baseline fit preprocessing chỉ trên train; metrics/predictions/model metadata chỉ phục vụ nghiệm thu kỹ thuật. |
| 22 | Phase 2 – Kiểm thử, môi trường sạch và đồng bộ trạng thái | Hoàn thành | Tag `p2-experiment-foundation-complete`; test suite hiện tại pass `179 passed`; Phase 2 docs/README/website đã đồng bộ. |
| 23 | Phase 3 – P3A: Xây dựng preprocessing contract chống leakage | Hoàn thành | `LeakageSafePreprocessor`; fit chỉ trên training features; target/identifier bị loại khỏi features. |
| 24 | Phase 3 – P3A: Bảo toàn fitted state và tách diagnostics | Hoàn thành | `transform()` không refit/extend state; diagnostics unseen categories không đi vào fitted metadata. |
| 25 | Phase 3 – P3A: Xử lý unseen categories và deterministic tie-breaking | Hoàn thành | Unknown token cho category mới; mode tie-break deterministic theo type và representation. |
| 26 | Phase 3 – P3B: Xây dựng WOE cho categorical features | Hoàn thành | WOE categorical, smoothing mặc định `0.5`, unknown fallback `0.0`; numeric giữ passthrough sau imputation. |
| 27 | Phase 3 – P3B: Xây dựng IterativeVIFSelector và xử lý scaling | Hoàn thành | VIF threshold mặc định `10.0`; lọc zero-variance, xử lý infinite VIF, optional train-only standard scaler. |
| 28 | Phase 3 – P3B: Tích hợp Protocol A preprocessing pipeline | Hoàn thành | `ProtocolAPreprocessingPipeline` kết hợp imputation, WOE, VIF và optional scaling. |
| 29 | Phase 3 – P3C: Xây dựng nested cross-validation infrastructure | Hoàn thành có lưu ý | Outer repeated stratified 2-fold, inner stratified k-fold; fake estimator/scorer chỉ kiểm thử hạ tầng. |
| 30 | Phase 3 – P3C: Seed derivation, fold hash và leakage checks | Hoàn thành có lưu ý | Seed từ canonical JSON + SHA-256; kiểm tra fold hash, overlap, coverage, class distribution; artifacts non-publishable. |
| 31 | Phase 3 – Kiểm thử và đồng bộ trạng thái hoàn thành | Hoàn thành | Tag `p3-leakage-safe-preprocessing-complete`; P3A/P3B/P3C docs và website methods đã cập nhật. |
| 32 | Phase 4 – P4A: Xác định metric specification và contract dùng chung | Hoàn thành | `docs/METRIC_SPECIFICATION.md`; positive class `1 = bad/default`, `y_score` là xác suất class `1`. |
| 33 | Phase 4 – P4A: Đặc tả ROC AUC, Brier Score, Partial Gini và EMP | Hoàn thành có lưu ý | Xác định direction, validation, undefined/failed result; decision record `docs/decisions/P4A_PARTIAL_GINI_AND_EMP.md`. |
| 34 | Phase 4 – P4B: Cài đặt validated metrics | Hoàn thành | `src/creditrep/metrics/`; ROC AUC xử lý ties, Brier Score, Partial Gini theo `y_score <= b`. |
| 35 | Phase 4 – P4B: Xử lý input validation và edge cases | Hoàn thành | Single-class fold trả `undefined`; input rỗng/sai length/NaN/score ngoài `[0,1]` trả `failed`. |
| 36 | Phase 4 – P4B: Giữ tương thích smoke runner và bổ sung unit tests | Hoàn thành | Flat smoke metric schema giữ nguyên; test contract và validated metrics trong `tests/test_p4a_*`, `tests/test_p4b_*`. |
| 37 | Phase 4 – P4C: Xây dựng metric registry và cấu hình chọn metric | Hoàn thành | `src/creditrep/metrics/registry.py`, `src/creditrep/config/metric_validation.py`, `configs/experiments/metric_validation_gc_reduced.yaml`. |
| 38 | Phase 4 – P4C: Tích hợp metric-validation workflow và artifact | Hoàn thành có lưu ý | `scripts/run_metric_validation.py`, `src/creditrep/experiments/metric_validation.py`, `src/creditrep/artifacts/metric_validation.py`; `publishable: false`, `result_scope: metric_validation`. |
| 39 | Phase 4 – P4C: Chốt EMP unsupported | Hoàn thành có lưu ý | EMP không tính numeric value do thiếu business parameters/economic assumptions có provenance; không tự đặt tham số giả. |
| 40 | Phase 4 – Kiểm thử, build website và đồng bộ trạng thái hoàn thành | Hoàn thành | Tag `p4-metric-validation-complete`; `python -m pytest -q` pass 179 tests; website lint/type-check/content validation/build pass. |

## Kết quả Sprint 1

Sprint 1 đã hoàn thành nền tảng dữ liệu, website production, hạ tầng thực nghiệm, preprocessing chống leakage, nested cross-validation và hệ thống metric validation. Dự án hiện đã hoàn thành đến hết Phase 4, với source of truth ghi `last_completed_phase: 4` và phase hiện tại là Phase 5. Các smoke artifact, reduced/fake preprocessing-validation artifact và metric-validation artifact chỉ là artifact nghiệm thu kỹ thuật, chưa phải kết quả khoa học để công bố. Phase tiếp theo là Phase 5 – triển khai XGBoost baseline theo protocol thực nghiệm chính thức, cùng các baseline cổ điển và ensemble liên quan.
