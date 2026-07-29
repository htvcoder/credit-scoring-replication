# Mô hình MLP theo độ sâu (P6B)

P6B đăng ký `mlp_1`, `mlp_3`, `mlp_5` dùng chung builder/trainer P6A. Hidden depth (1/3/5) là **paper_exact**. Width 64, ReLU, Adam 0.001, batch 32, 200 epochs, regularization tắt, seed 42 và early stopping (20, 0.0001) là **project_decision**, không phải paper architecture.

Mọi model trả logits, dùng `BCEWithLogitsLoss`, và `predict_proba` trả hai cột sklearn-style; cột 1 là `P(class 1 = bad/default)`. Ba model dùng budget ID `p6b_shared_v1`; chỉ depth khác nhau. Wrapper chỉ nhận numeric data đã preprocessing/scaling train-only, yêu cầu validation explicit khi early stopping, không split/tune/đánh giá hay ghi artifact.

PyTorch vẫn optional và lazy. P6B không chạy nested CV, dataset thật, scientific comparison hay core replication; metadata/config đều non-publishable (`mlp_model_validation`). P6C sẽ tích hợp inner validation, nested CV và fold artifacts.
