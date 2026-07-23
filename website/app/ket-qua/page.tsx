export default function ResultsPage() {
  return (
    <div className="page narrow">
      <section className="page-title">
        <p className="eyebrow">Kết quả</p>
        <h1>Kết quả thực nghiệm chưa được công bố</h1>
        <p className="lead">
          Chưa có kết quả thực nghiệm chính thức để công bố. Phase 3 đã hoàn
          thành preprocessing và nested-CV foundation, nhưng chưa có validated
          scientific metrics, core replication run hoặc kết quả khoa học để công
          bố.
        </p>
      </section>

      <section className="notice">
        <h2>Trạng thái công bố</h2>
        <p>
          Website hiện chỉ giữ placeholder cho kết quả. Không có số liệu minh
          họa, biểu đồ giả, bảng metric giả, ranking mô hình giả hoặc kết luận
          nghiên cứu giả.
        </p>
      </section>

      <section className="notice">
        <h2>Kết quả nghiệm thu kỹ thuật</h2>
        <ul>
          <li>Dataset pipeline đã chạy thành công trên các dataset công khai.</li>
          <li>Deterministic split đã được kiểm chứng với checksum và split hash.</li>
          <li>Logistic Regression và XGBoost smoke runs đã chạy thành công.</li>
          <li>Prediction, metrics và model metadata artifacts hợp lệ.</li>
          <li>Reproducibility check đã pass cho smoke run kiểm chứng.</li>
          <li>P3C preprocessing-validation artifacts là non-publishable.</li>
        </ul>
        <p className="caveat">
          Các kết quả này chỉ xác nhận hạ tầng thực nghiệm hoạt động đúng, không
          phải kết quả so sánh mô hình hoặc kết quả tái lập paper.
        </p>
      </section>
    </div>
  );
}
