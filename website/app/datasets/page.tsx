import { getDatasetSummaries } from "@/lib/datasets";

export default function DatasetsPage() {
  const datasets = getDatasetSummaries();

  return (
    <div className="page">
      <section className="page-title">
        <p className="eyebrow">Source of truth: data/datasets.yaml</p>
        <h1>Phạm vi dữ liệu công khai</h1>
        <p className="lead">
          Website chỉ hiển thị metadata công khai cần thiết. Raw path, processed
          path và dữ liệu thô không được đưa lên giao diện.
        </p>
      </section>

      <section className="dataset-grid">
        {datasets.map((dataset) => (
          <article className="dataset-card" key={dataset.id}>
            <div className="dataset-card-header">
              <span>{dataset.id}</span>
              <strong>{dataset.fullName}</strong>
            </div>
            <dl className="metric-list">
              <div>
                <dt>Số mẫu</dt>
                <dd>{dataset.rows.toLocaleString("vi-VN")}</dd>
              </div>
              <div>
                <dt>Input</dt>
                <dd>{dataset.inputCount}</dd>
              </div>
              <div>
                <dt>Target</dt>
                <dd>{dataset.targetColumn}</dd>
              </div>
              <div>
                <dt>Default rate</dt>
                <dd>{(dataset.defaultRate * 100).toFixed(1)}%</dd>
              </div>
            </dl>
            <p>{dataset.publicNote}</p>
            <p className="muted">Nguồn: {dataset.source}</p>
            <p className="muted">License/access: {dataset.license}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
