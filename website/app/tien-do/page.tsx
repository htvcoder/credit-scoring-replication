import { StatusBadge } from "@/components/StatusBadge";
import { getProgressContent } from "@/lib/content";

export default function ProgressPage() {
  const progress = getProgressContent();

  return (
    <div className="page">
      <section className="page-title">
        <p className="eyebrow">Roadmap</p>
        <h1>Tiến độ các phase</h1>
        <p className="lead">
          P1A không tự động đánh dấu Phase 1 Completed. Các phase sau vẫn ở
          trạng thái Planned cho đến khi có bằng chứng nghiệm thu tương ứng.
        </p>
      </section>

      <section className="timeline">
        {progress.phases.map((phase) => (
          <article className="timeline-item" key={phase.id}>
            <div className="timeline-index">{phase.id}</div>
            <div>
              <div className="timeline-title">
                <h2>{phase.title}</h2>
                <StatusBadge status={phase.status} />
              </div>
              <p>{phase.summary}</p>
              {phase.caveat ? <p className="caveat">{phase.caveat}</p> : null}
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}
