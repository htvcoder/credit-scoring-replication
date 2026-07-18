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
          Phase 1 đã vận hành production tại http://34.142.206.15, nhưng vẫn
          chưa đánh dấu Completed cho đến khi rollback production được kiểm thử
          thật. Các phase sau vẫn Planned cho đến khi có bằng chứng nghiệm thu.
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
