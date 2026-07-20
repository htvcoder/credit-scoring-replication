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
          Phase 1 đã Completed với website production và CI/CD. Phase 2 đã
          Completed ở mức experiment foundation; Phase 3 là bước tiếp theo để
          hoàn thiện preprocessing protocol chống leakage.
        </p>
      </section>

      <section className="timeline">
        {progress.phases.map((phase) => (
          <article className="timeline-item" key={phase.id}>
            <div className="timeline-index">{phase.id}</div>
            <div className="timeline-content">
              <div className="timeline-title">
                <div>
                  <h2>{phase.title}</h2>
                </div>
                <StatusBadge status={phase.status} />
              </div>
              <p className="phase-summary">{phase.summary}</p>
              <div className="phase-detail-grid">
                <div>
                  <h3>Công việc chính</h3>
                  <ul className="phase-list">
                    {phase.tasks.map((task) => (
                      <li key={task}>{task}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3>Đầu ra chính</h3>
                  <ul className="phase-list">
                    {phase.deliverables.map((deliverable) => (
                      <li key={deliverable}>{deliverable}</li>
                    ))}
                  </ul>
                  {phase.checkpoints?.length ? (
                    <div className="phase-checkpoints">
                      <h3>Checkpoint</h3>
                      <ul className="phase-list">
                        {phase.checkpoints.map((checkpoint) => (
                          <li key={checkpoint}>{checkpoint}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
              </div>
              {phase.caveat ? <p className="caveat">{phase.caveat}</p> : null}
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}
