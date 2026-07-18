import Image from "next/image";
import Link from "next/link";
import { getDatasetSummaries } from "@/lib/datasets";
import { getProgressContent, getProjectContent } from "@/lib/content";
import { StatusBadge } from "@/components/StatusBadge";

export default function Home() {
  const project = getProjectContent();
  const datasets = getDatasetSummaries();
  const progress = getProgressContent();

  return (
    <div className="page">
      <section className="hero">
        <div className="hero-brand">
          <Image
            alt="Logo dự án Tái lập và đánh giá lại mô hình tính điểm tín dụng"
            className="hero-logo"
            height="1248"
            priority
            sizes="(max-width: 560px) 240px, (max-width: 860px) 300px, 360px"
            src="/brand/csr-logo-full.png"
            unoptimized
            width="936"
          />
        </div>
        <div className="hero-copy">
          <p className="eyebrow">Đề tài nghiên cứu</p>
          <h1>{project.title}</h1>
          <p className="lead">{project.short_description}</p>
          <div className="actions">
            <Link className="button primary" href="/gioi-thieu/">
              Xem giới thiệu
            </Link>
            <Link className="button secondary" href="/tien-do/">
              Tiến độ
            </Link>
          </div>
        </div>
      </section>

      <section className="section-grid">
        <article>
          <h2>Câu hỏi nghiên cứu</h2>
          <div className="rq-list">
            {project.research_questions.map((rq) => (
              <div className="rq-item" key={rq.id}>
                <span>{rq.id}</span>
                <p>{rq.question}</p>
              </div>
            ))}
          </div>
        </article>
        <article>
          <h2>Vai trò của website</h2>
          <p>{project.role_of_website}</p>
          <ul>
            {project.scope_notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </article>
      </section>

      <section>
        <div className="section-heading">
          <div>
            <p className="eyebrow">Phạm vi dữ liệu</p>
            <h2>6 dataset công khai</h2>
          </div>
          <Link className="text-link" href="/datasets/">
            Chi tiết dataset
          </Link>
        </div>
        <div className="dataset-strip">
          {datasets.map((dataset) => (
            <div className="dataset-chip" key={dataset.id}>
              <strong>{dataset.id}</strong>
              <span>{dataset.rows.toLocaleString("vi-VN")} mẫu</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <div className="section-heading">
          <div>
            <p className="eyebrow">Tiến độ</p>
            <h2>Trạng thái phase</h2>
          </div>
          <Link className="text-link" href="/tien-do/">
            Xem roadmap
          </Link>
        </div>
        <div className="phase-preview">
          {progress.phases.slice(0, 3).map((phase) => (
            <article className="phase-card" key={phase.id}>
              <div>
                <span className="muted">{phase.id}</span>
                <h3>{phase.title}</h3>
              </div>
              <StatusBadge status={phase.status} />
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
