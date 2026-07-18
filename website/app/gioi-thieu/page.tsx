import {
  getInternshipContent,
  getMarkdownContent,
  getPaperContent,
} from "@/lib/content";
import { renderMarkdown } from "@/lib/markdown";

export default function InternshipPage() {
  const internship = getInternshipContent();
  const paper = getPaperContent();
  const background = getMarkdownContent("background.md");
  const limitations = getMarkdownContent("limitations.md");
  const details = [
    ["Tên đề tài", internship.title],
    ["Mô tả ngắn", internship.short_description],
    ["Mục tiêu dự án", internship.project_goal],
    ["Người thực hiện", internship.student],
    ["Người hướng dẫn", internship.supervisor],
    ["Đơn vị hoặc chương trình", internship.unit_or_program],
    ["Thời gian thực hiện", internship.timeframe],
    ["Repository", "repository"],
    ["Vai trò của website", internship.website_role],
  ];

  return (
    <div className="page">
      <section className="page-title">
        <p className="eyebrow">Giới thiệu đề tài</p>
        <h1>{internship.title}</h1>
        <p className="lead">{internship.short_description}</p>
      </section>

      <section className="intro-layout">
        <div className="detail-list">
          {details.map(([label, value]) => (
            <div className="detail-row" key={label}>
              <dt>{label}</dt>
              <dd>
                {value === "repository" ? (
                  <a
                    className="text-link repository-link"
                    href={internship.repository}
                    rel="noopener noreferrer"
                    target="_blank"
                  >
                    {internship.repository}
                  </a>
                ) : (
                  value
                )}
              </dd>
            </div>
          ))}
        </div>
        <article className="content-card">
          <h2>Phạm vi công việc</h2>
          <ul>
            {internship.work_scope.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>
      </section>

      <section className="content-card">
        <p className="eyebrow">Bài báo gốc</p>
        <h2>{paper.title}</h2>
        <dl className="paper-facts">
          <div>
            <dt>Tác giả</dt>
            <dd>{paper.authors}</dd>
          </div>
          <div>
            <dt>Năm công bố</dt>
            <dd>{paper.year}</dd>
          </div>
          <div>
            <dt>Tạp chí / nguồn xuất bản</dt>
            <dd>{paper.publication}</dd>
          </div>
          <div>
            <dt>Nhà xuất bản</dt>
            <dd>{paper.publisher}</dd>
          </div>
          <div>
            <dt>DOI</dt>
            <dd>
              <a className="text-link" href={paper.official_link}>
                {paper.doi}
              </a>
            </dd>
          </div>
        </dl>
        <p>{paper.main_objective}</p>
      </section>

      <section className="section-grid">
        <article className="content-card">
          <h2>Nội dung nghiên cứu của paper</h2>
          <p>{paper.problem_context}</p>
          <h3>Mô hình được so sánh</h3>
          <ul>
            {paper.compared_models.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <p>
            Paper sử dụng {paper.dataset_count} dataset credit scoring trong
            thiết kế thực nghiệm.
          </p>
        </article>

        <article className="content-card">
          <h2>Thiết kế thực nghiệm</h2>
          <h3>Preprocessing</h3>
          <ul>
            {paper.preprocessing.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <h3>Cross-validation</h3>
          <ul>
            {paper.cross_validation.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>
      </section>

      <section className="section-grid">
        <article className="content-card">
          <h2>Metric và kiểm định</h2>
          <h3>Metric chính</h3>
          <ul>
            {paper.metrics.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <h3>Phân tích thống kê</h3>
          <ul>
            {paper.statistical_analysis.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>

        <article className="content-card">
          <h2>Kết quả chính paper báo cáo</h2>
          <ul>
            {paper.reported_findings.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <p className="caveat">Đề tài thực tập chưa công bố kết quả thực nghiệm.</p>
        </article>
      </section>

      <section className="prose content-card">
        {renderMarkdown(limitations)}
      </section>

      <section className="prose content-card">
        {renderMarkdown(background)}
      </section>
    </div>
  );
}
