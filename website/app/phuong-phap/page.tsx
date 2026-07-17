import { getMarkdownContent } from "@/lib/content";
import { renderMarkdown } from "@/lib/markdown";

export default function MethodsPage() {
  const methods = getMarkdownContent("methods.md");
  const deviations = getMarkdownContent("deviations.md");

  return (
    <div className="page narrow">
      <section className="page-title">
        <p className="eyebrow">Phương pháp dự kiến</p>
        <h1>Replication core và modern reassessment</h1>
      </section>
      <section className="prose">{renderMarkdown(methods)}</section>
      <section className="prose">
        <h2>Deviation và giới hạn</h2>
        {renderMarkdown(deviations)}
      </section>
    </div>
  );
}
