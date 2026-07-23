import type { PhaseStatus } from "@/lib/content";

const statusLabels: Record<PhaseStatus, string> = {
  planned: "Planned",
  next: "Next",
  in_progress: "In progress",
  completed: "Completed",
  blocked: "Blocked",
  deferred: "Deferred",
};

export function StatusBadge({ status }: { status: PhaseStatus }) {
  return <span className={`status status-${status.replace("_", "-")}`}>{statusLabels[status]}</span>;
}
