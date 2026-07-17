import type { PhaseStatus } from "@/lib/content";

export function StatusBadge({ status }: { status: PhaseStatus }) {
  return <span className={`status status-${status.toLowerCase().replace(" ", "-")}`}>{status}</span>;
}
