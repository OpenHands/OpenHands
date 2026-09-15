import { cn } from "#/utils/utils";

interface GitSyncStatusPillProps {
  tone: "success" | "neutral" | "warning";
  label: string;
  testId?: string;
}

export function GitSyncStatusPill({
  tone,
  label,
  testId,
}: GitSyncStatusPillProps) {
  return (
    <span
      data-testid={testId}
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium",
        tone === "success" && "bg-success/15 text-success",
        tone === "warning" && "bg-warning/15 text-warning",
        tone === "neutral" && "bg-surface-raised text-muted",
      )}
    >
      {label}
    </span>
  );
}
