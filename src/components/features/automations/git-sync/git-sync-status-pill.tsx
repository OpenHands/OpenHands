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
        tone === "success" &&
          "bg-[var(--oh-success)]/15 text-[var(--oh-success)]",
        tone === "warning" &&
          "bg-[var(--oh-warning)]/15 text-[var(--oh-warning)]",
        tone === "neutral" && "bg-surface-raised text-muted",
      )}
    >
      {label}
    </span>
  );
}
