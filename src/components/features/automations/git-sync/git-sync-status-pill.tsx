import { cn } from "#/utils/utils";
import { statusToneBadgeClassName } from "#/utils/status-tone-classes";

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
          cn(
            statusToneBadgeClassName.success,
            "dark:bg-semantic-success/15 dark:text-semantic-success",
          ),
        tone === "warning" &&
          cn(
            statusToneBadgeClassName.warning,
            "dark:bg-warning/15 dark:text-warning",
          ),
        tone === "neutral" && "bg-surface-raised text-muted",
      )}
    >
      {label}
    </span>
  );
}
