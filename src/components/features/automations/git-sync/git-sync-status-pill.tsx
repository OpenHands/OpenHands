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
        tone === "success" && statusToneBadgeClassName.success,
        tone === "warning" && statusToneBadgeClassName.warning,
        tone === "neutral" && "bg-surface-raised text-muted",
      )}
    >
      {label}
    </span>
  );
}
