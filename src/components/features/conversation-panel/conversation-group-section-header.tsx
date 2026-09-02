import { Box } from "lucide-react";

interface ConversationGroupSectionHeaderProps {
  label: string;
}

/**
 * Non-interactive heading rendered above a container's folders when both
 * "Group by container" and "Group by workspace" are active (#15607).
 *
 * Only the nested (both-toggles-on) view renders this — the single-axis
 * grouped views keep rendering their existing flat folder list unchanged,
 * with no header at all.
 */
export function ConversationGroupSectionHeader({
  label,
}: ConversationGroupSectionHeaderProps) {
  return (
    <div
      data-testid="conversation-group-section-header"
      className="flex min-w-0 items-center gap-1.5 px-2 pb-1 pt-3 first:pt-1"
    >
      <Box className="h-3 w-3 shrink-0 text-[var(--oh-muted)]" aria-hidden />
      <span className="min-w-0 truncate text-[11px] font-semibold uppercase tracking-wide text-[var(--oh-muted)]">
        {label}
      </span>
    </div>
  );
}
