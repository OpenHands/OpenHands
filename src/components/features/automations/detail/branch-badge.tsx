interface BranchBadgeProps {
  branch: string;
}

export function BranchBadge({ branch }: BranchBadgeProps) {
  return (
    <span className="inline-flex items-center rounded-full border border-[var(--oh-border)] bg-surface-raised px-2.5 py-0.5 text-xs text-muted">
      {branch}
    </span>
  );
}
