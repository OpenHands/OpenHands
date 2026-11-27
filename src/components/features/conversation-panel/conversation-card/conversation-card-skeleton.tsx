interface ConversationCardSkeletonProps {
  compact?: boolean;
}

/**
 * Loading placeholders for the conversation list. Non-compact: three bars;
 * compact: three small bars for the icon rail. Pulse stagger comes from
 * `.skeleton-stagger` in `tailwind.css`.
 */
export function ConversationCardSkeleton({
  compact = false,
}: ConversationCardSkeletonProps) {
  if (compact) {
    return (
      <div
        data-testid="conversation-card-skeleton-compact"
        className="skeleton-stagger flex flex-col items-center gap-1.5 py-1"
        aria-hidden
      >
        {[0, 1, 2].map((i) => (
          <div
            key={`conversation-skeleton-compact-${i}`}
            className="h-1.5 w-7 shrink-0 skeleton"
          />
        ))}
      </div>
    );
  }

  return (
    <div
      data-testid="conversation-card-skeleton"
      className="skeleton-stagger flex flex-col gap-1.5 py-0.5"
      aria-hidden
    >
      {[0, 1, 2].map((i) => (
        <div
          key={`conversation-skeleton-row-${i}`}
          className="h-6 min-h-6 w-full skeleton"
        />
      ))}
    </div>
  );
}
