import type { ReactNode } from "react";
import { cn } from "#/utils/utils";

type ConversationTabEmptyStateProps = {
  icon: ReactNode;
  children: ReactNode;
  action?: ReactNode;
  className?: string;
};

/**
 * Shared empty state for right-drawer conversation tabs: small muted icon,
 * centered caption, optional action (use {@link BrandButton} variant="secondary").
 */
export function ConversationTabEmptyState({
  icon,
  children,
  action,
  className,
}: ConversationTabEmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center w-full h-full p-10 gap-4 text-center",
        className,
      )}
    >
      <div
        className="shrink-0 text-[var(--oh-muted)] [&_svg]:size-10 [&_svg]:max-h-10 [&_svg]:max-w-10 [&_svg]:shrink-0"
        aria-hidden
      >
        {icon}
      </div>
      <p className="max-w-sm text-center text-sm font-normal leading-5 text-[var(--oh-muted)]">
        {children}
      </p>
      {action ? <div className="flex justify-center pt-1">{action}</div> : null}
    </div>
  );
}
