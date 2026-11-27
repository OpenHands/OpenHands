import type { ReactNode } from "react";
import { cn } from "#/utils/utils";

interface ErrorAlertProps {
  children: ReactNode;
  testId?: string;
  className?: string;
}

/** Inline error box announced to assistive tech; keeps server line breaks. */
export function ErrorAlert({ children, testId, className }: ErrorAlertProps) {
  return (
    <div
      role="alert"
      data-testid={testId}
      className={cn(
        "rounded-md border border-danger/40 bg-danger-soft p-3 text-sm text-danger-soft-foreground dark:border-status-error/40 dark:bg-status-error/10 dark:text-status-error whitespace-pre-wrap break-words",
        className,
      )}
    >
      {children}
    </div>
  );
}
