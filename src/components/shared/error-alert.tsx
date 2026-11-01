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
        "rounded-md border border-status-error/40 bg-status-error/10 p-3 text-sm text-status-error whitespace-pre-wrap break-words",
        className,
      )}
    >
      {children}
    </div>
  );
}
