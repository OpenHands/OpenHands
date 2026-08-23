import type { ReactNode } from "react";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

export const automationPreviewListClassName =
  "custom-scrollbar-always mx-6 flex min-h-0 flex-1 flex-col divide-y divide-[var(--oh-border)] overflow-y-auto rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] py-4";

type AutomationPreviewFieldLayout = "inline" | "stacked";

interface AutomationPreviewFieldProps {
  label: string;
  value: string;
  icon?: ReactNode;
  chips?: string[];
  /** Short rows sit label | value. Long copy (prompt, feeds) stays stacked. */
  layout?: AutomationPreviewFieldLayout;
}

const WRAP_CHAR_THRESHOLD = 48;

/** Label/value row used by the import preview and setup review. */
export function AutomationPreviewField({
  label,
  value,
  icon,
  chips,
  layout = "inline",
}: AutomationPreviewFieldProps) {
  const wraps =
    layout === "stacked" ||
    value.includes("\n") ||
    (chips?.length ?? 0) > 1 ||
    (chips?.some((chip) => chip.length > WRAP_CHAR_THRESHOLD) ?? false) ||
    (!chips && value.length > WRAP_CHAR_THRESHOLD);
  const stacked = wraps;

  return (
    <div
      className={cn(
        "px-4 py-4 first:pt-0 last:pb-0",
        stacked
          ? "flex flex-col gap-1"
          : "flex items-start justify-between gap-4",
      )}
    >
      <dt className="flex shrink-0 items-center gap-2 text-xs font-medium text-muted">
        {icon ? (
          <span className="size-3.5 shrink-0 text-muted" aria-hidden>
            {icon}
          </span>
        ) : null}
        {label}
      </dt>
      <dd
        className={cn(
          "min-w-0 text-xs text-content",
          chips?.length
            ? cn(
                "flex w-full flex-wrap gap-1.5",
                stacked ? "justify-start" : "justify-end",
              )
            : cn(
                "whitespace-pre-wrap break-words",
                stacked ? "text-left" : "text-right",
              ),
        )}
      >
        {chips?.length
          ? chips.map((chip) => (
              <span
                key={chip}
                className={cn(
                  extensionModuleCardPillClassName,
                  // Pill chrome is nowrap + hug-content. Long chips must be able
                  // to wrap: max-width wins over content, and min-width must not
                  // be max-content (that beats max-width and keeps one line).
                  "inline-block w-max max-w-full min-w-0 shrink-0 rounded-xl whitespace-normal break-words px-2.5 py-1 text-left text-xs leading-4 hyphens-none",
                )}
              >
                {chip}
              </span>
            ))
          : value}
      </dd>
    </div>
  );
}
