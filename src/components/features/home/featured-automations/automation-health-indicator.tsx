import { cn } from "#/utils/utils";
import type { AutomationRunHealth } from "./automation-run-health";

interface AutomationHealthIndicatorProps {
  health: AutomationRunHealth;
}

/**
 * Visual-only latest-run health mark. Decorative on purpose: the owning
 * control carries the translated status text (e.g. via sr-only), so the
 * indicator itself is hidden from assistive technology.
 */
export function AutomationHealthIndicator({
  health,
}: AutomationHealthIndicatorProps) {
  if (health === "success") {
    return (
      <svg
        viewBox="0 0 12 12"
        className="h-2.5 w-2.5 shrink-0 stroke-[var(--oh-status-success)]"
        fill="none"
        strokeWidth={2.25}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M2.5 6.5 5 9l4.5-5.5" />
      </svg>
    );
  }

  return (
    <span
      aria-hidden="true"
      className={cn(
        "h-1.5 w-1.5 shrink-0 rounded-full",
        health === "failed" && "bg-[var(--oh-status-error)]",
        health === "in_progress" &&
          "animate-pulse bg-[var(--oh-status-success)] motion-reduce:animate-none",
        (health === "none" || health === "unknown") && "bg-[var(--oh-border)]",
      )}
    />
  );
}
