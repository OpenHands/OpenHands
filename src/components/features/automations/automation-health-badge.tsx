import {
  HEALTH_LABEL_KEYS,
  type AutomationHealth,
} from "#/manifests/automation-insights";
import type { InterfaceListInsights } from "#/manifests/types";
import { cn } from "#/utils/utils";

/**
 * The states and their colors are the host's; the captions come from the
 * manifest's `insights.health` block.
 */
const HEALTH_STYLES: Record<AutomationHealth, string> = {
  healthy:
    "border-[var(--oh-success)]/50 bg-[var(--oh-success)]/10 text-[var(--oh-success)]",
  failing: "border-[var(--oh-danger)]/50 bg-[var(--oh-danger)]/10 text-danger",
  running: "border-[var(--oh-border)] bg-surface-raised text-muted",
  disabled: "border-[var(--oh-border)] bg-surface-raised text-muted",
  "never-run":
    "border-[var(--oh-warning)]/50 bg-[var(--oh-warning)]/10 text-[var(--oh-warning)]",
  unknown: "border-[var(--oh-border)] bg-surface-raised text-muted",
};

interface AutomationHealthBadgeProps {
  health: AutomationHealth;
  labels: InterfaceListInsights["health"];
}

export function AutomationHealthBadge({
  health,
  labels,
}: AutomationHealthBadgeProps) {
  return (
    <span
      data-testid="automation-health-badge"
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium",
        HEALTH_STYLES[health],
      )}
    >
      <span className="size-1.5 rounded-full bg-current" aria-hidden />
      {labels[HEALTH_LABEL_KEYS[health]]}
    </span>
  );
}
