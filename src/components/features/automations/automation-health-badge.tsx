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
    "border-semantic-success/50 bg-semantic-success/10 text-semantic-success",
  failing: "border-semantic-danger/50 bg-semantic-danger/10 text-danger",
  running: "border-border bg-surface-raised text-muted",
  disabled: "border-border bg-surface-raised text-muted",
  "never-run": "border-warning/50 bg-warning/10 text-warning",
  unknown: "border-border bg-surface-raised text-muted",
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
