import { Tooltip } from "@heroui/react";
import { AutomationRunStatus } from "#/types/automation";
import { cn } from "#/utils/utils";

/**
 * Whether a run's phase is worth showing at all: it answers "what is it doing
 * now" or "where did it stop", so a finished, cancelled or skipped run has
 * nothing to add. Shared by every surface, so one run cannot show a phase on
 * one screen and hide it on another.
 */
export function shouldShowRunPhase(
  status: AutomationRunStatus | null | undefined,
): boolean {
  return (
    status === AutomationRunStatus.FAILED ||
    status === AutomationRunStatus.PENDING ||
    status === AutomationRunStatus.RUNNING
  );
}

/**
 * The one place a stored phase becomes text, so a run cannot read one way on
 * a card and another way in that card's own tooltip.
 *
 * Automation ≥1.9.0 sends `current_phase` as already user-facing copy — not a
 * translation key, never passed through `t()`. Whitespace-only is treated as
 * absent: the service stores such a field as sent, and rendering it would put
 * an empty span on the row.
 */
export function resolveRunPhaseText(
  currentPhase?: string | null,
): string | null {
  return currentPhase?.trim() || null;
}

interface RunPhaseProps {
  /** `AutomationRun.current_phase` — `null`/absent means no phase reported. */
  currentPhase?: string | null;
  /** More width before clipping, for rows far wider than a card. */
  wide?: boolean;
}

/**
 * A run's current or last-known phase, clipped to the room the surface has
 * with the full text one hover away. The string is data, not interface copy,
 * so it is rendered as-is.
 *
 * The text stays in the accessibility tree rather than hiding behind an
 * accessible name on a focusable wrapper. Every surface nests this inside a
 * link — the activity log's `<a>`, the cards' `role="link"` — where a tab
 * stop is invalid interactive nesting, and Enter on it bubbled to the card
 * and navigated the user away from the text they were trying to read.
 * Truncation is CSS only, so the full label is already in the DOM and reads
 * in full; the tooltip is the sighted mouse user's route to it.
 */
export function RunPhase({ currentPhase, wide = false }: RunPhaseProps) {
  const text = resolveRunPhaseText(currentPhase);
  if (!text) return null;

  return (
    <Tooltip
      content={text}
      placement="top"
      closeDelay={100}
      disableAnimation={import.meta.env.MODE === "test"}
      classNames={{
        content:
          "max-w-xs whitespace-pre-wrap break-words rounded-xl border border-[var(--oh-border)] bg-base-secondary px-3 py-2 text-left text-xs text-white shadow-xl",
      }}
    >
      <span className="flex min-w-0 cursor-default items-center gap-1">
        <span
          data-testid="run-phase"
          className={cn(
            "min-w-0 truncate text-xs text-muted",
            wide ? "max-w-[28rem]" : "max-w-[12rem]",
          )}
        >
          {text}
        </span>
      </span>
    </Tooltip>
  );
}
