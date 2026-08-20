import { Tooltip } from "@heroui/react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
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

interface RunPhaseProps {
  /** `AutomationRun.phase_code` — `null`/absent means no phase reported. */
  code: string | null | undefined;
  /** `AutomationRun.phase_label` — free-form author text, not interface copy. */
  label: string | null | undefined;
  /** More width before clipping, for rows far wider than a card. */
  wide?: boolean;
}

/**
 * Codes the frontend can translate: the automation service's own milestones
 * and the preset templates'. Any other code is by design unknown — custom
 * automations emit their own — and falls back to `phase_label`.
 */
const KNOWN_PHASE_CODES: Record<string, I18nKey> = {
  queued: I18nKey.AUTOMATIONS$DETAIL$PHASE_QUEUED,
  sandbox_provisioning: I18nKey.AUTOMATIONS$DETAIL$PHASE_SANDBOX_PROVISIONING,
  bundle_upload: I18nKey.AUTOMATIONS$DETAIL$PHASE_BUNDLE_UPLOAD,
  entrypoint_start: I18nKey.AUTOMATIONS$DETAIL$PHASE_ENTRYPOINT_START,
  preparing: I18nKey.AUTOMATIONS$DETAIL$PHASE_PREPARING,
  running_agent: I18nKey.AUTOMATIONS$DETAIL$PHASE_RUNNING_AGENT,
};

/**
 * The one place a stored phase becomes text, so a run cannot read one way on
 * a card and another way in that card's own tooltip. An absent code counts as
 * unrecognized rather than as an absent phase: the service accepts a phase
 * carrying only a label, and dropping those would hide a real phase.
 */
export function resolveRunPhaseText(
  t: (key: I18nKey) => string,
  code: string | null | undefined,
  label: string | null | undefined,
): string | null {
  const knownKey = code ? KNOWN_PHASE_CODES[code] : undefined;
  if (knownKey) return t(knownKey);
  return label || null;
}

/**
 * A run's current or last-known phase, clipped to the room the surface has
 * with the full text one hover away — author-supplied labels routinely
 * outgrow any row. The label is data, not interface copy, so it is rendered
 * as-is: passing it through `t()` would be wrong, it is not a key.
 */
export function RunPhase({ code, label, wide = false }: RunPhaseProps) {
  const { t } = useTranslation("openhands");

  const text = resolveRunPhaseText(t, code, label);
  if (!text) return null;

  return (
    <Tooltip
      content={text}
      placement="top"
      closeDelay={100}
      disableAnimation={import.meta.env.MODE === "test"}
      // The `content` slot, not `className`: HeroUI leaves the slot itself
      // transparent, so styling the component instead of the slot renders the
      // text straight onto whatever is behind the tooltip.
      classNames={{
        content:
          "max-w-xs whitespace-pre-wrap break-words rounded-xl border border-[var(--oh-border)] bg-base-secondary px-3 py-2 text-left text-xs text-white shadow-xl",
      }}
    >
      <span
        data-testid="run-phase"
        className={cn(
          "min-w-0 cursor-default truncate text-xs text-muted",
          wide ? "max-w-[28rem]" : "max-w-[12rem]",
        )}
      >
        {text}
      </span>
    </Tooltip>
  );
}
