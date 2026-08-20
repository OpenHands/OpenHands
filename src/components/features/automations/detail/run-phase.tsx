import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus } from "#/types/automation";

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
 * A run's current or last-known phase: the translation of a code the frontend
 * knows, otherwise the author-supplied label verbatim, and nothing at all
 * when neither is usable. An absent code counts as unrecognized rather than
 * as an absent phase — the service accepts a phase carrying only a label.
 *
 * The label is data, not interface copy, so it is rendered as-is: passing it
 * through `t()` would be wrong, it is not a key.
 */
export function RunPhase({ code, label }: RunPhaseProps) {
  const { t } = useTranslation("openhands");

  const knownKey = code ? KNOWN_PHASE_CODES[code] : undefined;
  if (knownKey) {
    return (
      <span
        data-testid="run-phase"
        className="min-w-0 max-w-[12rem] truncate text-xs text-muted"
      >
        {t(knownKey)}
      </span>
    );
  }

  if (!label) return null;

  return (
    <span
      data-testid="run-phase"
      className="min-w-0 max-w-[12rem] truncate text-xs text-muted"
    >
      {label}
    </span>
  );
}
