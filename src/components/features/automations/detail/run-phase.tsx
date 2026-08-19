import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

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
 * Renders a run's current/last-known phase: a translated label for a code
 * the frontend recognizes, the author-supplied `phase_label` verbatim for an
 * unrecognized code, or nothing at all when there is no code, or the code is
 * unrecognized and the label is empty.
 *
 * `phase_label` is free-form data from an automation author (not interface
 * copy) and is rendered as-is — routing it through `t()` would be wrong,
 * since it is not a translation key.
 */
export function RunPhase({ code, label }: RunPhaseProps) {
  const { t } = useTranslation("openhands");

  if (!code) return null;

  const knownKey = KNOWN_PHASE_CODES[code];
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
