import { Tooltip } from "@heroui/react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";
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
  /** `AutomationRun.phase_updated_at` — when this phase was last written. */
  updatedAt?: string | null;
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
 *
 * `code` and `label` are independently optional in the service's contract, so
 * the last resort is the raw code — a code-only phase is a real phase and
 * must reach the screen. It is shown as stored rather than prettified: the
 * code is data like the label, and turning `poll_prs` into "Poll prs" would
 * invent English-shaped copy no automation author wrote.
 */
export function resolveRunPhaseText(
  t: (key: I18nKey) => string,
  code: string | null | undefined,
  label: string | null | undefined,
): string | null {
  const knownKey = code ? KNOWN_PHASE_CODES[code] : undefined;
  if (knownKey) return t(knownKey);
  return label || code || null;
}

/**
 * How long the run has been in this phase, as localized relative time.
 *
 * This is the half of the phase that separates progress from a stall: the
 * phase text alone says a run is "Running agent", and only its age says
 * whether it entered that phase seconds ago or forty minutes ago. Returns
 * `null` when the service reported no timestamp (or an unparseable one) —
 * an older service omits the field entirely, and an age nobody can compute
 * must not surface as "Invalid Date".
 */
export function formatRunPhaseAge(
  updatedAt: string | null | undefined,
  locale: string,
  t: (key: I18nKey, options?: Record<string, unknown>) => string,
): string | null {
  if (!updatedAt) return null;
  const parsed = new Date(updatedAt).getTime();
  if (Number.isNaN(parsed)) return null;
  return formatRelativeTime(updatedAt, locale, t);
}

/**
 * A run's current or last-known phase and how long it has held it, clipped to
 * the room the surface has with the full text one hover away — author-supplied
 * labels routinely outgrow any row. The label is data, not interface copy, so
 * it is rendered as-is: passing it through `t()` would be wrong, it is not a
 * key. The age sits outside the clipped text so a long label can never push it
 * out of sight — it is the part that stays legible when everything else is cut.
 */
export function RunPhase({
  code,
  label,
  updatedAt,
  wide = false,
}: RunPhaseProps) {
  const { t, i18n } = useTranslation("openhands");

  const text = resolveRunPhaseText(t, code, label);
  if (!text) return null;

  const age = formatRunPhaseAge(updatedAt, i18n.language, t);

  return (
    <Tooltip
      content={
        <>
          {text}
          {age ? <span className="mt-1 block text-muted">{age}</span> : null}
        </>
      }
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
        {age ? (
          <span
            data-testid="run-phase-age"
            className="shrink-0 whitespace-nowrap text-xs text-muted"
          >
            · {age}
          </span>
        ) : null}
      </span>
    </Tooltip>
  );
}
