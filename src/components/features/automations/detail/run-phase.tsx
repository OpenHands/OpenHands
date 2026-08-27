import { Tooltip } from "@heroui/react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { AutomationRunStatus } from "#/types/automation";
import {
  formatRelativeTime,
  isInvalidTimestamp,
} from "#/utils/format-relative-time";
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
 * Whether the phase's age is worth showing. Only for a run still in flight:
 * the age exists to separate a moving run from a stalled one, and a run that
 * has failed is neither — its phase is the place it stopped. Left in, the age
 * of an old failure degrades to an absolute date (`formatRelativeTime` gives
 * up past a week), which reads as a second timestamp beside the one the row
 * already shows.
 */
export function shouldShowRunPhaseAge(
  status: AutomationRunStatus | null | undefined,
): boolean {
  return (
    status === AutomationRunStatus.PENDING ||
    status === AutomationRunStatus.RUNNING
  );
}

interface RunPhaseFields {
  /** `AutomationRun.status` — decides whether the age is meaningful. */
  status: AutomationRunStatus | null | undefined;
  /** `AutomationRun.phase_code` — `null`/absent means no phase reported. */
  code: string | null | undefined;
  /** `AutomationRun.phase_label` — free-form author text, not interface copy. */
  label: string | null | undefined;
  /** `AutomationRun.phase_updated_at` — when this phase was last written. */
  updatedAt?: string | null;
}

interface RunPhaseProps extends RunPhaseFields {
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
 *
 * Both fields are author-supplied, which is why the lookup is an own-property
 * check and both are trimmed. A code of `toString` would otherwise resolve to
 * `Object.prototype.toString` and be handed to `t()`, and the service stores
 * a whitespace-only field as sent — it rejects only a phase blank on *both*.
 */
export function resolveRunPhaseText(
  t: (key: I18nKey) => string,
  code: string | null | undefined,
  label: string | null | undefined,
): string | null {
  const knownKey =
    code && Object.hasOwn(KNOWN_PHASE_CODES, code)
      ? KNOWN_PHASE_CODES[code]
      : undefined;
  if (knownKey) return t(knownKey);
  return label?.trim() || code?.trim() || null;
}

/**
 * How long the run has been in this phase, as localized relative time.
 *
 * This is the half of the phase that separates progress from a stall: the
 * phase text alone says a run is "Running agent", and only its age says
 * whether it entered that phase seconds ago or forty minutes ago. Returns
 * `null` when the service reported no usable timestamp — an older service
 * omits the field entirely, and an unset datetime arrives as the epoch — so
 * an age nobody can compute never surfaces as "Invalid Date" or "Jan 1, 1970".
 */
export function formatRunPhaseAge(
  updatedAt: string | null | undefined,
  locale: string,
  t: (key: I18nKey, options?: Record<string, unknown>) => string,
): string | null {
  if (!updatedAt || isInvalidTimestamp(updatedAt)) return null;
  return formatRelativeTime(updatedAt, locale, t);
}

/**
 * The resolved phase of a run, or `null` when it has none worth showing.
 * Every surface goes through this — the clipped row below, the home
 * hovercard's wrapping one — so they cannot drift into resolving a phase, or
 * deciding to show its age, on their own terms.
 */
export function useRunPhase({
  status,
  code,
  label,
  updatedAt,
}: RunPhaseFields): { text: string; age: string | null } | null {
  const { t, i18n } = useTranslation("openhands");

  const text = resolveRunPhaseText(t, code, label);
  if (!text) return null;

  const age = shouldShowRunPhaseAge(status)
    ? formatRunPhaseAge(updatedAt, i18n.language, t)
    : null;

  return { text, age };
}

/**
 * A run's current or last-known phase and how long it has held it, clipped to
 * the room the surface has with the full text one hover away — author-supplied
 * labels routinely outgrow any row. The label is data, not interface copy, so
 * it is rendered as-is: passing it through `t()` would be wrong, it is not a
 * key. The age sits outside the clipped text so a long label can never push it
 * out of sight — it is the part that stays legible when everything else is cut.
 *
 * The text stays in the accessibility tree rather than hiding behind an
 * accessible name on a focusable wrapper. Every surface nests this inside a
 * link — the activity log's `<a>`, the cards' `role="link"` — where a tab
 * stop is invalid interactive nesting, and Enter on it bubbled to the card
 * and navigated the user away from the text they were trying to read.
 * Truncation is CSS only, so the full label is already in the DOM and reads
 * in full; the tooltip is the sighted mouse user's route to it.
 */
export function RunPhase({
  status,
  code,
  label,
  updatedAt,
  wide = false,
}: RunPhaseProps) {
  const phase = useRunPhase({ status, code, label, updatedAt });
  if (!phase) return null;

  const { text, age } = phase;

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
