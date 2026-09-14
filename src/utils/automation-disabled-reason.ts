import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

/** Detail `source` values written by the automation service on disable. */
const MANUAL_DETAIL_SOURCES = new Set(["user"]);
/** Top-level `disabled_reason` values written for user-initiated disables. */
const MANUAL_REASON_VALUES = new Set(["manual", "manual_delete"]);

/**
 * Whether an automation is inactive *because* of a recorded disablement
 * reason (vs. simply toggled off by an older client path, or an automation
 * service that predates disablement tracking).
 *
 * The automation service overwrites `disabled_reason`/`disabled_detail`/
 * `disabled_at` on every disable and clears them on every re-enable, so when
 * an automation is `enabled === false` and carries a `disabled_reason`, that
 * reason is the latest reason it went inactive.
 */
export function hasDisablementReason(
  automation: Pick<Automation, "enabled" | "disabled_reason">,
): boolean {
  return !automation.enabled && Boolean(automation.disabled_reason);
}

/**
 * Whether the recorded disablement was user-initiated (manual toggle / delete)
 * rather than an automatic pause for failing runs.
 */
export function isManualDisable(
  automation: Pick<Automation, "disabled_reason" | "disabled_detail">,
): boolean {
  const reason = automation.disabled_reason;
  if (reason && MANUAL_REASON_VALUES.has(reason)) return true;
  const source = automation.disabled_detail?.source;
  if (source && MANUAL_DETAIL_SOURCES.has(source)) return true;
  return false;
}

export interface DisablementReasonDisplay {
  /** i18n key for the short label shown next to the badge / in lists. */
  labelKey: I18nKey;
  /**
   * Full reason text to render. For automatic disables this is the backend's
   * human-readable `disabled_reason` (already phrased for users, e.g.
   * "Paused automatically: auth — ..."). For manual disables the backend
   * stores the opaque "manual" string, so we substitute a localized label.
   */
  text: string;
}

/**
 * Resolve the disablement reason into display text. Returns `null` when the
 * automation is not inactive-for-a-recorded-reason.
 */
export function getDisablementReasonDisplay(
  automation: Pick<
    Automation,
    "enabled" | "disabled_reason" | "disabled_detail"
  >,
  t: (key: I18nKey) => string,
): DisablementReasonDisplay | null {
  if (!hasDisablementReason(automation)) return null;
  if (isManualDisable(automation)) {
    return {
      labelKey: I18nKey.AUTOMATIONS$DETAIL$DISABLED_MANUAL,
      text: t(I18nKey.AUTOMATIONS$DETAIL$DISABLED_MANUAL),
    };
  }
  return {
    labelKey: I18nKey.AUTOMATIONS$DETAIL$DISABLED_AUTO,
    text: automation.disabled_reason as string,
  };
}
