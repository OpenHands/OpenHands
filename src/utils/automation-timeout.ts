import { I18nKey } from "#/i18n/declaration";

/**
 * Per-automation run timeout bounds. These mirror the automation server
 * (`openhands/automation/utils/timeout.py`): a `null`/omitted timeout falls back
 * to the default, and the server rejects anything above the maximum.
 */
export const AUTOMATION_TIMEOUT_DEFAULT_SECONDS = 600; // 10 minutes
export const AUTOMATION_TIMEOUT_MAX_SECONDS = 1800; // 30 minutes

export type AutomationTimeoutValidation =
  | { value: number | null }
  | { errorKey: I18nKey };

/**
 * Validate a raw timeout string from the edit form. A blank string means "use
 * the server default" (resolved as `null`). Otherwise the value must be a
 * positive integer no greater than {@link AUTOMATION_TIMEOUT_MAX_SECONDS}.
 */
export function validateAutomationTimeout(
  raw: string,
): AutomationTimeoutValidation {
  const trimmed = raw.trim();
  if (!trimmed) return { value: null };

  const seconds = Number(trimmed);
  if (!Number.isInteger(seconds)) {
    return { errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_INVALID_NUMBER };
  }
  if (seconds <= 0) {
    return { errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_POSITIVE };
  }
  if (seconds > AUTOMATION_TIMEOUT_MAX_SECONDS) {
    return { errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_MAX_EXCEEDED };
  }
  return { value: seconds };
}
