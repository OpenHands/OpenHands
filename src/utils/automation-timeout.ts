import { I18nKey } from "#/i18n/declaration";

/**
 * Per-automation run timeout defaults. The deployment reports the maximum it
 * enforces through capability discovery.
 */
export const AUTOMATION_TIMEOUT_DEFAULT_SECONDS = 600; // 10 minutes

export type AutomationTimeoutValidation =
  | { value: number | null }
  | { errorKey: I18nKey };

/**
 * Validate a raw timeout string from the edit form. A blank string means "use
 * the server default" (resolved as `null`). Otherwise the value must be a
 * positive integer. When capability discovery supplies a ceiling, values above
 * it are rejected before the request reaches the service.
 */
export function validateAutomationTimeout(
  raw: string,
  maxSeconds?: number,
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
  if (maxSeconds !== undefined && seconds > maxSeconds) {
    return { errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_MAX_EXCEEDED };
  }
  return { value: seconds };
}
