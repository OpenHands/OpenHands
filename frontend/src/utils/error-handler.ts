import { handleStatusMessage } from "#/services/actions";
import { displayErrorToast } from "./custom-toast-handlers";

interface ErrorDetails {
  message: string;
  source?: string;
  metadata?: Record<string, unknown>;
  msgId?: string;
}

// PostHog capture removed — error tracking is now handled server-side
export function trackError(
  details: ErrorDetails, // eslint-disable-line @typescript-eslint/no-unused-vars
): void {}

export function showErrorToast({
  message,
  source,
  metadata = {},
}: ErrorDetails) {
  trackError({ message, source, metadata });
  displayErrorToast(message);
}

export function showChatError({
  message,
  source,
  metadata = {},
  msgId,
}: ErrorDetails) {
  trackError({ message, source, metadata });
  handleStatusMessage({
    type: "error",
    message,
    id: msgId,
    status_update: true,
  });
}

/**
 * Checks if an error message indicates an OpenHands budget or credit limit issue.
 * Provider-side errors can also mention "credits"; those should stay as raw
 * provider errors instead of being rewritten as OpenHands billing failures.
 */
export type BudgetErrorType = "budget" | "credit" | null;

export function classifyBudgetOrCreditError(
  errorMessage: string,
): BudgetErrorType {
  const lowerCaseError = errorMessage.toLowerCase();
  const isOpenHandsCreditError =
    /openhands\s+credits?/.test(lowerCaseError) ||
    lowerCaseError.includes("credit limit reached");
  if (isOpenHandsCreditError) {
    return "credit";
  }

  const isBudgetLimitError =
    lowerCaseError.includes("budget") &&
    /(exceed|exceeded|limit|max budget|spent)/.test(lowerCaseError);
  if (isBudgetLimitError) {
    return "budget";
  }

  return null;
}
