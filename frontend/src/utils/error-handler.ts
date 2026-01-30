import type { PostHog } from "posthog-js";
import { handleStatusMessage } from "#/services/actions";
import { displayErrorToast } from "./custom-toast-handlers";

interface ErrorDetails {
  message: string;
  source?: string;
  metadata?: Record<string, unknown>;
  msgId?: string;
  posthog?: PostHog;
  userEmail?: string | null;
}

export function trackError({
  message,
  source,
  metadata = {},
  posthog,
  userEmail,
}: ErrorDetails) {
  if (!posthog) return;

  const error = new Error(message);
  posthog.captureException(error, {
    error_source: source || "unknown",
    user_email: userEmail ?? null,
    ...metadata,
  });
}

export function showErrorToast({
  message,
  source,
  metadata = {},
  posthog,
  userEmail,
}: ErrorDetails) {
  trackError({ message, source, metadata, posthog, userEmail });
  displayErrorToast(message);
}

export function showChatError({
  message,
  source,
  metadata = {},
  msgId,
  posthog,
  userEmail,
}: ErrorDetails) {
  trackError({ message, source, metadata, posthog, userEmail });
  handleStatusMessage({
    type: "error",
    message,
    id: msgId,
    status_update: true,
  });
}

/**
 * Checks if an error message indicates a budget or credit limit issue
 */
export function isBudgetOrCreditError(errorMessage: string): boolean {
  const lowerCaseError = errorMessage.toLowerCase();
  return lowerCaseError.includes("budget") || lowerCaseError.includes("credit");
}
