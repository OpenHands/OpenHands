import { useCallback } from "react";
import { FEEDBACK_COMMAND } from "#/utils/constants";

/** The existing anonymous feedback form, from issue #15583. */
export const FEEDBACK_FORM_URL = "https://forms.gle/chHc5VdS3wty5DwW6";

const FEEDBACK_PREFIX = `${FEEDBACK_COMMAND} `;

/**
 * Intercepts "/feedback" and opens the anonymous feedback form in a new tab.
 * No backend involvement. Everything else falls through to `onSubmit`.
 * Passthrough when `conversationId` is null.
 */
export const useFeedbackInterceptor = (
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) => {
  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isFeedback =
        trimmed === FEEDBACK_COMMAND || trimmed.startsWith(FEEDBACK_PREFIX);
      if (!conversationId || !isFeedback) {
        onSubmit(message);
        return;
      }

      window.open(FEEDBACK_FORM_URL, "_blank");
    },
    [conversationId, onSubmit],
  );
};
