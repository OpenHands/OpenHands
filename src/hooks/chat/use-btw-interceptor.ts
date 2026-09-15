import { useCallback } from "react";
import { withRetry } from "#/api/with-retry";
import { askAgent } from "#/hooks/mutation/conversation-mutation-utils";
import { useBtwStore } from "#/stores/btw-store";
import { BTW_COMMAND } from "#/utils/constants";

const BTW_PREFIX = `${BTW_COMMAND} `;

/**
 * Whether a failed ask_agent call is worth retrying. Only retry when the
 * server positively responded with a transient status (5xx such as the
 * intermittent "cannot pickle 'generator' object" 500, or 429), which may
 * succeed on a later attempt. Anything else — client errors, or failures
 * without an HTTP status — fails fast so a repeated request cannot
 * double-execute agent work for a failure we do not understand.
 */
const isRetryableBtwError = (error: unknown): boolean => {
  const status =
    typeof error === "object" && error !== null
      ? (error as { status?: unknown }).status
      : undefined;
  return typeof status === "number" && (status === 429 || status >= 500);
};

/**
 * Intercepts "/btw <question>" submissions and routes them through the
 * ask_agent side-channel. Everything else falls through to `onSubmit`.
 * Passthrough when `conversationId` is null.
 */
export const useBtwInterceptor = (
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) => {
  const addPending = useBtwStore((s) => s.addPending);
  const resolve = useBtwStore((s) => s.resolve);
  const fail = useBtwStore((s) => s.fail);

  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isBtw = trimmed === BTW_COMMAND || trimmed.startsWith(BTW_PREFIX);
      if (!conversationId || !isBtw) {
        onSubmit(message);
        return;
      }
      const question = trimmed.slice(BTW_COMMAND.length).trim();
      if (!question) return;

      const entryId = addPending(conversationId, question);
      // Retry once on transient agent-server failures: the /btw side-channel
      // intermittently 500s (e.g. while the agent is streaming), and a later
      // attempt typically succeeds. Client errors fail fast without retrying.
      withRetry(
        () => askAgent(conversationId, question),
        2,
        500,
        isRetryableBtwError,
      )
        .then(({ response }) => resolve(conversationId, entryId, response))
        .catch((err) =>
          fail(conversationId, entryId, err?.message ?? "Failed to ask agent"),
        );
    },
    [conversationId, onSubmit, addPending, resolve, fail],
  );
};
