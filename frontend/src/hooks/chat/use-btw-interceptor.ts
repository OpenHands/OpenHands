import { useCallback } from "react";
import { askV1Agent } from "#/hooks/mutation/conversation-mutation-utils";
import { useBtwStore } from "#/stores/btw-store";
import { BTW_COMMAND } from "#/utils/constants";

const BTW_PREFIX = `${BTW_COMMAND} `;

/**
 * Intercepts "/btw <question>" submissions and routes them through the
 * ask_agent side-channel. Everything else falls through to `onSubmit`.
 * Passthrough when `conversationId` is null (e.g. V0 conversations).
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
      askV1Agent(conversationId, question)
        .then(({ response }) => resolve(conversationId, entryId, response))
        .catch((err) =>
          fail(conversationId, entryId, err?.message ?? "Failed to ask agent"),
        );
    },
    [conversationId, onSubmit, addPending, resolve, fail],
  );
};
