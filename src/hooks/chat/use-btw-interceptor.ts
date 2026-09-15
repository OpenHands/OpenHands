import { useCallback } from "react";
import { askAgent } from "#/hooks/mutation/conversation-mutation-utils";
import { useBtwStore } from "#/stores/btw-store";
import { AgentState } from "#/types/agent-state";
import { BTW_COMMAND } from "#/utils/constants";

const BTW_PREFIX = `${BTW_COMMAND} `;

/**
 * Intercepts "/btw <question>" submissions and routes them through the
 * ask_agent side-channel. Everything else falls through to `onSubmit`.
 * Passthrough when `conversationId` is null.
 * Swallows the command while the agent is running: ask_agent races the
 * live run and the server answers 500, so the question is dropped and the
 * user retries once the agent is idle (same convention as /plan and /code).
 */
export const useBtwInterceptor = (
  conversationId: string | null | undefined,
  curAgentState: AgentState,
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
      if (curAgentState === AgentState.RUNNING) {
        return;
      }
      const question = trimmed.slice(BTW_COMMAND.length).trim();
      if (!question) return;

      const entryId = addPending(conversationId, question);
      askAgent(conversationId, question)
        .then(({ response }) => resolve(conversationId, entryId, response))
        .catch((err) =>
          fail(conversationId, entryId, err?.message ?? "Failed to ask agent"),
        );
    },
    [conversationId, curAgentState, onSubmit, addPending, resolve, fail],
  );
};
