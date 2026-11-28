import { useCallback } from "react";
import { condenseConversation } from "#/hooks/mutation/conversation-mutation-utils";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { CONDENSE_COMMAND } from "#/utils/constants";

const CONDENSE_PREFIX = `${CONDENSE_COMMAND} `;

/**
 * Intercepts "/condense" and forces condensation ("compact") of the
 * conversation history via the agent-server condensation endpoint
 * (`POST /api/conversations/{id}/condense`, exposed to the client as
 * `AgentServerConversationService.condenseConversation()`). A failure is
 * surfaced as an error toast instead of failing silently. Everything else
 * falls through to `onSubmit`. Passthrough when `conversationId` is null.
 */
export const useCondenseInterceptor = (
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) => {
  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isCondense =
        trimmed === CONDENSE_COMMAND || trimmed.startsWith(CONDENSE_PREFIX);
      if (!conversationId || !isCondense) {
        onSubmit(message);
        return;
      }

      condenseConversation(conversationId)
        .then(() => displaySuccessToast("Conversation history condensed."))
        .catch((err: unknown) =>
          displayErrorToast(
            err instanceof Error && err.message
              ? err.message
              : "Failed to condense the conversation.",
          ),
        );
    },
    [conversationId, onSubmit],
  );
};
