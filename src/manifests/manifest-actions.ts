/**
 * The action bridge — the only channel through which a manifest can make this
 * host *do* something.
 *
 * A manifest is data from another repository, so it never describes a request.
 * It chooses between the two outcomes this host offers, and it chooses by
 * declaring a `mode`: a direct entry produces a create request the host derives,
 * an assisted entry hands setup to a conversation.
 */

import { useCallback } from "react";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useConversationStore } from "#/stores/conversation-store";
import {
  setConversationState,
  setPendingTaskDraft,
} from "#/utils/conversation-local-storage";
import { buildAssistedMessage } from "./automation-setup";
import type { SetupEntry, SetupFormValues, SetupRequestBody } from "./types";

export interface SetupActionResult {
  /** The created resource, or the conversation that will finish setup. */
  response: Record<string, unknown>;
}

export function useSetupAction() {
  const createConversation = useCreateConversation();
  const setMessageToSend = useConversationStore(
    (state) => state.setMessageToSend,
  );

  const startConversation = useCallback(
    async (message: string): Promise<SetupActionResult> => {
      const conversation = await createConversation.mutateAsync({});

      // Seed the message the same way the rest of the app does, so the
      // conversation opens with it queued whichever launch path applies.
      if (
        conversation.conversation_id.startsWith("task-") &&
        conversation.task_id
      ) {
        setPendingTaskDraft(conversation.task_id, message);
      } else {
        setConversationState(conversation.conversation_id, {
          draftMessage: message,
        });
      }
      window.setTimeout(() => setMessageToSend(message), 0);

      return { response: { ...conversation } };
    },
    [createConversation, setMessageToSend],
  );

  return useCallback(
    async (
      entry: SetupEntry,
      values: SetupFormValues,
      /** Present for a direct entry, and absent for an assisted one. */
      payload: SetupRequestBody | null,
    ): Promise<SetupActionResult> => {
      if (!payload) {
        return startConversation(buildAssistedMessage(entry, values));
      }
      const response = await AutomationService.createAutomationDraft(payload);
      return { response };
    },
    [startConversation],
  );
}
