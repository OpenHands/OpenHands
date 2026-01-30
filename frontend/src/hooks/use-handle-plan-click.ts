import { useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";

/**
 * Custom hook that encapsulates the logic for handling plan creation.
 * Returns a function that can be called to create a plan conversation and
 * the pending state of the conversation creation.
 *
 * @returns An object containing handlePlanClick function and isCreatingConversation boolean
 */
export const useHandlePlanClick = () => {
  const { t } = useTranslation();
  const {
    setConversationMode,
    setSubConversationTaskId,
    subConversationTaskId,
  } = useConversationStore();
  const { data: conversation } = useActiveConversation();
  const { mutate: createConversation, isPending: isCreatingConversation } =
    useCreateConversation();

  // Restore subConversationTaskId from localStorage on conversation load
  // This handles the case where page was refreshed while sub-conversation creation was in progress
  useEffect(() => {
    if (!conversation?.conversation_id) return;

    const storedState = getConversationState(conversation.conversation_id);
    if (storedState.subConversationTaskId && !subConversationTaskId) {
      setSubConversationTaskId(storedState.subConversationTaskId);
    }
  }, [
    conversation?.conversation_id,
    subConversationTaskId,
    setSubConversationTaskId,
  ]);

  const handlePlanClick = useCallback(
    (event?: React.MouseEvent<HTMLButtonElement> | KeyboardEvent) => {
      event?.preventDefault();
      event?.stopPropagation();

      // Set conversation mode to "plan" immediately
      setConversationMode("plan");

      // Check if sub_conversation_ids is not empty or if a sub-conversation creation is already in progress
      if (
        (conversation?.sub_conversation_ids &&
          conversation.sub_conversation_ids.length > 0) ||
        !conversation?.conversation_id ||
        subConversationTaskId
      ) {
        // Do nothing if any condition is true
        return;
      }

      // Create a new sub-conversation if we have a current conversation ID
      createConversation(
        {
          parentConversationId: conversation.conversation_id,
          agentType: "plan",
        },
        {
          onSuccess: (data) => {
            displaySuccessToast(
              t(I18nKey.PLANNING_AGENTT$PLANNING_AGENT_INITIALIZED),
            );
            // Track the task ID to poll for sub-conversation creation
            if (data.v1_task_id) {
              setSubConversationTaskId(data.v1_task_id);
              // Persist to localStorage so it survives page refresh
              setConversationState(conversation.conversation_id, {
                subConversationTaskId: data.v1_task_id,
              });
            }
          },
        },
      );
    },
    [
      conversation,
      createConversation,
      setConversationMode,
      setSubConversationTaskId,
      subConversationTaskId,
      t,
    ],
  );

  return { handlePlanClick, isCreatingConversation };
};
