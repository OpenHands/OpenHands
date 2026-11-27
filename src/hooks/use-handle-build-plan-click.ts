import { useCallback } from "react";
import { useConversationStore } from "#/stores/conversation-store";
import { useSendMessage } from "#/hooks/use-send-message";
import { createChatMessage } from "#/services/chat-service";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";

/**
 * Custom hook that encapsulates the logic for handling the Build button click.
 * Switches to code mode and sends a prompt to execute the plan.
 *
 * @returns An object containing handleBuildClick function
 */
export const useHandleBuildPlanClick = () => {
  const { setConversationMode } = useConversationStore();
  const { send } = useSendMessage();
  const { conversationId } = useOptionalConversationId();
  const enqueuePendingMessage = useOptimisticUserMessageStore(
    (state) => state.enqueuePendingMessage,
  );
  const markPendingMessageError = useOptimisticUserMessageStore(
    (state) => state.markPendingMessageError,
  );

  const handleBuildPlanClick = useCallback(
    (event?: React.MouseEvent<HTMLButtonElement> | KeyboardEvent) => {
      event?.preventDefault();
      event?.stopPropagation();

      // Switch to code mode
      setConversationMode("code");

      // Create the build prompt to execute the plan
      const buildPrompt = `Execute the plan based on the .agents_tmp/PLAN.md file.`;

      // Show the prompt as a pending message and send it to the code agent.
      // Skip the pending bubble if we somehow don't have a conversation id —
      // the message still gets sent, just without the optimistic queue entry.
      const timestamp = new Date().toISOString();
      const pendingId = conversationId
        ? enqueuePendingMessage({
            conversationId,
            text: buildPrompt,
            timestamp,
          })
        : null;
      send(createChatMessage(buildPrompt, [], [], timestamp)).catch((error) => {
        if (!pendingId) return;
        const errorMessage =
          error instanceof Error ? error.message : "Failed to send message";
        markPendingMessageError(pendingId, errorMessage);
      });
    },
    [
      setConversationMode,
      send,
      conversationId,
      enqueuePendingMessage,
      markPendingMessageError,
    ],
  );

  return { handleBuildPlanClick };
};
