import { useCallback } from "react";
import { useConversationStore } from "#/stores/conversation-store";
import { useSendMessage } from "#/hooks/use-send-message";
import { createChatMessage } from "#/services/chat-service";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";

/**
 * Custom hook that encapsulates the logic for handling the Build button click.
 * Switches to code mode and sends a prompt to execute the plan.
 *
 * @returns An object containing handleBuildClick function
 */
export const useHandleBuildPlanClick = () => {
  const { setConversationMode } = useConversationStore();
  const { send } = useSendMessage();
  const { setOptimisticUserMessage } = useOptimisticUserMessageStore();

  const handleBuildPlanClick = useCallback(
    (event?: React.MouseEvent<HTMLButtonElement> | KeyboardEvent) => {
      event?.preventDefault();
      event?.stopPropagation();

      // Switch to code mode
      setConversationMode("code");

      // Create the build prompt to execute the plan
      const buildPrompt = `Execute the plan based on the workspace/project/PLAN.md file.`;

      // Send the message to the code agent
      const timestamp = new Date().toISOString();
      send(createChatMessage(buildPrompt, [], [], timestamp));
      setOptimisticUserMessage(buildPrompt);
    },
    [setConversationMode, send, setOptimisticUserMessage],
  );

  return { handleBuildPlanClick };
};
