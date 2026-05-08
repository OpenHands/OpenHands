import { useMutation, useQueryClient } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

interface SwitchLlmProfileVars {
  conversationId: string;
  profileName: string;
}

export const useSwitchLlmProfile = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ conversationId, profileName }: SwitchLlmProfileVars) =>
      V1ConversationService.switchProfile(conversationId, profileName),
    onSuccess: (_data, { conversationId }) => {
      // Refetch the conversation so the chat header (and anything else
      // reading `conversation.llm_model`) picks up the new model. The
      // backend persisted it as part of the switch.
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", conversationId],
      });
    },
    meta: { disableToast: true },
  });
};
