import { useMutation, useQueryClient } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

interface SwitchAcpModelVars {
  conversationId: string;
  model: string;
}

export const useSwitchAcpModel = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ conversationId, model }: SwitchAcpModelVars) =>
      V1ConversationService.switchAcpModel(conversationId, model),
    onSuccess: (_data, { conversationId }) => {
      // Refetch so the chat header picks up the new model (backend persisted it).
      queryClient.invalidateQueries({
        queryKey: ["user", "conversation", conversationId],
      });
    },
    meta: { disableToast: true },
  });
};
