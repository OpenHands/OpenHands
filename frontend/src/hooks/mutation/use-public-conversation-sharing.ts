import { useMutation, useQueryClient } from "@tanstack/react-query";
import { PublicConversationService, PublicSharingRequest } from "#/api/public-conversation-service";

export const useUpdatePublicSharing = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ conversationId, data }: { conversationId: string; data: PublicSharingRequest }) =>
      PublicConversationService.updatePublicSharing(conversationId, data),
    onSuccess: (_, { conversationId }) => {
      // Invalidate the public sharing query for this conversation
      queryClient.invalidateQueries({
        queryKey: ["public-sharing", conversationId],
      });
    },
  });
};
