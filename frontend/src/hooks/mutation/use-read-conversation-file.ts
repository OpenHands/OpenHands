import { useMutation } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

interface UseReadConversationFileVariables {
  conversationId: string;
  filePath?: string;
}

export const useReadConversationFile = () =>
  useMutation({
    mutationKey: ["read-conversation-file"],
    mutationFn: async ({
      conversationId,
      filePath,
    }: UseReadConversationFileVariables): Promise<string> =>
      V1ConversationService.readConversationFile(conversationId, filePath),
  });
