import { useMutation } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";

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
      AgentServerConversationService.readConversationFile(
        conversationId,
        filePath,
      ),
  });
