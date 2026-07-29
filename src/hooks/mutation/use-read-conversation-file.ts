import { useMutation } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";

interface UseReadConversationFileVariables {
  conversationId: string;
  filePath?: string;
}

export const useReadConversationFile = () =>
  useMutation({
    mutationKey: ["read-conversation-file"],
    // Reading PLAN.md is an existence check that legitimately 404s when no plan
    // exists yet; callers handle that locally. Opt out of the global
    // MutationCache error toast (query-client-config) so the 404 isn't shown.
    meta: { disableToast: true },
    mutationFn: async ({
      conversationId,
      filePath,
    }: UseReadConversationFileVariables): Promise<string> =>
      AgentServerConversationService.readConversationFile(
        conversationId,
        filePath,
      ),
  });
