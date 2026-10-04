import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";

interface CondenseConversationVariables {
  conversationId: string;
  conversationUrl: string | null;
  sessionApiKey: string | null;
}

/**
 * Manually compact ("condense") a conversation's context via
 * `POST /api/conversations/{id}/condense`. Refreshes the usage metrics
 * afterwards so the context meter reflects the shrunken history.
 */
export const useCondenseConversation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (variables: CondenseConversationVariables) =>
      AgentServerConversationService.condenseConversation(
        variables.conversationId,
        variables.conversationUrl,
        variables.sessionApiKey,
      ),
    onSettled: (_, __, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["conversation-metrics", variables.conversationId],
      });
    },
  });
};
