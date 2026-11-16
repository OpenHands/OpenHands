import { useMutation, useQueryClient } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  persistHomeAcpSelection,
  refreshAfterAcpSelection,
} from "./acp-selection";

interface SwitchAcpEffortVars {
  /** Live switch for this conversation; null saves the home launch default. */
  conversationId: string | null;
  /** Null saves no effort, so the next conversation starts on the agent's own. */
  effort: string | null;
}

/** ACP reasoning-effort counterpart of {@link useSwitchAcpModel}. */
export const useSwitchAcpEffort = () => {
  const queryClient = useQueryClient();
  const { backend, orgId } = useActiveBackend();

  return useMutation({
    mutationFn: async ({ conversationId, effort }: SwitchAcpEffortVars) => {
      if (conversationId) {
        if (effort) {
          await AgentServerConversationService.switchAcpEffort(
            conversationId,
            effort,
          );
        }
        return;
      }
      await persistHomeAcpSelection(queryClient, backend.id, orgId, {
        acp_effort: effort,
      });
    },
    onSuccess: (_data, { conversationId }) =>
      refreshAfterAcpSelection(queryClient, conversationId),
  });
};
