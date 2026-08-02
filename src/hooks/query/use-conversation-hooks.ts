import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useConversationId } from "../use-conversation-id";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CONVERSATION_HOOKS_QUERY_KEYS } from "./query-keys";

export const useConversationHooks = () => {
  const { conversationId } = useConversationId();
  const { curAgentState } = useAgentState();
  const { backend, orgId } = useActiveBackend();
  return useQuery({
    queryKey: CONVERSATION_HOOKS_QUERY_KEYS.detail(
      backend.id,
      orgId ?? null,
      conversationId ?? null,
    ),
    queryFn: async () => {
      if (!conversationId) {
        throw new Error("No conversation ID provided");
      }

      const { hooks } =
        await AgentServerConversationService.getHooks(conversationId);
      return hooks;
    },
    enabled:
      !!conversationId &&
      curAgentState !== AgentState.LOADING &&
      curAgentState !== AgentState.INIT,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
};
