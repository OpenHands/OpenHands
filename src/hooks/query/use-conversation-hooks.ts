import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useOptionalConversationId } from "../use-conversation-id";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";

export const useConversationHooks = (
  conversationIdOverride?: string | null,
) => {
  const { conversationId: routeConversationId } = useOptionalConversationId();
  const conversationId = conversationIdOverride ?? routeConversationId;
  const { curAgentState } = useAgentState();
  return useQuery({
    queryKey: ["conversation", conversationId, "hooks"],
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
