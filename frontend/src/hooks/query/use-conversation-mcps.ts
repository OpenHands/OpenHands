import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useConversationId } from "../use-conversation-id";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useSettings } from "./use-settings";

export const useConversationMcps = () => {
  const { conversationId } = useConversationId();
  const { curAgentState } = useAgentState();
  const { data: settings } = useSettings();

  return useQuery({
    queryKey: ["conversation", conversationId, "mcps"],
    queryFn: async () => {
      if (!conversationId) {
        throw new Error("No conversation ID provided");
      }
      const data = await V1ConversationService.getMcps(conversationId);
      return data.mcp_servers;
    },
    enabled:
      !!conversationId &&
      !!settings?.v1_enabled &&
      curAgentState !== AgentState.LOADING &&
      curAgentState !== AgentState.INIT,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
  });
};
