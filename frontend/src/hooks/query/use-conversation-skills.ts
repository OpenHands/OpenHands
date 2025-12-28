import { useQuery } from "@tanstack/react-query";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useConversationId } from "../use-conversation-id";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useSettings } from "./use-settings";

export const useConversationSkills = () => {
  const { conversationId } = useConversationId();
  const { curAgentState } = useAgentState();
  const { data: settings } = useSettings();

  return useQuery({
    queryKey: ["conversation", conversationId, "skills", settings?.v1_enabled],
    queryFn: async () => {
      if (!conversationId) {
        throw new Error("No conversation ID provided");
      }

      // Check if V1 is enabled and use the appropriate API
      if (settings?.v1_enabled) {
        const data = await V1ConversationService.getSkills(conversationId);
        return data.skills;
      }

      const data = await ConversationService.getMicroagents(conversationId);
      return data.microagents;
    },
    enabled:
      !!conversationId &&
      curAgentState !== AgentState.LOADING &&
      curAgentState !== AgentState.INIT,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
};
