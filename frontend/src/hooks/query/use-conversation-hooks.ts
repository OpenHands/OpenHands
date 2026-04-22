import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useConversationId } from "../use-conversation-id";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import { useSettings } from "./use-settings";

export const useConversationHooks = () => {
  const { conversationId } = useConversationId();
  const { curAgentState } = useAgentState();
  const { data: settings } = useSettings();

  return useQuery({
    queryKey: ["conversation", conversationId, "hooks", settings?.v1_enabled],
    queryFn: async () => {
      if (!conversationId) {
        throw new Error("No conversation ID provided");
      }

      // Hooks are only available for V1 conversations
      if (!settings?.v1_enabled) {
        return [];
      }

      const data = await V1ConversationService.getHooks(conversationId);
      return data.hooks;
    },
    enabled:
      !!conversationId &&
      !!settings?.v1_enabled &&
      curAgentState !== AgentState.LOADING &&
      curAgentState !== AgentState.INIT,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
};
