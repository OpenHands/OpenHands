import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useConversationId } from "../use-conversation-id";
import { useActiveConversation } from "./use-active-conversation";

export const useConversationSkills = () => {
  const { conversationId } = useConversationId();
  const executionStatus = useActiveConversation().data?.execution_status;

  return useQuery({
    queryKey: ["conversation", conversationId, "skills"],
    queryFn: async () => {
      if (!conversationId) {
        throw new Error("No conversation ID provided");
      }

      // Check if V1 is enabled and use the appropriate API
      const data = await V1ConversationService.getSkills(conversationId);
      return data.skills;
    },
    enabled: !!executionStatus,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
};
