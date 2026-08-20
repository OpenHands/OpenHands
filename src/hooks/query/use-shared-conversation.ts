import { useQuery } from "@tanstack/react-query";
import { SharedClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";

export const useSharedConversation = (conversationId?: string) =>
  useQuery({
    queryKey: ["shared-conversation", conversationId],
    queryFn: () => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      return new SharedClient(
        getAgentServerClientOptions(),
      ).getSharedConversation(conversationId);
    },
    enabled: !!conversationId,
    retry: false, // Don't retry for shared conversations
  });
