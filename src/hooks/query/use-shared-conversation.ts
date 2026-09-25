import { useQuery } from "@tanstack/react-query";
import { SharedClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { getCloudSharedConversation } from "#/api/cloud/shared-conversation-service.api";

interface UseSharedConversationOptions {
  enabled?: boolean;
}

export const useSharedConversation = (
  conversationId?: string,
  options: UseSharedConversationOptions = {},
) =>
  useQuery({
    queryKey: ["shared-conversation", conversationId],
    queryFn: () => {
      if (!conversationId) {
        throw new Error("Conversation ID is required");
      }
      if (getActiveBackend().backend.kind === "cloud") {
        return getCloudSharedConversation(conversationId);
      }
      return new SharedClient(
        getAgentServerClientOptions(),
      ).getSharedConversation(conversationId);
    },
    enabled: !!conversationId && (options.enabled ?? true),
    retry: false, // Don't retry for shared conversations
    // The shared page renders its own not-found state, and the conversation
    // route uses this query as a silent probe before reporting a miss.
    meta: { disableToast: true },
  });
