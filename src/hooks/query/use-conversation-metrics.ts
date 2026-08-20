import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getCombinedMetrics } from "#/utils/conversation-metrics";
import type { MetricsSnapshot } from "#/api/conversation-service/agent-server-conversation-service.types";

export const useConversationMetrics = (
  conversationId: string | null | undefined,
  conversationUrl: string | null | undefined,
  sessionApiKey: string | null | undefined,
  enabled: boolean = true,
): {
  data: MetricsSnapshot | undefined;
  isLoading: boolean;
  error: unknown;
} => {
  const query = useQuery({
    queryKey: [
      "conversation-metrics",
      conversationId,
      conversationUrl,
      sessionApiKey,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("Conversation ID is required");
      const conversationInfo =
        await AgentServerConversationService.getRuntimeConversation(
          conversationId,
          conversationUrl,
          sessionApiKey,
        );
      return getCombinedMetrics(conversationInfo);
    },
    // conversation_url is only set for cloud conversations; local ones are
    // served by the ConversationClient fallback in getRuntimeConversation.
    // Gating on it left local conversations with no REST snapshot at all
    // (zeros after a page reload until live WS metrics arrived).
    enabled: enabled && !!conversationId,
    staleTime: 1000 * 30,
    gcTime: 1000 * 60 * 5,
    refetchInterval: 1000 * 30,
    retry: false,
  });

  return {
    data: query.data,
    isLoading: query.isLoading,
    error: query.error,
  };
};
