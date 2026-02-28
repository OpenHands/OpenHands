import { useQuery } from "@tanstack/react-query";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { getCombinedMetrics } from "#/utils/conversation-metrics";
import type { V1MetricsSnapshot } from "#/api/conversation-service/v1-conversation-service.types";

/**
 * Hook to fetch metrics directly from the sandbox for V1 conversations
 * @param conversationId The conversation ID
 * @param conversationUrl The conversation URL from the active conversation
 * @param sessionApiKey The session API key from the active conversation
 * @param enabled Whether the query should be enabled (typically when modal is open and conversation is V1)
 */
export const useSandboxMetrics = (
  conversationId: string | null | undefined,
  conversationUrl: string | null | undefined,
  sessionApiKey: string | null | undefined,
  enabled: boolean = true,
): {
  data: V1MetricsSnapshot | undefined;
  isLoading: boolean;
  error: unknown;
} => {
  const query = useQuery({
    queryKey: [
      "sandbox-metrics",
      conversationId,
      conversationUrl,
      sessionApiKey,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("Conversation ID is required");
      const conversationInfo =
        await V1ConversationService.getRuntimeConversation(
          conversationId,
          conversationUrl,
          sessionApiKey,
        );
      return getCombinedMetrics(conversationInfo);
    },
    enabled:
      enabled && !!conversationId && !!conversationUrl && !!sessionApiKey,
    staleTime: 1000 * 30, // 30 seconds
    gcTime: 1000 * 60 * 5, // 5 minutes
    refetchInterval: 1000 * 30, // Refetch every 30 seconds
    retry: false, // Don't retry on failure since this is a new endpoint
  });

  return {
    data: query.data,
    isLoading: query.isLoading,
    error: query.error,
  };
};
