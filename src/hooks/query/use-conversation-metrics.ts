import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { combineUsageMetrics } from "#/utils/conversation-metrics";
import type {
  MetricsSnapshot,
  RuntimeConversationStats,
} from "#/api/conversation-service/agent-server-conversation-service.types";

const selectStats = (stats: RuntimeConversationStats) => stats;

/**
 * One cached REST read of the conversation's runtime stats, shared by every
 * consumer: each selects its own view so they never poll twice.
 */
function useConversationStatsQuery<TData>(
  conversationId: string | null | undefined,
  conversationUrl: string | null | undefined,
  sessionApiKey: string | null | undefined,
  enabled: boolean,
  select: (stats: RuntimeConversationStats) => TData,
) {
  const { backend } = useActiveBackend();
  const runtimeEndpointReady =
    backend.kind !== "cloud" || Boolean(conversationUrl);

  return useQuery({
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
      return conversationInfo.stats;
    },
    select,
    // conversation_url is only set for cloud conversations; local ones are
    // served by the ConversationClient fallback in getRuntimeConversation.
    // Gating local on it left local conversations with no REST snapshot at all
    // (zeros after a page reload until live WS metrics arrived). Cloud must
    // wait for the per-conversation runtime URL; otherwise the typed client has
    // no local backend to fall back to and throws "No backend is configured".
    enabled: enabled && !!conversationId && runtimeEndpointReady,
    staleTime: 1000 * 30,
    gcTime: 1000 * 60 * 5,
    refetchInterval: 1000 * 30,
    retry: false,
  });
}

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
  const query = useConversationStatsQuery(
    conversationId,
    conversationUrl,
    sessionApiKey,
    enabled,
    combineUsageMetrics,
  );

  return {
    data: query.data,
    isLoading: query.isLoading,
    error: query.error,
  };
};

/**
 * The raw runtime stats, including the per-call `token_usages` that the
 * combined {@link useConversationMetrics} snapshot drops.
 */
export const useConversationStats = (
  conversationId: string | null | undefined,
  conversationUrl: string | null | undefined,
  sessionApiKey: string | null | undefined,
  enabled: boolean = true,
) =>
  useConversationStatsQuery(
    conversationId,
    conversationUrl,
    sessionApiKey,
    enabled,
    selectStats,
  );
