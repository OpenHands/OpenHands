import { useQuery } from "@tanstack/react-query";
import { getActiveBackend } from "#/api/backend-registry/active-store";
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
  // The runtime REST fetch (`getRuntimeConversation`) on cloud backends
  // tunnels through `/api/cloud-proxy`, which was removed from the
  // agent-server. This affects both deployment shapes:
  //
  //   - Local Canvas + Cloud backend: the local agent-server no longer
  //     exposes the endpoint (404).
  //   - Cloud Canvas + Cloud backend: the cloud host never had it (405).
  //
  // `metrics-modal.tsx` already falls back to the WebSocket-fed
  // `useMetricsStore`, so the hook stays idle on cloud to avoid the
  // error toast the user sees when opening the Display Cost modal.
  const isCloud = getActiveBackend().backend.kind === "cloud";

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
    enabled: enabled && !!conversationId && !!conversationUrl && !isCloud,
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
