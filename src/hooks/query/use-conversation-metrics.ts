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
  // Cloud backends have no runtime REST fetch path: the runtime lives at
  // `*.prod-runtime.all-hands.dev`, which is CORS-blocked from the browser,
  // and the SDK no longer exposes a `/api/cloud-proxy` hop to reach it.
  // `metrics-modal.tsx` already falls back to the WebSocket-fed
  // `useMetricsStore`, so the hook must stay idle on cloud to avoid the
  // 405 toast the user sees when opening the Display Cost modal.
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
