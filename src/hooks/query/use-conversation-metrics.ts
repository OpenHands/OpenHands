import { useQuery } from "@tanstack/react-query";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { getAgentServerBaseUrl } from "#/api/agent-server-config";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { getCombinedMetrics } from "#/utils/conversation-metrics";
import type { MetricsSnapshot } from "#/api/conversation-service/agent-server-conversation-service.types";

/**
 * When the Canvas is itself hosted on the Cloud (e.g. served from
 * `app.all-hands.dev`), there is no local agent-server and therefore no
 * `/api/cloud-proxy` endpoint to tunnel runtime requests through. The cloud
 * host returns 405 for that path, so the runtime REST query must be skipped
 * and the modal falls back to the WebSocket-fed `useMetricsStore`.
 *
 * In the "local Canvas + Cloud backend" case the local agent-server DOES
 * expose `/api/cloud-proxy`, so the query is kept enabled and the cloud-proxy
 * path inside `getRuntimeConversation` handles the fetch.
 */
function isCloudHostedCanvas(): boolean {
  const { backend } = getActiveBackend();
  if (backend.kind !== "cloud") return false;

  const baseUrl = getAgentServerBaseUrl();
  if (!baseUrl || !backend.host) return false;

  try {
    const canvasOrigin = new URL(baseUrl).origin;
    const backendOrigin = new URL(backend.host).origin;
    return canvasOrigin === backendOrigin;
  } catch {
    return false;
  }
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
  const skipRuntimeQuery = isCloudHostedCanvas();

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
    enabled:
      enabled && !!conversationId && !!conversationUrl && !skipRuntimeQuery,
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
