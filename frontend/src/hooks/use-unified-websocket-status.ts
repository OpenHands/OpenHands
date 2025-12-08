import { useMemo } from "react";
import { useWsClient, V0_WebSocketStatus } from "#/context/ws-client-provider";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import { useConversationId } from "#/hooks/use-conversation-id";

/**
 * Unified hook that returns the current WebSocket status
 * - For V0 conversations: Returns status from useWsClient
 * - For V1 conversations: Returns status from ConversationWebSocketProvider
 */
export function useUnifiedWebSocketStatus(): V0_WebSocketStatus {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const v0Status = useWsClient();
  const v1Context = useConversationWebSocket();

  // Check if this is a V1 conversation:
  const isV1Conversation =
    conversationId.startsWith("task-") ||
    conversation?.conversation_version === "V1";

  const webSocketStatus = useMemo(() => {
    if (isV1Conversation) {
      // Map V1 connection state to WebSocketStatus
      if (!v1Context) return "DISCONNECTED";

      switch (v1Context.connectionState) {
        case "OPEN":
          return "CONNECTED";
        case "CONNECTING":
          return "CONNECTING";
        case "CLOSED":
        case "CLOSING":
          return "DISCONNECTED";
        default:
          return "DISCONNECTED";
      }
    }
    return v0Status.webSocketStatus;
  }, [
    isV1Conversation,
    v1Context,
    v0Status.webSocketStatus,
    conversationId,
    conversation,
  ]);

  return webSocketStatus;
}
