import {
  useConversationWebSocket,
  WebSocketConnectionState,
} from "#/contexts/conversation-websocket-context";

/**
 * Returns the current conversation WebSocket status.
 */
export function useUnifiedWebSocketStatus(): WebSocketConnectionState {
  const conversationContext = useConversationWebSocket();
  return conversationContext ? conversationContext.connectionState : "CLOSED";
}
