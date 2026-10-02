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

/**
 * The main connection's own status, unmerged with the planning connection.
 * Use for actions that only address the main conversation (e.g. `/code`) —
 * `useUnifiedWebSocketStatus` can report non-OPEN purely from a momentary
 * planning reconnect.
 */
export function useMainWebSocketStatus(): WebSocketConnectionState {
  const conversationContext = useConversationWebSocket();
  return conversationContext
    ? conversationContext.mainConnectionState
    : "CLOSED";
}
