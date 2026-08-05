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
 * Returns the main (code-agent) connection's own status, unmerged with the
 * planning connection. Use this for actions that only ever address the main
 * conversation (e.g. sending `/code`) — the merged status from
 * `useUnifiedWebSocketStatus` can report non-OPEN purely because the
 * *planning* socket is momentarily reconnecting.
 */
export function useMainWebSocketStatus(): WebSocketConnectionState {
  const conversationContext = useConversationWebSocket();
  return conversationContext
    ? conversationContext.mainConnectionState
    : "CLOSED";
}
