import {
  useConversationWebSocket,
  V1_WebSocketConnectionState,
} from "#/contexts/conversation-websocket-context";

/**
 * Unified hook that returns the current WebSocket status
 * - For V0 conversations: Returns status from useWsClient
 * - For V1 conversations: Returns status from ConversationWebSocketProvider
 */
export function useUnifiedWebSocketStatus(): V1_WebSocketConnectionState {
  const v1Context = useConversationWebSocket();
  return v1Context ? v1Context.connectionState : "CLOSED";
}
