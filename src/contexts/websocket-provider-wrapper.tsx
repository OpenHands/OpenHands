import React from "react";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";

interface WebSocketProviderWrapperProps {
  children: React.ReactNode;
  conversationId: string;
  /**
   * When false, this tree is a secondary conversation (e.g. a popout):
   * it streams into its own event-store bucket and does not claim the process-
   * wide current-conversation singleton or shared chrome stores.
   */
  sharedSideEffects?: boolean;
}

export function WebSocketProviderWrapper({
  children,
  conversationId,
  sharedSideEffects = true,
}: WebSocketProviderWrapperProps) {
  // claimCurrentConversation follows ConversationRenderScope (false inside
  // popouts), so we don't need to pass it explicitly here.
  const { data: conversation } = useActiveConversation();
  // Popouts are chat-only — skip the planning-agent sub-conversation.
  const { data: subConversations } = useSubConversations(
    sharedSideEffects ? (conversation?.sub_conversation_ids ?? []) : [],
  );

  const filteredSubConversations = subConversations?.filter(
    (subConversation) => subConversation !== null,
  );

  // Don't pass a conversation URL to the WebSocket provider while the cloud
  // sandbox is PAUSED. The URL still points to the old sandbox host, which
  // rejects connections until the sandbox has fully resumed. Treating the URL
  // as absent here keeps wsUrl === null in ConversationWebSocketProvider, so
  // no connection is attempted until useActiveConversation detects the
  // transition out of PAUSED (via fast 3-second polling).
  const conversationUrl =
    conversation?.sandbox_status === "PAUSED"
      ? null
      : conversation?.conversation_url;

  return (
    <ConversationWebSocketProvider
      conversationId={conversationId}
      conversationUrl={conversationUrl}
      sessionApiKey={conversation?.session_api_key}
      subConversationIds={
        sharedSideEffects ? conversation?.sub_conversation_ids : undefined
      }
      subConversations={
        sharedSideEffects ? filteredSubConversations : undefined
      }
      sharedSideEffects={sharedSideEffects}
    >
      {children}
    </ConversationWebSocketProvider>
  );
}
