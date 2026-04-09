import React from "react";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import { useSandboxRecovery } from "#/hooks/use-sandbox-recovery";
import { isTaskConversationId } from "#/utils/conversation-local-storage";

interface WebSocketProviderWrapperProps {
  children: React.ReactNode;
  conversationId: string;
}

/**
 * A wrapper component that conditionally renders either the old v0 WebSocket provider
 * or the new v1 WebSocket provider based on the version prop.
 *
 * @param conversationId - The conversation ID to pass to the provider
 * @param children - The child components to wrap
 */
export function WebSocketProviderWrapper({
  children,
  conversationId,
}: WebSocketProviderWrapperProps) {
  // Get conversation data for V1 provider
  const {
    data: conversation,
    refetch: refetchConversation,
    isFetched,
  } = useActiveConversation();
  // Get sub-conversation data for V1 provider
  const { data: subConversations } = useSubConversations(
    conversation?.sub_conversation_ids ?? [],
  );

  // Filter out null sub-conversations
  const filteredSubConversations = subConversations?.filter(
    (subConversation) => subConversation !== null,
  );

  const isConversationReady =
    !isTaskConversationId(conversationId) && isFetched && !!conversation;
  // Recovery for V1 conversations - handles page refresh and tab focus
  // Does NOT resume on WebSocket disconnect (server pauses after 20 min inactivity)
  useSandboxRecovery({
    conversationId,
    sandboxStatus: conversation?.sandbox_status,
    refetchConversation: isConversationReady ? refetchConversation : undefined,
  });

  return (
    <ConversationWebSocketProvider
      conversationId={conversationId}
      conversationUrl={conversation?.conversation_url}
      sessionApiKey={conversation?.session_api_key}
      subConversationIds={conversation?.sub_conversation_ids}
      subConversations={filteredSubConversations}
    >
      {children}
    </ConversationWebSocketProvider>
  );
}
