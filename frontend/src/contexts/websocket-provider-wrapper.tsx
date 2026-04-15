import React from "react";
import { ConversationWebSocketProvider } from "#/contexts/conversation-websocket-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import { useSandboxRecovery } from "#/hooks/use-sandbox-recovery";
import { useV1ConversationStateStore } from "#/stores/v1-conversation-state-store";
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

  // Prime the execution status from the backend-reported value on the
  // AppConversation. The V1 store is otherwise only populated via WebSocket
  // ConversationStateUpdateEvents, which are not persisted in event history.
  // Without this, after resuming a closed conversation the UI has no source of
  // truth for agent status until a live state-update event arrives — and the
  // first resume may never emit one if the agent is idle, leaving the UI stuck.
  const backendExecutionStatus = conversation?.execution_status ?? null;
  React.useEffect(() => {
    if (backendExecutionStatus) {
      useV1ConversationStateStore
        .getState()
        .setExecutionStatus(backendExecutionStatus);
    }
  }, [backendExecutionStatus]);

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
