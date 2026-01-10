import React from "react";
import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";

import { useConversationId } from "#/hooks/use-conversation-id";
import { useCommandStore } from "#/stores/command-store";
import { useConversationStore } from "#/stores/conversation-store";
import { useAgentStore } from "#/stores/agent-store";
import { AgentState } from "#/types/agent-state";

import { useBatchFeedback } from "#/hooks/query/use-batch-feedback";
import { EventHandler } from "../wrapper/event-handler";
import { useConversationConfig } from "#/hooks/query/use-conversation-config";

import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useTaskPolling } from "#/hooks/query/use-task-polling";

import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { ConversationSubscriptionsProvider } from "#/context/conversation-subscriptions-provider";
import { useUserProviders } from "#/hooks/use-user-providers";

import { ConversationMain } from "#/components/features/conversation/conversation-main/conversation-main";
import { ConversationNameWithStatus } from "#/components/features/conversation/conversation-name-with-status";

import { ConversationTabs } from "#/components/features/conversation/conversation-tabs/conversation-tabs";
import { WebSocketProviderWrapper } from "#/contexts/websocket-provider-wrapper";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useUnifiedResumeConversationSandbox } from "#/hooks/mutation/use-unified-start-conversation";
import { I18nKey } from "#/i18n/declaration";
import { useEventStore } from "#/stores/use-event-store";

function AppContent() {
  useConversationConfig();
  const { t } = useTranslation();
  const { conversationId } = useConversationId();
  const clearEvents = useEventStore((state) => state.clearEvents);

  // Handle both task IDs (task-{uuid}) and regular conversation IDs
  const { isTask, taskStatus, taskDetail } = useTaskPolling();

  const { data: conversation, isFetched, refetch } = useActiveConversation();
  const { mutate: startConversation, isPending: isStarting } =
    useUnifiedResumeConversationSandbox();
  const { data: isAuthed } = useIsAuthed();
  const { providers } = useUserProviders();
  const { resetConversationState } = useConversationStore();
  const navigate = useNavigate();
  const clearTerminal = useCommandStore((state) => state.clearTerminal);
  const setCurrentAgentState = useAgentStore(
    (state) => state.setCurrentAgentState,
  );
  const removeErrorMessage = useErrorMessageStore(
    (state) => state.removeErrorMessage,
  );

  // Track which conversation ID we've auto-started to prevent auto-restart after manual stop
  const processedConversationId = React.useRef<string | null>(null);

  // Fetch batch feedback data when conversation is loaded
  useBatchFeedback();

  // 1. Cleanup Effect - runs when navigating to a different conversation
  React.useEffect(() => {
    clearTerminal();
    resetConversationState();
    setCurrentAgentState(AgentState.LOADING);
    removeErrorMessage();
    clearEvents();

    // Reset tracking ONLY if we're navigating to a DIFFERENT conversation
    // Don't reset on StrictMode remounts (conversationId is the same)
    if (processedConversationId.current !== conversationId) {
      processedConversationId.current = null;
    }
  }, [
    conversationId,
    clearTerminal,
    resetConversationState,
    setCurrentAgentState,
    removeErrorMessage,
    clearEvents,
  ]);

  // 2. Task Error Display Effect
  React.useEffect(() => {
    if (isTask && taskStatus === "ERROR") {
      displayErrorToast(
        taskDetail || t(I18nKey.CONVERSATION$FAILED_TO_START_FROM_TASK),
      );
    }
  }, [isTask, taskStatus, taskDetail, t]);

  // 3. Auto-start Effect - handles conversation not found and auto-starting STOPPED conversations
  React.useEffect(() => {
    // Wait for data to be fetched
    if (!isFetched || !isAuthed) return;

    // Handle conversation not found
    if (!conversation) {
      displayErrorToast(t(I18nKey.CONVERSATION$NOT_EXIST_OR_NO_PERMISSION));
      navigate("/");
      return;
    }

    const currentConversationId = conversation.conversation_id;
    const currentStatus = conversation.status;

    // Skip if we've already processed this conversation
    if (processedConversationId.current === currentConversationId) {
      return;
    }

    // Mark as processed immediately to prevent duplicate calls
    processedConversationId.current = currentConversationId;

    // Auto-start STOPPED conversations on initial load only
    if (currentStatus === "STOPPED" && !isStarting) {
      startConversation(
        { conversationId: currentConversationId, providers },
        {
          onError: (error) => {
            displayErrorToast(
              t(I18nKey.CONVERSATION$FAILED_TO_START_WITH_ERROR, {
                error: error.message,
              }),
            );
            refetch();
          },
        },
      );
    }
    // NOTE: conversation?.status is intentionally NOT in dependencies
    // We only want to run when conversation ID changes, not when status changes
    // This prevents duplicate calls when stale cache data is replaced with fresh data
  }, [
    conversation?.conversation_id,
    isFetched,
    isAuthed,
    isStarting,
    providers,
    startConversation,
    navigate,
    refetch,
    t,
  ]);

  const isV0Conversation = conversation?.conversation_version === "V0";

  const content = (
    <ConversationSubscriptionsProvider>
      <EventHandler>
        <div
          data-testid="app-route"
          className="p-3 md:p-0 flex flex-col h-full gap-3"
        >
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4.5 pt-2 lg:pt-0">
            <ConversationNameWithStatus />
            <ConversationTabs />
          </div>

          <ConversationMain />
        </div>
      </EventHandler>
    </ConversationSubscriptionsProvider>
  );

  // Render WebSocket provider immediately to avoid mount/remount cycles
  // The providers internally handle waiting for conversation data to be ready
  return (
    <WebSocketProviderWrapper
      version={isV0Conversation ? 0 : 1}
      conversationId={conversationId}
    >
      {content}
    </WebSocketProviderWrapper>
  );
}

function App() {
  return <AppContent />;
}

export default App;
