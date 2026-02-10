import React from "react";
import { usePostHog } from "posthog-js/react";
import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { convertImageToBase64 } from "#/utils/convert-image-to-base-64";
import { TrajectoryActions } from "../trajectory/trajectory-actions";
import { createChatMessage } from "#/services/chat-service";
import { InteractiveChatBox } from "./interactive-chat-box";
import { AgentState } from "#/types/agent-state";
import { useFilteredEvents } from "#/hooks/use-filtered-events";
import { FeedbackModal } from "../feedback/feedback-modal";
import { useScrollToBottom } from "#/hooks/use-scroll-to-bottom";
import { TypingIndicator } from "./typing-indicator";
import { useWsClient } from "#/context/ws-client-provider";
import { Messages as V0Messages } from "./messages";
import { ChatSuggestions } from "./chat-suggestions";
import { ScrollProvider } from "#/context/scroll-context";
import { useInitialQueryStore } from "#/stores/initial-query-store";
import { useSendMessage } from "#/hooks/use-send-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { useHandleBuildPlanClick } from "#/hooks/use-handle-build-plan-click";
import { USE_PLANNING_AGENT } from "#/utils/feature-flags";

import { ScrollToBottomButton } from "#/components/shared/buttons/scroll-to-bottom-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ChatMessagesSkeleton } from "./chat-messages-skeleton";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { ErrorMessageBanner } from "./error-message-banner";
import { Messages as V1Messages } from "#/components/v1/chat";
import { useUnifiedUploadFiles } from "#/hooks/mutation/use-unified-upload-files";
import { useConfig } from "#/hooks/query/use-config";
import { validateFiles } from "#/utils/file-validation";
import { useConversationStore } from "#/stores/conversation-store";
import ConfirmationModeEnabled from "./confirmation-mode-enabled";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import ChatStatusIndicator from "./chat-status-indicator";
import { getStatusColor, getStatusText } from "#/utils/utils";

function getEntryPoint(
  hasRepository: boolean | null,
  hasReplayJson: boolean | null,
): string {
  if (hasRepository) return "github";
  if (hasReplayJson) return "replay";
  return "direct";
}

export function ChatInterface() {
  const posthog = usePostHog();
  const { setMessageToSend } = useConversationStore();
  const { data: conversation } = useActiveConversation();
  const { errorMessage, removeErrorMessage } = useErrorMessageStore();
  const { isLoadingMessages } = useWsClient();
  const { isTask, taskStatus, taskDetail } = useTaskPolling();
  const conversationWebSocket = useConversationWebSocket();
  const { send } = useSendMessage();
  const {
    v0Events,
    v1UiEvents,
    v1FullEvents,
    totalEvents,
    hasSubstantiveAgentActions,
    v0UserEventsExist,
    v1UserEventsExist,
    userEventsExist,
  } = useFilteredEvents();
  const { setOptimisticUserMessage, getOptimisticUserMessage } =
    useOptimisticUserMessageStore();
  const { t } = useTranslation();
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const {
    scrollDomToBottom,
    onChatBodyScroll,
    hitBottom,
    autoScroll,
    setAutoScroll,
    setHitBottom,
  } = useScrollToBottom(scrollRef);
  const { data: config } = useConfig();

  const { curAgentState } = useAgentState();
  const { handleBuildPlanClick } = useHandleBuildPlanClick();
  const shouldUsePlanningAgent = USE_PLANNING_AGENT();

  // Disable Build button while agent is running (streaming)
  const isAgentRunning =
    curAgentState === AgentState.RUNNING ||
    curAgentState === AgentState.LOADING;

  // Global keyboard shortcut for Build button (Cmd+Enter / Ctrl+Enter)
  // This is placed here instead of PlanPreview to avoid duplicate listeners
  // when multiple PlanPreview components exist in the chat
  React.useEffect(() => {
    if (!shouldUsePlanningAgent || isAgentRunning) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      // Check for Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux)
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        handleBuildPlanClick(event);
        scrollDomToBottom();
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [
    shouldUsePlanningAgent,
    isAgentRunning,
    handleBuildPlanClick,
    scrollDomToBottom,
  ]);

  const [feedbackPolarity, setFeedbackPolarity] = React.useState<
    "positive" | "negative"
  >("positive");
  const [feedbackModalIsOpen, setFeedbackModalIsOpen] = React.useState(false);
  const { selectedRepository, replayJson } = useInitialQueryStore();
  const params = useParams();
  const { mutateAsync: uploadFiles } = useUnifiedUploadFiles();

  const optimisticUserMessage = getOptimisticUserMessage();

  const isV1Conversation = conversation?.conversation_version === "V1";

  // Show V1 messages immediately if events exist in store (e.g., remount),
  // or once loading completes. This replaces the old transition-observation
  // pattern (useState + useEffect watching loading→loaded) which always showed
  // skeleton on remount because local state initialized to false.
  const showV1Messages =
    v1FullEvents.length > 0 || !conversationWebSocket?.isLoadingHistory;

  const isReturningToConversation = !!params.conversationId;
  // Only show loading skeleton when genuinely loading AND no events in store yet.
  // If events exist (e.g., remount after data was already fetched), skip skeleton.
  const isHistoryLoading =
    (isLoadingMessages && !isV1Conversation && v0Events.length === 0) ||
    (isV1Conversation && !showV1Messages);
  const isChatLoading = isHistoryLoading && !isTask;

  const handleSendMessage = async (
    content: string,
    originalImages: File[],
    originalFiles: File[],
  ) => {
    // Create mutable copies of the arrays
    const images = [...originalImages];
    const files = [...originalFiles];
    if (totalEvents === 0) {
      posthog.capture("initial_query_submitted", {
        entry_point: getEntryPoint(
          selectedRepository !== null,
          replayJson !== null,
        ),
        query_character_length: content.length,
        replay_json_size: replayJson?.length,
      });
    } else {
      posthog.capture("user_message_sent", {
        session_message_count: totalEvents,
        current_message_length: content.length,
      });
    }

    // Validate file sizes before any processing
    const allFiles = [...images, ...files];
    const validation = validateFiles(allFiles);

    if (!validation.isValid) {
      displayErrorToast(`Error: ${validation.errorMessage}`);
      return; // Stop processing if validation fails
    }

    const promises = images.map((image) => convertImageToBase64(image));
    const imageUrls = await Promise.all(promises);

    const timestamp = new Date().toISOString();

    const { skipped_files: skippedFiles, uploaded_files: uploadedFiles } =
      files.length > 0
        ? await uploadFiles({ conversationId: params.conversationId!, files })
        : { skipped_files: [], uploaded_files: [] };

    skippedFiles.forEach((f) => displayErrorToast(f.reason));

    const filePrompt = `${t("CHAT_INTERFACE$AUGMENTED_PROMPT_FILES_TITLE")}: ${uploadedFiles.join("\n\n")}`;
    const prompt =
      uploadedFiles.length > 0 ? `${content}\n\n${filePrompt}` : content;

    send(createChatMessage(prompt, imageUrls, uploadedFiles, timestamp));
    setOptimisticUserMessage(content);
    setMessageToSend("");
  };

  const onClickShareFeedbackActionButton = async (
    polarity: "positive" | "negative",
  ) => {
    setFeedbackModalIsOpen(true);
    setFeedbackPolarity(polarity);
  };

  // Create a ScrollProvider with the scroll hook values
  const scrollProviderValue = {
    scrollRef,
    autoScroll,
    setAutoScroll,
    scrollDomToBottom,
    hitBottom,
    setHitBottom,
    onChatBodyScroll,
  };

  // Get server status indicator props
  const isStartingStatus =
    curAgentState === AgentState.LOADING || curAgentState === AgentState.INIT;
  const isStopStatus = curAgentState === AgentState.STOPPED;
  const isPausing = curAgentState === AgentState.PAUSED;
  const serverStatusColor = getStatusColor({
    isPausing,
    isTask,
    taskStatus,
    isStartingStatus,
    isStopStatus,
    curAgentState,
  });
  const serverStatusText = getStatusText({
    isPausing,
    isTask,
    taskStatus,
    taskDetail,
    isStartingStatus,
    isStopStatus,
    curAgentState,
    errorMessage,
    t,
  });

  return (
    <ScrollProvider value={scrollProviderValue}>
      <div className="h-full flex flex-col justify-between pr-0 md:pr-4 relative">
        {!hasSubstantiveAgentActions &&
          !optimisticUserMessage &&
          !userEventsExist &&
          !isChatLoading && (
            <ChatSuggestions
              onSuggestionsClick={(message) => setMessageToSend(message)}
            />
          )}
        {/* Note: We only hide chat suggestions when there's a user message */}

        <div
          ref={scrollRef}
          onScroll={(e) => onChatBodyScroll(e.currentTarget)}
          className="custom-scrollbar-always flex flex-col grow overflow-y-auto overflow-x-hidden px-4 pt-4 gap-2"
        >
          {isChatLoading && isReturningToConversation && (
            <ChatMessagesSkeleton />
          )}

          {isChatLoading && !isReturningToConversation && (
            <div className="flex justify-center" data-testid="loading-spinner">
              <LoadingSpinner size="small" />
            </div>
          )}

          {(!isLoadingMessages || v0Events.length > 0) && v0UserEventsExist && (
            <V0Messages
              messages={v0Events}
              isAwaitingUserConfirmation={
                curAgentState === AgentState.AWAITING_USER_CONFIRMATION
              }
            />
          )}

          {showV1Messages && v1UserEventsExist && (
            <V1Messages messages={v1UiEvents} allEvents={v1FullEvents} />
          )}
        </div>

        <div className="flex flex-col gap-[6px]">
          <div className="flex justify-between relative">
            <div className="flex items-end gap-1">
              <ConfirmationModeEnabled />
              {isStartingStatus && (
                <ChatStatusIndicator
                  statusColor={serverStatusColor}
                  status={serverStatusText}
                />
              )}
              {totalEvents > 0 && !isV1Conversation && (
                <TrajectoryActions
                  onPositiveFeedback={() =>
                    onClickShareFeedbackActionButton("positive")
                  }
                  onNegativeFeedback={() =>
                    onClickShareFeedbackActionButton("negative")
                  }
                  isSaasMode={config?.app_mode === "saas"}
                />
              )}
            </div>

            <div className="absolute left-1/2 transform -translate-x-1/2 bottom-0">
              {curAgentState === AgentState.RUNNING && <TypingIndicator />}
            </div>

            {!hitBottom && <ScrollToBottomButton onClick={scrollDomToBottom} />}
          </div>

          {errorMessage && (
            <ErrorMessageBanner
              message={errorMessage}
              onDismiss={removeErrorMessage}
            />
          )}

          <InteractiveChatBox onSubmit={handleSendMessage} />
        </div>

        {config?.app_mode !== "saas" && !isV1Conversation && (
          <FeedbackModal
            isOpen={feedbackModalIsOpen}
            onClose={() => setFeedbackModalIsOpen(false)}
            polarity={feedbackPolarity}
          />
        )}
      </div>
    </ScrollProvider>
  );
}
