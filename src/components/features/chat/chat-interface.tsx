import React from "react";
import { useNavigate } from "react-router";
import { useTracking } from "#/hooks/use-tracking";
import { useTranslation } from "react-i18next";
import { isAcpAuthErrorCode } from "#/utils/acp-error-codes";
import { convertImageToBase64 } from "#/utils/convert-image-to-base-64";
import { createChatMessage } from "#/services/chat-service";
import { BtwMessages } from "./btw-messages";
import { GoalStatusBanner } from "./goal-status-banner";
import { ModelMessages } from "./model-messages";
import { useModelStore } from "#/stores/model-store";
import { useGoalStore } from "#/stores/goal-store";
import { InteractiveChatBox } from "./interactive-chat-box";
import { AgentState } from "#/types/agent-state";
import { useFilteredEvents } from "#/hooks/use-filtered-events";
import { useScrollToBottom } from "#/hooks/use-scroll-to-bottom";
import { useLoadOlderEvents } from "#/hooks/use-load-older-events";
import { TypingIndicator } from "./typing-indicator";
import { ChatSuggestions } from "./chat-suggestions";
import { ScrollProvider } from "#/context/scroll-context";
import { useInitialQueryStore } from "#/stores/initial-query-store";
import { useSendMessage } from "#/hooks/use-send-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { useIsArchivedConversation } from "#/hooks/use-is-archived-conversation";
import { useHandleBuildPlanClick } from "#/hooks/use-handle-build-plan-click";

import { ScrollToBottomButton } from "#/components/shared/buttons/scroll-to-bottom-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ChatMessagesSkeleton } from "./chat-messages-skeleton";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { SERVER_CONNECTION_ERROR_MESSAGE } from "#/constants/server-connection-error";
import { ErrorMessageBanner } from "./error-message-banner";
import { SkillInstallRestartBanner } from "./skill-install-restart-banner";
import { LlmNotConfiguredBanner } from "#/components/features/home/llm-not-configured-banner";
import { useLlmConfigured } from "#/hooks/use-llm-configured";
import { Messages } from "#/components/conversation-events/chat/messages";
import { PendingUserMessages } from "./pending-user-messages";
import { useUnifiedUploadFiles } from "#/hooks/mutation/use-unified-upload-files";
import { validateFiles } from "#/utils/file-validation";
import { useConversationStore } from "#/stores/conversation-store";
import ConfirmationModeEnabled from "./confirmation-mode-enabled";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import { matchesPendingConversationId } from "#/utils/pending-task-message-link";
import { useConversationWebSocket } from "#/contexts/conversation-websocket-context";
import ChatStatusIndicator from "./chat-status-indicator";
import { getStatusColor, getStatusText } from "#/utils/utils";
import { useNewConversationCommand } from "#/hooks/mutation/use-new-conversation-command";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { I18nKey } from "#/i18n/declaration";
import { hasConversationStarted } from "./components/resolve-picker-kind";

function getEntryPoint(
  hasRepository: boolean | null,
  hasReplayJson: boolean | null,
): string {
  if (hasRepository) return "github";
  if (hasReplayJson) return "replay";
  return "direct";
}

export function ChatInterface() {
  const { trackInitialQuerySubmitted, trackUserMessageSent } = useTracking();
  const { setMessageToSend } = useConversationStore();
  const { errorMessage, errorCode, removeErrorMessage, setErrorMessage } =
    useErrorMessageStore();
  const navigate = useNavigate();
  const { isTask, taskStatus, taskDetail } = useTaskPolling();
  // Hide empty-state chrome for the entire `/conversations/task-{uuid}` route,
  // including the brief READY window before redirect completes.
  const isProvisioningTask = isTask;
  const conversationWebSocket = useConversationWebSocket();
  const { send } = useSendMessage();
  const {
    renderableEvents,
    allConversationEvents,
    totalEvents,
    hasSubstantiveAgentActions,
    userEventsExist,
  } = useFilteredEvents();
  const enqueuePendingMessage = useOptimisticUserMessageStore(
    (state) => state.enqueuePendingMessage,
  );
  const markPendingMessageError = useOptimisticUserMessageStore(
    (state) => state.markPendingMessageError,
  );
  const pendingMessages = useOptimisticUserMessageStore(
    (state) => state.pendingMessages,
  );
  const { t } = useTranslation("openhands");
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const {
    scrollDomToBottom,
    onChatBodyScroll,
    hitBottom,
    autoScroll,
    setAutoScroll,
    setHitBottom,
  } = useScrollToBottom(scrollRef);
  const {
    mutate: newConversationCommand,
    isPending: isNewConversationPending,
  } = useNewConversationCommand();

  const { curAgentState } = useAgentState();
  const { handleBuildPlanClick } = useHandleBuildPlanClick();

  // Cloud conversations whose sandbox is MISSING or ERROR are read-only:
  // the sandbox is gone and cannot be resumed, so we hide the chat input
  // and show an explanatory banner. For local backends sandbox_status is
  // always null, so this is effectively a no-op for non-cloud use.
  const { data: activeConversation } = useActiveConversation();
  const sandboxStatus = activeConversation?.sandbox_status ?? null;
  const isArchivedConversation = useIsArchivedConversation();

  // Block sending in a resumed conversation that has no usable LLM, and show
  // the same setup banner as the home screen so the dead end is explained.
  const { isConfigured: isLlmConfigured, isLoading: isLlmConfigLoading } =
    useLlmConfigured();
  const llmBlocked = !isLlmConfigLoading && !isLlmConfigured;

  // Disable Build button while agent is running (streaming)
  const isAgentRunning =
    curAgentState === AgentState.RUNNING ||
    curAgentState === AgentState.LOADING;

  // Global keyboard shortcut for Build button (Cmd+Enter / Ctrl+Enter)
  // This is placed here instead of PlanPreview to avoid duplicate listeners
  // when multiple PlanPreview components exist in the chat
  React.useEffect(() => {
    if (isAgentRunning) {
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
  }, [isAgentRunning, handleBuildPlanClick, scrollDomToBottom]);

  const { selectedRepository, replayJson } = useInitialQueryStore();
  const { conversationId } = useOptionalConversationId();

  // The live goal banner renders in the scroll stream but advances via store
  // updates (in-progress goal events are filtered out of `renderableEvents`),
  // so the bottom-following effect would not react to it. This key changes as
  // an active loop appears and advances each round; feeding it into that effect
  // keeps the banner in view when the user is pinned to the bottom.
  const activeGoalScrollKey = useGoalStore((s) => {
    const goal = conversationId
      ? s.statusByConversation[conversationId]
      : undefined;
    return goal?.active ? `${goal.iteration}:${goal.status}` : null;
  });
  const { mutateAsync: uploadFiles } = useUnifiedUploadFiles();

  // Lazy "scroll up to load older events" backfill. Initial REST fetch only
  // returns the most recent page; this hook paginates older events into the
  // store on demand so the chat doesn't load (potentially) thousands of
  // events on first render.
  const {
    isLoading: isLoadingOlderEvents,
    hasMore: hasMoreOlderEvents,
    loadOlder,
  } = useLoadOlderEvents(conversationId);

  // Trigger `loadOlder` and preserve the visual scroll position once the
  // older page is merged in (otherwise prepending events would jump the
  // chat far down). We fire from three places to cover the cases the
  // browser's scroll event misses:
  //
  //   - `onScroll`: normal "user scrolled near the top" path.
  //   - `onWheel`:  user is already pinned at scrollTop=0 and tries to
  //                 wheel further up — no scroll event fires past 0.
  //   - effect:     content is shorter than the viewport (no scrollbar
  //                 at all), so the user has nothing to scroll. Re-runs
  //                 as more pages arrive until there's overflow or the
  //                 server runs out of older events.
  const SCROLL_TOP_THRESHOLD_PX = 80;
  const preserveScrollPosition = React.useRef<{
    scrollHeight: number;
    scrollTop: number;
  } | null>(null);
  const maybeLoadOlder = React.useCallback(
    (target: HTMLElement) => {
      if (isProvisioningTask || isLoadingOlderEvents || !hasMoreOlderEvents) {
        return;
      }

      const atTop = target.scrollTop <= SCROLL_TOP_THRESHOLD_PX;
      const noOverflow =
        target.scrollHeight <= target.clientHeight + SCROLL_TOP_THRESHOLD_PX;
      if (!atTop && !noOverflow) return;

      preserveScrollPosition.current = {
        scrollHeight: target.scrollHeight,
        scrollTop: target.scrollTop,
      };
      loadOlder().catch((error) => {
        preserveScrollPosition.current = null;
        const message =
          error instanceof Error && error.message
            ? error.message
            : t(I18nKey.ERROR$GENERIC);
        setErrorMessage(message);
      });
    },
    [
      hasMoreOlderEvents,
      isLoadingOlderEvents,
      isProvisioningTask,
      loadOlder,
      setErrorMessage,
      t,
    ],
  );

  const handleWheelForPagination = React.useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      // Browsers don't dispatch a scroll event when scrollTop is already
      // 0 and the user wheels upward, so onScroll alone misses this case.
      if (e.deltaY < 0 && e.currentTarget.scrollTop <= 0) {
        maybeLoadOlder(e.currentTarget);
      }
    },
    [maybeLoadOlder],
  );

  const hasPendingUserMessages = React.useMemo(
    () =>
      conversationId
        ? pendingMessages.some((message) =>
            matchesPendingConversationId(
              conversationId,
              message.conversationId,
            ),
          )
        : false,
    [pendingMessages, conversationId],
  );

  const hasModelEntries = useModelStore((s) =>
    conversationId
      ? (s.entriesByConversation[conversationId]?.length ?? 0) > 0
      : false,
  );
  const hasStartedConversation = hasConversationStarted({
    isLoadingHistory: conversationWebSocket?.isLoadingHistory === true,
    hasUserEvents: userEventsExist,
    hasPendingUserMessages,
    hasSubstantiveAgentActions,
    hasModelEntries,
  });

  // Show V1 messages immediately if events exist in store (e.g., remount),
  // if the user already has a locally-tracked pending bubble (home-page cloud
  // submit while history/WS catch up), or once loading completes. This
  // replaces the old transition-observation pattern (useState + useEffect
  // watching loading→loaded) which always showed skeleton on remount because
  // local state initialized to false.
  const showConversationMessages =
    allConversationEvents.length > 0 ||
    hasPendingUserMessages ||
    !conversationWebSocket?.isLoadingHistory;

  const isReturningToConversation = !!conversationId;
  // Only show loading skeleton when genuinely loading AND no events in store yet.
  // If events exist (e.g., remount after data was already fetched), skip skeleton.
  const isHistoryLoading = !showConversationMessages;
  const isChatLoading = isHistoryLoading && !isTask;

  const handleSendMessage = async (
    content: string,
    originalImages: File[],
    originalFiles: File[],
  ) => {
    // Handle /new command for V1 conversations
    if (content.trim() === "/new") {
      if (!conversationId) {
        displayErrorToast(t(I18nKey.CONVERSATION$CLEAR_NO_ID));
        return;
      }
      if (totalEvents === 0) {
        displayErrorToast(t(I18nKey.CONVERSATION$CLEAR_EMPTY));
        return;
      }
      if (isNewConversationPending) {
        return;
      }
      newConversationCommand();
      return;
    }

    // Create mutable copies of the arrays
    const images = [...originalImages];
    const files = [...originalFiles];
    if (totalEvents === 0) {
      trackInitialQuerySubmitted({
        entryPoint: getEntryPoint(
          selectedRepository !== null,
          replayJson !== null,
        ),
        queryCharacterLength: content.length,
        replayJsonSize: replayJson?.length,
      });
    } else {
      trackUserMessageSent({
        sessionMessageCount: totalEvents,
        currentMessageLength: content.length,
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
        ? await uploadFiles({ conversationId: conversationId!, files })
        : { skipped_files: [], uploaded_files: [] };

    skippedFiles.forEach((f) => displayErrorToast(f.reason));

    const filePrompt = `${t(I18nKey.CHAT_INTERFACE$AUGMENTED_PROMPT_FILES_TITLE)}: ${uploadedFiles.join("\n\n")}`;
    const prompt =
      uploadedFiles.length > 0 ? `${content}\n\n${filePrompt}` : content;

    // Enqueue the message into the local pending queue with status "sending"
    // so the user immediately sees it in the chat with a faded treatment. The
    // entry is removed when the WebSocket echoes back the corresponding
    // `UserMessageEvent`. If the API call to send the message fails, the entry
    // is flipped to "error" with a retry link.
    const pendingId = enqueuePendingMessage({
      conversationId: conversationId!,
      // `text` is what the user sees in the bubble; `content` is what we
      // actually hand to the server (the prompt may include an appended
      // "Files uploaded: …" block) and is what the echo will be matched
      // against. They're different when there are file attachments.
      text: content,
      content: prompt,
      imageUrls,
      fileUrls: uploadedFiles,
      timestamp,
    });
    // Submitting a new prompt should always pull the chat back to the
    // latest message even if the user had scrolled up. This also re-arms
    // autoScroll so the streamed agent reply auto-follows.
    scrollDomToBottom();
    setMessageToSend("");

    try {
      await send(
        createChatMessage(prompt, imageUrls, uploadedFiles, timestamp),
      );
    } catch (sendError) {
      const sendErrorMessage =
        sendError instanceof Error
          ? sendError.message
          : t(I18nKey.CHAT_INTERFACE$FAILED_TO_SEND_MESSAGE);
      markPendingMessageError(pendingId, sendErrorMessage);
    }
  };

  // Auto-scroll to bottom when new messages arrive — but only if the user is
  // already pinned to the bottom. Scrolling up to load older events also
  // grows `renderableEvents`, and we don't want to yank the user back to the
  // bottom in that case.
  React.useEffect(() => {
    // If a "load older" was just triggered, restore the scroll position so
    // the conversation appears to extend upward instead of jumping.
    if (preserveScrollPosition.current && scrollRef.current) {
      const { scrollHeight: prevHeight, scrollTop: prevTop } =
        preserveScrollPosition.current;
      const dom = scrollRef.current;
      const delta = dom.scrollHeight - prevHeight;
      if (delta > 0) {
        dom.scrollTop = prevTop + delta;
      }
      preserveScrollPosition.current = null;
      return;
    }

    if (autoScroll) {
      scrollDomToBottom();
    }
    // Note: We intentionally exclude autoScroll from deps because we only want
    // to scroll when message content changes, not when autoScroll state changes.
  }, [
    renderableEvents.length,
    hasPendingUserMessages,
    activeGoalScrollKey,
    scrollDomToBottom,
  ]);

  // Auto-load older events when the chat content doesn't overflow the
  // scroll area (no scrollbar to drag, no wheel events past 0). We
  // re-run only when the rendered list grows or `hasMore` flips, NOT
  // when `maybeLoadOlder` re-creates: the underlying hook's `loadOlder`
  // ref changes whenever its internal `isLoading` toggles, so depending
  // on `maybeLoadOlder` would re-fire the effect on every failed page
  // and tight-loop until the server recovered. Driving off
  // `renderableEvents.length` instead means a successful page (events
  // grow) chains the next request, while a failed page (events
  // unchanged) waits for the user to retry.
  const maybeLoadOlderRef = React.useRef(maybeLoadOlder);
  React.useEffect(() => {
    maybeLoadOlderRef.current = maybeLoadOlder;
  });
  React.useEffect(() => {
    const target = scrollRef.current;
    if (!target) return;
    maybeLoadOlderRef.current(target);
  }, [renderableEvents.length, hasMoreOlderEvents]);

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
      <div
        className="relative flex h-full flex-col justify-between px-4"
        data-testid="chat-interface"
      >
        {!hasSubstantiveAgentActions &&
          !hasPendingUserMessages &&
          !userEventsExist &&
          !hasModelEntries &&
          !isChatLoading &&
          !isProvisioningTask &&
          totalEvents === 0 &&
          !isArchivedConversation &&
          // With no usable LLM the suggestions can't be acted on (the input is
          // disabled). They're also a `pointer-events-auto` overlay that would
          // sit over the LlmNotConfiguredBanner below and swallow clicks on its
          // setup button — so hide them and let the banner be the lone CTA.
          !llmBlocked && (
            <ChatSuggestions
              onSuggestionsClick={(message) => setMessageToSend(message)}
            />
          )}
        {/* Note: We only hide chat suggestions when there's a user message */}

        <div
          ref={scrollRef}
          data-testid="chat-scroll-container"
          onScroll={(e) => {
            onChatBodyScroll(e.currentTarget);
            maybeLoadOlder(e.currentTarget);
          }}
          onWheel={handleWheelForPagination}
          className="custom-scrollbar-always flex min-h-0 grow flex-col gap-2 overflow-x-hidden overflow-y-auto px-0 pt-4 pb-8 md:px-4"
        >
          {isChatLoading && isReturningToConversation && (
            <ChatMessagesSkeleton />
          )}

          {isChatLoading && !isReturningToConversation && (
            <div className="flex justify-center" data-testid="loading-spinner">
              <LoadingSpinner size="small" />
            </div>
          )}

          {isLoadingOlderEvents && (
            <div
              className="flex items-center justify-center gap-2 py-3 text-sm text-neutral-400"
              data-testid="loading-older-events"
            >
              <LoadingSpinner size="small" />
              <span>{t(I18nKey.CHAT_INTERFACE$FETCHING_OLDER_MESSAGES)}</span>
            </div>
          )}

          {/*
           * Render whenever there's anything to display. Previously this
           * was gated on `conversationUserEventsExist`, but with the lazy
           * "50 most recent" REST fetch the initial window may not include
           * any `source: "user"` events (long agent runs between user
           * turns). That left the chat blank, leaving the user nothing to
           * scroll — which is why "scroll up to load older" appeared
           * broken. The empty-state ChatSuggestions block above still
           * keeps its own gate (`!userEventsExist && !hasSubstantiveAgentActions`)
           * so brand-new conversations show suggestions, not an empty chat.
           */}
          {/* /model entries created before any event is rendered are
              anchored to `null` and live above the message list. */}
          <ModelMessages conversationId={conversationId} anchorEventId={null} />

          {showConversationMessages && renderableEvents.length > 0 && (
            <Messages
              messages={renderableEvents}
              allEvents={allConversationEvents}
            />
          )}

          {/*
            Render the local pending-message queue independently so messages
            the user just submitted show up immediately (with a faded "sending"
            treatment) even before any real conversation event has come back
            from the server. Entries drain (FIFO) when the matching
            UserMessageEvent echoes back over the WebSocket, so this never
            double-renders alongside the real event list.
          */}
          <PendingUserMessages />

          {/* Goal-loop status sits at the end of the message flow — above the
              composer and its typing indicator — so progress stays in view. */}
          <GoalStatusBanner conversationId={conversationId} />
        </div>

        <div className="flex shrink-0 flex-col gap-[6px] pb-4">
          <SkillInstallRestartBanner conversationId={conversationId} />
          <BtwMessages conversationId={conversationId} />
          {errorMessage && (
            <ErrorMessageBanner
              message={errorMessage}
              code={errorCode}
              onDismiss={removeErrorMessage}
              onRetry={
                errorMessage === SERVER_CONNECTION_ERROR_MESSAGE
                  ? () => conversationWebSocket?.reconnect()
                  : undefined
              }
              onReauth={
                isAcpAuthErrorCode(errorCode)
                  ? () => navigate("/settings/agents")
                  : undefined
              }
            />
          )}

          {llmBlocked && !isArchivedConversation && <LlmNotConfiguredBanner />}

          {isArchivedConversation ? (
            // Archived / sandbox-error: show a read-only notice in place of
            // the chat input. The conversation history above is still visible.
            <div
              data-testid="archived-conversation-banner"
              className="mx-1 px-4 py-3 rounded-lg bg-[var(--oh-surface)] border border-[var(--oh-border-subtle)]"
            >
              <p className="text-xs font-semibold text-[var(--oh-foreground)]">
                {sandboxStatus === "ERROR"
                  ? t(I18nKey.CHAT_INTERFACE$ERROR_SANDBOX_TITLE)
                  : t(I18nKey.CHAT_INTERFACE$ARCHIVED_SANDBOX_TITLE)}
              </p>
              <p className="text-xs text-[var(--oh-muted)] mt-0.5">
                {sandboxStatus === "ERROR"
                  ? t(I18nKey.CHAT_INTERFACE$ERROR_SANDBOX_DESCRIPTION)
                  : t(I18nKey.CHAT_INTERFACE$ARCHIVED_SANDBOX_DESCRIPTION)}
              </p>
            </div>
          ) : (
            <div className="relative">
              <div className="pointer-events-none absolute inset-x-0 bottom-full mb-1 z-20">
                <div className="flex justify-between relative">
                  <div className="flex items-end gap-1 pointer-events-auto">
                    <ConfirmationModeEnabled />
                    {isStartingStatus && (
                      <ChatStatusIndicator
                        statusColor={serverStatusColor}
                        status={serverStatusText}
                      />
                    )}
                  </div>

                  {!hitBottom ? (
                    <div className="absolute left-1/2 transform -translate-x-1/2 bottom-0 pointer-events-auto">
                      <ScrollToBottomButton onClick={scrollDomToBottom} />
                    </div>
                  ) : (
                    curAgentState === AgentState.RUNNING && (
                      <div className="absolute left-1/2 transform -translate-x-1/2 bottom-0 pointer-events-auto">
                        <TypingIndicator />
                      </div>
                    )
                  )}
                </div>
              </div>

              <InteractiveChatBox
                onSubmit={handleSendMessage}
                disabled={isNewConversationPending || llmBlocked}
                hasStartedConversation={hasStartedConversation}
              />
            </div>
          )}
        </div>
      </div>
    </ScrollProvider>
  );
}
