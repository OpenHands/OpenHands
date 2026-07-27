import React, {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useState,
  useCallback,
  useMemo,
  useRef,
} from "react";
import { ConversationClient } from "@openhands/typescript-client/clients";

import { useQueryClient } from "@tanstack/react-query";
import { useWebSocket, WebSocketHookOptions } from "#/hooks/use-websocket";
import { SERVER_CONNECTION_ERROR_MESSAGE } from "#/constants/server-connection-error";
import { useEventStore } from "#/stores/use-event-store";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { useCommandStore } from "#/stores/command-store";
import { useBrowserStore } from "#/stores/browser-store";
import { useGoalStore } from "#/stores/goal-store";
import {
  isAgentServerEvent,
  isAgentErrorEvent,
  isUserMessageEvent,
  isActionEvent,
  isConversationStateUpdateEvent,
  isFullStateConversationStateUpdateEvent,
  isAgentStatusConversationStateUpdateEvent,
  isStatsConversationStateUpdateEvent,
  isGoalConversationStateUpdateEvent,
  isExecuteBashActionEvent,
  isExecuteBashObservationEvent,
  isDisplayableErrorEvent,
  isPlanningFileEditorObservationEvent,
  isBrowserObservationEvent,
  isBrowserNavigateActionEvent,
  isSwitchLLMObservationEvent,
  isCanvasUIActionEvent,
} from "#/types/agent-server/type-guards";
import { handleCanvasUIAction } from "#/services/canvas-ui";
import { ConversationStateUpdateEventStats } from "#/types/agent-server/core/events/conversation-state-event";
import type {
  ConversationErrorEvent,
  ServerErrorEvent,
} from "#/types/agent-server/core/events/conversation-state-event";
import { handleActionEventCacheInvalidation } from "#/utils/cache-utils";
import { buildWebSocketUrl } from "#/utils/websocket-url";
import type {
  AppConversation,
  SendMessageRequest,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { useConversationStore } from "#/stores/conversation-store";
import { trackError } from "#/utils/error-handler";
import { useReadConversationFile } from "#/hooks/mutation/use-read-conversation-file";
import useMetricsStore from "#/stores/metrics-store";
import { useConversationHistory } from "#/hooks/query/use-conversation-history";
import { setConversationState } from "#/utils/conversation-local-storage";
import {
  recordModelSwitchMessage,
  seedModelSwitchesFromHistory,
} from "#/hooks/chat/record-model-switch-message";
import {
  invalidateConversationQueries,
  updateConversationLlmModelInCache,
} from "#/hooks/mutation/conversation-mutation-utils";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";

export type WebSocketConnectionState =
  | "CONNECTING"
  | "OPEN"
  | "CLOSED"
  | "CLOSING";

interface SendMessageResult {
  queued: boolean; // true if message was queued for later delivery, false if sent immediately
}

interface ConversationWebSocketContextType {
  connectionState: WebSocketConnectionState;
  sendMessage: (message: SendMessageRequest) => Promise<SendMessageResult>;
  isLoadingHistory: boolean;
  reconnect: () => void;
}

const ConversationWebSocketContext = createContext<
  ConversationWebSocketContextType | undefined
>(undefined);

/**
 * Extract the text body of an echoed user `MessageEvent` for matching against
 * the optimistic pending-message queue. The server wraps the original
 * `args.content` string in one or more `TextContent` entries (alongside any
 * `ImageContent` entries for inline images), so concatenating the `text`
 * fields gives us back the exact prompt we sent.
 */
function extractMessageEventText(
  event: import("#/types/agent-server/core/events/message-event").MessageEvent,
): string {
  return event.llm_message.content
    .filter(
      (part): part is { type: "text"; text: string } => part.type === "text",
    )
    .map((part) => part.text)
    .join("");
}

export function ConversationWebSocketProvider({
  children,
  conversationId,
  conversationUrl,
  sessionApiKey,
  subConversations,
  subConversationIds,
}: {
  children: React.ReactNode;
  conversationId?: string;
  conversationUrl?: string | null;
  sessionApiKey?: string | null;
  subConversations?: AppConversation[];
  subConversationIds?: string[];
}) {
  // Separate connection state tracking for each WebSocket
  const [mainConnectionState, setMainConnectionState] =
    useState<WebSocketConnectionState>("CONNECTING");
  const [planningConnectionState, setPlanningConnectionState] =
    useState<WebSocketConnectionState>("CONNECTING");

  // Track if we've ever successfully connected for each connection
  // Don't show errors until after first successful connection
  const hasConnectedRefMain = React.useRef(false);
  const hasConnectedRefPlanning = React.useRef(false);

  const queryClient = useQueryClient();
  const addEvent = useEventStore((state) => state.addEvent);
  const addEvents = useEventStore((state) => state.addEvents);
  const clearEventsForConversation = useEventStore(
    (state) => state.clearEventsForConversation,
  );
  const { setErrorMessage, removeErrorMessage, clearConnectionError } =
    useErrorMessageStore();
  const consumeMatchingPendingMessage = useOptimisticUserMessageStore(
    (state) => state.consumeMatchingPendingMessage,
  );
  const { setExecutionStatus } = useConversationStateStore();
  const { appendInput, appendOutput } = useCommandStore();
  const resetBrowserStore = useBrowserStore((state) => state.reset);

  // History loading state.
  // - Main conversation history is now loaded via REST (`useConversationHistory`),
  //   so its loading state mirrors the REST query state (see below).
  // - Planning sub-conversation history still streams over the WebSocket using
  //   `resend_mode='all'`, so we keep the count-based detection for it.
  const [isLoadingHistoryPlanning, setIsLoadingHistoryPlanning] =
    useState(true);
  const [expectedEventCountPlanning, setExpectedEventCountPlanning] = useState<
    number | null
  >(null);

  const { setPlanContent } = useConversationStore();

  // Hook for reading conversation file
  const { mutate: readConversationFile } = useReadConversationFile();

  // Track planning-agent received events (still WS-driven).
  const receivedEventCountRefPlanning = useRef(0);

  // Track the latest PlanningFileEditorObservation for Plan.md during history replay
  const latestPlanningFileEventRef = useRef<{
    path: string;
    conversationId: string;
  } | null>(null);

  const isPlanFilePath = (path: string | null): boolean =>
    path?.toUpperCase().endsWith("PLAN.MD") ?? false;

  const handleNonErrorEvent = useCallback(() => {
    // A normal event means connectivity recovered: clear a transient connection
    // error, but keep sticky conversation errors (e.g. a wrong API key).
    clearConnectionError();
  }, [clearConnectionError]);

  // Helper function to update metrics from stats event
  const updateMetricsFromStats = useCallback(
    (event: ConversationStateUpdateEventStats) => {
      if (event.value.usage_to_metrics?.agent) {
        const agentMetrics = event.value.usage_to_metrics.agent;
        const metrics = {
          cost: agentMetrics.accumulated_cost,
          max_budget_per_task: agentMetrics.max_budget_per_task ?? null,
          usage: agentMetrics.accumulated_token_usage
            ? {
                prompt_tokens:
                  agentMetrics.accumulated_token_usage.prompt_tokens,
                completion_tokens:
                  agentMetrics.accumulated_token_usage.completion_tokens,
                cache_read_tokens:
                  agentMetrics.accumulated_token_usage.cache_read_tokens,
                cache_write_tokens:
                  agentMetrics.accumulated_token_usage.cache_write_tokens,
                context_window:
                  agentMetrics.accumulated_token_usage.context_window,
                per_turn_token:
                  agentMetrics.accumulated_token_usage.per_turn_token,
              }
            : null,
        };
        useMetricsStore.getState().setMetrics(metrics);
      }
    },
    [],
  );

  // Initial REST history load: fetch the most recent events and seed the
  // store. Older events are paginated in via `useLoadOlderEvents` when the
  // user scrolls to the top of the chat. The WebSocket connection waits for
  // this query so it can subscribe with `resend_mode='since'` and avoid
  // re-streaming everything REST already returned.
  const {
    data: preloadedHistory,
    isPending: isPreloadingHistory,
    isFetching: isFetchingHistory,
    isError: isPreloadHistoryError,
  } = useConversationHistory(conversationId);

  // Skeleton only on the genuine first load (no cached data yet). On return the
  // cached page is present, so `isPending` is false and we render the
  // last-known discussion immediately while the tail refetch runs in the
  // background — see the socket gate below, which waits on `isFetching`.
  const isLoadingHistoryMain = !!conversationId && isPreloadingHistory;

  // Clear the (global, not conversation-scoped) event store when the active
  // conversation changes, BEFORE the preloaded-history effect below re-seeds
  // it. This MUST live here rather than in the route component: a parent's
  // passive effect runs *after* this child's layout effects, so clearing from
  // the route would wipe the freshly seeded history. On a conversation switch
  // the history page is already cached, so `preloadedHistory` is available
  // synchronously — without ordering the clear first, the user's already-echoed
  // message gets seeded then immediately wiped, leaving only the `since`
  // WebSocket resend (the agent's reply). Re-entering the same conversation is
  // a no-op, so the store survives navigating away to Settings and back.
  useLayoutEffect(() => {
    const nextId = conversationId ?? null;
    if (useEventStore.getState().loadedConversationId === nextId) {
      return;
    }
    // Single atomic action: clears the previous conversation's events and
    // records the new loaded id in one `set`, so no subscriber can observe a
    // half-applied state (events gone but the old id still reported).
    clearEventsForConversation(nextId);
    resetBrowserStore();
  }, [conversationId, clearEventsForConversation, resetBrowserStore]);

  useLayoutEffect(() => {
    if (!preloadedHistory || preloadedHistory.events.length === 0) {
      return;
    }
    addEvents(preloadedHistory.events);

    // The first user message of a cloud start-task conversation is persisted
    // server-side and reaches us via this REST preload, not over the WebSocket
    // (which subscribes with resend_mode='since' after the latest preloaded
    // timestamp). Consume any matching optimistic "Sending…" bubble here too —
    // mirroring the WS handler — so it doesn't linger as a duplicate of the echo.
    if (conversationId) {
      // Rebuild inline "Switched to" messages from the REST-preloaded history.
      // The live store writers (WS handler / user action) never see preloaded
      // events, so without this past model switches wouldn't render on reload.
      // Read the post-`addEvents` `uiEvents` (actions replaced by observations,
      // Think/Finish observations dropped) — not the raw history — so anchors
      // match the ids the renderer actually mounts.
      seedModelSwitchesFromHistory(
        conversationId,
        useEventStore.getState().uiEvents,
      );

      for (const event of preloadedHistory.events) {
        if (isUserMessageEvent(event)) {
          consumeMatchingPendingMessage(
            conversationId,
            extractMessageEventText(event),
          );
        }
      }
    }
  }, [
    preloadedHistory,
    addEvents,
    conversationId,
    consumeMatchingPendingMessage,
  ]);

  /**
   * Timestamp of the latest event we already have from REST. Used as
   * `after_timestamp` when opening the WebSocket so the server only resends
   * events strictly after this point. `null` until the REST query settles
   * (we hold the WS connection open until then to avoid an `all` resend).
   */
  const initialAfterTimestamp = useMemo<string | null>(() => {
    // Wait for the history query to settle — including the refetch fired when
    // returning to a conversation — so we anchor `since` to the freshest event
    // we have rather than a stale cached tail.
    if (isFetchingHistory) return null;
    const events = preloadedHistory?.events ?? [];
    const latest = events[events.length - 1];
    if (!latest || !("timestamp" in latest) || !latest.timestamp) return null;
    return latest.timestamp;
  }, [preloadedHistory, isFetchingHistory]);

  // Build WebSocket URL from props.
  //
  // We deliberately wait for the history fetch to settle before opening the
  // socket so the WS subscription can use `resend_mode='since'` with a
  // meaningful `after_timestamp`. This gate is on `isFetching`, not just the
  // first-load `isPending`, so the refetch fired when returning to a
  // conversation also completes first: the socket bakes `after_timestamp` into
  // its URL at connect time and will NOT re-subscribe when the value changes
  // later (see use-websocket — options live in a ref, reconnect keys on the URL
  // only). Connecting mid-fetch would therefore pin `since` to the stale cached
  // tail and replay the entire backlog over the socket. Without any gate the WS
  // would instead fall back to `resend_mode='all'`. If the REST query errored we
  // fall through and connect with `resend_mode='all'` so the user still sees
  // live events.
  const wsUrl = useMemo(() => {
    if (!conversationId || !conversationUrl) {
      return null;
    }
    if (isFetchingHistory && !isPreloadHistoryError) {
      return null;
    }
    return buildWebSocketUrl(conversationId, conversationUrl);
  }, [
    conversationId,
    conversationUrl,
    isFetchingHistory,
    isPreloadHistoryError,
  ]);

  const planningAgentWsUrl = useMemo(() => {
    if (!subConversations?.length) {
      return null;
    }

    // Currently, there is only one sub-conversation and it uses the planning agent.
    const planningAgentConversation = subConversations[0];

    if (
      !planningAgentConversation?.id ||
      !planningAgentConversation.conversation_url
    ) {
      return null;
    }

    return buildWebSocketUrl(
      planningAgentConversation.id,
      planningAgentConversation.conversation_url,
    );
  }, [subConversations]);

  // Merged connection state - reflects combined status of both connections
  const connectionState = useMemo<WebSocketConnectionState>(() => {
    // If planning agent connection doesn't exist, use main connection state
    if (!planningAgentWsUrl) {
      return mainConnectionState;
    }

    // If either is connecting, merged state is connecting
    if (
      mainConnectionState === "CONNECTING" ||
      planningConnectionState === "CONNECTING"
    ) {
      return "CONNECTING";
    }

    // If both are open, merged state is open
    if (mainConnectionState === "OPEN" && planningConnectionState === "OPEN") {
      return "OPEN";
    }

    // If both are closed, merged state is closed
    if (
      mainConnectionState === "CLOSED" &&
      planningConnectionState === "CLOSED"
    ) {
      return "CLOSED";
    }

    // If either is closing, merged state is closing
    if (
      mainConnectionState === "CLOSING" ||
      planningConnectionState === "CLOSING"
    ) {
      return "CLOSING";
    }

    // Default to closed if states don't match expected patterns
    return "CLOSED";
  }, [mainConnectionState, planningConnectionState, planningAgentWsUrl]);

  useEffect(() => {
    if (
      expectedEventCountPlanning !== null &&
      receivedEventCountRefPlanning.current >= expectedEventCountPlanning &&
      isLoadingHistoryPlanning
    ) {
      setIsLoadingHistoryPlanning(false);
    }
  }, [
    expectedEventCountPlanning,
    isLoadingHistoryPlanning,
    receivedEventCountRefPlanning,
  ]);

  // Call API once after history loading completes if we tracked any PlanningFileEditorObservation events
  useEffect(() => {
    if (!isLoadingHistoryPlanning && latestPlanningFileEventRef.current) {
      const { path, conversationId: currentPlanningConversationId } =
        latestPlanningFileEventRef.current;

      readConversationFile(
        {
          conversationId: currentPlanningConversationId,
          filePath: path,
        },
        {
          onSuccess: (fileContent) => {
            setPlanContent(fileContent);
          },
          onError: (error) => {
            console.warn("Failed to read conversation file:", error);
          },
        },
      );

      // Clear the ref after calling the API
      latestPlanningFileEventRef.current = null;
    }
  }, [isLoadingHistoryPlanning, readConversationFile, setPlanContent]);

  useEffect(() => {
    hasConnectedRefMain.current = false;
    setIsLoadingHistoryPlanning(!!subConversationIds?.length);
    setExpectedEventCountPlanning(null);
    receivedEventCountRefPlanning.current = 0;
    // Reset the tracked event ref when sub-conversations change
    latestPlanningFileEventRef.current = null;
  }, [subConversationIds]);

  // Reset hasConnected flags when the conversation changes.
  useEffect(() => {
    hasConnectedRefMain.current = false;
    hasConnectedRefPlanning.current = false;
    // Reset the tracked event ref when conversation changes
    latestPlanningFileEventRef.current = null;
  }, [conversationId]);

  // Merged loading history state - true if either connection is still loading
  const isLoadingHistory = useMemo(
    () => isLoadingHistoryMain || isLoadingHistoryPlanning,
    [isLoadingHistoryMain, isLoadingHistoryPlanning],
  );

  // Separate message handlers for each connection
  const handleMainMessage = useCallback(
    (messageEvent: MessageEvent) => {
      try {
        const event = JSON.parse(messageEvent.data);

        // History loading for the main conversation is REST-driven now;
        // every WS message is a new event we add to the store.

        // Use type guard to validate v1 event structure
        if (isAgentServerEvent(event)) {
          // A reconnect replays the backlog from a stale anchor. The store
          // dedups by id, but the side-effects below aren't idempotent, so skip
          // them for replayed events (#1656).
          const isDuplicateEvent = useEventStore
            .getState()
            .eventIds.has(event.id);
          const switchLLMObservation = isSwitchLLMObservationEvent(event)
            ? event
            : null;
          addEvent(event);
          if (isDuplicateEvent) {
            return;
          }

          // Handle displayable error events - show error banner
          // AgentErrorEvent errors are displayed inline in the chat, not as banners
          if (isDisplayableErrorEvent(event)) {
            const errorEvent = event as
              | ConversationErrorEvent
              | ServerErrorEvent;
            trackError({
              message: errorEvent.detail,
              source: "conversation",
              metadata: {
                eventId: errorEvent.id,
                errorCode: errorEvent.code,
              },
            });
            setErrorMessage(errorEvent.detail, "conversation", errorEvent.code);
          } else {
            handleNonErrorEvent();
          }

          // LLM errors render inline in the chat (see ErrorEventMessage); track
          // them for analytics but keep them out of the banner above the chat box.
          if (isAgentErrorEvent(event)) {
            trackError({
              message: event.error,
              source: "agent",
              metadata: {
                eventId: event.id,
                toolName: event.tool_name,
                toolCallId: event.tool_call_id,
              },
            });
          }

          // Clear optimistic user message when a user message is confirmed.
          // We match by the echoed text content (with FIFO fallback inside the
          // store), so an echo for "second" pops "second" — not whichever
          // pending entry happens to be oldest — protecting against any
          // out-of-order delivery between conversations or sub-agents.
          if (isUserMessageEvent(event)) {
            if (conversationId) {
              consumeMatchingPendingMessage(
                conversationId,
                extractMessageEventText(event),
              );
              // Clear draft from localStorage - message was successfully delivered
              setConversationState(conversationId, { draftMessage: null });
            }
          }

          // Handle cache invalidation for ActionEvent
          if (isActionEvent(event)) {
            const currentConversationId =
              conversationId || "test-conversation-id"; // TODO: Get from context
            handleActionEventCacheInvalidation(
              event,
              currentConversationId,
              queryClient,
            );
          }

          // Handle conversation state updates
          // TODO: Tests
          if (isConversationStateUpdateEvent(event)) {
            if (isFullStateConversationStateUpdateEvent(event)) {
              setExecutionStatus(event.value.execution_status);
            }
            if (isAgentStatusConversationStateUpdateEvent(event)) {
              setExecutionStatus(event.value);
            }
            if (isStatsConversationStateUpdateEvent(event)) {
              updateMetricsFromStats(event);
            }
            // Mirror goal status into the store. Intentionally duplicated across
            // the main and planning WebSocket handlers (like the execution_status
            // and stats branches above), not a merge artifact.
            if (isGoalConversationStateUpdateEvent(event) && conversationId) {
              useGoalStore.getState().setStatus(conversationId, event.value);
            }
          }

          // Handle ExecuteBashAction events - add command as input to terminal
          if (isExecuteBashActionEvent(event)) {
            appendInput(event.action.command);
          }

          // Handle ExecuteBashObservation events - add output to terminal
          if (isExecuteBashObservationEvent(event)) {
            // Extract text content from the observation content array
            const textContent = event.observation.content
              .filter((c) => c.type === "text")
              .map((c) => c.text)
              .join("\n");
            appendOutput(textContent);
          }

          // Handle BrowserObservation events - update browser store with screenshot
          if (isBrowserObservationEvent(event)) {
            const { screenshot_data: screenshotData } = event.observation;
            if (screenshotData) {
              const screenshotSrc = screenshotData.startsWith("data:")
                ? screenshotData
                : `data:image/png;base64,${screenshotData}`;
              useBrowserStore.getState().setScreenshotSrc(screenshotSrc);
            }
          }

          // Handle BrowserNavigateAction events - update browser store with URL
          if (isBrowserNavigateActionEvent(event)) {
            useBrowserStore.getState().setUrl(event.action.url);
          }

          if (
            conversationId &&
            switchLLMObservation &&
            !switchLLMObservation.observation.is_error
          ) {
            recordModelSwitchMessage(
              conversationId,
              switchLLMObservation.observation.profile_name,
            );

            // Mirror the user-driven `/model` path: persist the profile so the
            // chat-header switcher shows the right name after a reload, even
            // when several profiles share a model (#1082).
            const prevMetadata = getStoredConversationMetadata(conversationId);
            setStoredConversationMetadata(conversationId, {
              selected_repository: prevMetadata?.selected_repository ?? null,
              selected_branch: prevMetadata?.selected_branch ?? null,
              git_provider: prevMetadata?.git_provider ?? null,
              selected_workspace: prevMetadata?.selected_workspace ?? null,
              active_profile: switchLLMObservation.observation.profile_name,
              // Full-object replace: carry the plugins snapshot forward so the
              // in-conversation plugins view survives a profile switch.
              plugins: prevMetadata?.plugins ?? null,
            });

            if (switchLLMObservation.observation.active_model) {
              updateConversationLlmModelInCache(
                queryClient,
                conversationId,
                switchLLMObservation.observation.active_model,
              );
            }

            invalidateConversationQueries(queryClient, conversationId);
          }

          // Handle canvas_ui ActionEvents from both the legacy Python tool and
          // the client-defined JSON tool. The server acknowledges immediately;
          // the actual UI change happens here on the client.
          if (isCanvasUIActionEvent(event)) {
            handleCanvasUIAction(event.action, conversationId ?? null);
          }
        }
      } catch (error) {
        console.warn("Failed to parse WebSocket message as JSON:", error);
      }
    },
    [
      addEvent,
      setErrorMessage,
      consumeMatchingPendingMessage,
      queryClient,
      conversationId,
      setExecutionStatus,
      appendInput,
      appendOutput,
      updateMetricsFromStats,
      handleNonErrorEvent,
    ],
  );

  const handlePlanningMessage = useCallback(
    (messageEvent: MessageEvent) => {
      try {
        const event = JSON.parse(messageEvent.data);

        // Track received events for history loading (count ALL events from WebSocket)
        // Always count when loading, even if we don't have the expected count yet
        if (isLoadingHistoryPlanning) {
          receivedEventCountRefPlanning.current += 1;

          if (
            expectedEventCountPlanning !== null &&
            receivedEventCountRefPlanning.current >= expectedEventCountPlanning
          ) {
            setIsLoadingHistoryPlanning(false);
          }
        }

        // Use type guard to validate v1 event structure
        if (isAgentServerEvent(event)) {
          // Skip non-idempotent side-effects for replayed events, as in the
          // main handler (#1656).
          const isDuplicateEvent = useEventStore
            .getState()
            .eventIds.has(event.id);
          // Mark this event as coming from the planning agent
          const eventWithPlanningFlag = {
            ...event,
            isFromPlanningAgent: true,
          };
          addEvent(eventWithPlanningFlag);
          if (isDuplicateEvent) {
            return;
          }

          // Handle displayable error events - show error banner
          // AgentErrorEvent errors are displayed inline in the chat, not as banners
          if (isDisplayableErrorEvent(event)) {
            const errorEvent = event as
              | ConversationErrorEvent
              | ServerErrorEvent;
            trackError({
              message: errorEvent.detail,
              source: "planning_conversation",
              metadata: {
                eventId: errorEvent.id,
                errorCode: errorEvent.code,
              },
            });
            setErrorMessage(errorEvent.detail, "conversation", errorEvent.code);
          } else {
            handleNonErrorEvent();
          }

          // LLM errors render inline in the chat (see ErrorEventMessage); track
          // them for analytics but keep them out of the banner above the chat box.
          if (isAgentErrorEvent(event)) {
            trackError({
              message: event.error,
              source: "planning_agent",
              metadata: {
                eventId: event.id,
                toolName: event.tool_name,
                toolCallId: event.tool_call_id,
              },
            });
          }

          // Clear optimistic user message when a user message is confirmed.
          // Always scope to the main `conversationId` (where the user types)
          // and match on the echoed content so the planning sub-agent's own
          // events can never consume a main-conversation pending entry.
          if (isUserMessageEvent(event)) {
            if (conversationId) {
              consumeMatchingPendingMessage(
                conversationId,
                extractMessageEventText(event),
              );
              setConversationState(conversationId, { draftMessage: null });
            }
          }

          // Handle cache invalidation for ActionEvent
          if (isActionEvent(event)) {
            const planningAgentConversation = subConversations?.[0];
            const currentConversationId =
              planningAgentConversation?.id || "test-conversation-id"; // TODO: Get from context
            handleActionEventCacheInvalidation(
              event,
              currentConversationId,
              queryClient,
            );
          }

          // Handle conversation state updates
          // TODO: Tests
          if (isConversationStateUpdateEvent(event)) {
            if (isFullStateConversationStateUpdateEvent(event)) {
              setExecutionStatus(event.value.execution_status);
            }
            if (isAgentStatusConversationStateUpdateEvent(event)) {
              setExecutionStatus(event.value);
            }
            if (isStatsConversationStateUpdateEvent(event)) {
              updateMetricsFromStats(event);
            }
            // Mirror goal status into the store. Intentionally duplicated across
            // the main and planning WebSocket handlers (like the execution_status
            // and stats branches above), not a merge artifact.
            if (isGoalConversationStateUpdateEvent(event) && conversationId) {
              useGoalStore.getState().setStatus(conversationId, event.value);
            }
          }

          // Handle ExecuteBashAction events - add command as input to terminal
          if (isExecuteBashActionEvent(event)) {
            appendInput(event.action.command);
          }

          // Handle ExecuteBashObservation events - add output to terminal
          if (isExecuteBashObservationEvent(event)) {
            // Extract text content from the observation content array
            const textContent = event.observation.content
              .filter((c) => c.type === "text")
              .map((c) => c.text)
              .join("\n");
            appendOutput(textContent);
          }

          // Handle PlanningFileEditorObservation - only update plan for Plan.md
          if (isPlanningFileEditorObservationEvent(event)) {
            const { path } = event.observation;
            if (isPlanFilePath(path)) {
              const planningAgentConversation = subConversations?.[0];
              const planningConversationId = planningAgentConversation?.id;

              if (planningConversationId && path) {
                if (isLoadingHistoryPlanning) {
                  latestPlanningFileEventRef.current = {
                    path,
                    conversationId: planningConversationId,
                  };
                } else {
                  readConversationFile(
                    {
                      conversationId: planningConversationId,
                      filePath: path,
                    },
                    {
                      onSuccess: (fileContent) => {
                        setPlanContent(fileContent);
                      },
                      onError: (error) => {
                        console.warn(
                          "Failed to read conversation file:",
                          error,
                        );
                      },
                    },
                  );
                }
              }
            }
          }
        }
      } catch (error) {
        console.warn("Failed to parse WebSocket message as JSON:", error);
      }
    },
    [
      addEvent,
      isLoadingHistoryPlanning,
      expectedEventCountPlanning,
      setErrorMessage,
      consumeMatchingPendingMessage,
      queryClient,
      subConversations,
      conversationId,
      setExecutionStatus,
      appendInput,
      appendOutput,
      readConversationFile,
      setPlanContent,
      updateMetricsFromStats,
      handleNonErrorEvent,
    ],
  );

  // Separate WebSocket options for main connection
  const mainWebsocketOptions: WebSocketHookOptions = useMemo(() => {
    // History was already loaded over REST (`useConversationHistory`).
    // Subscribe with `resend_mode='since'` so the server only resends events
    // strictly after the latest one we already have. If REST returned no
    // events at all (brand-new conversation), fall back to `'all'` so any
    // events that may have been written between the REST call and the WS
    // handshake still show up. Dedup in the event store handles overlap.
    const queryParams: Record<string, string | boolean> = initialAfterTimestamp
      ? { resend_mode: "since", after_timestamp: initialAfterTimestamp }
      : { resend_mode: "all" };

    // Add session_api_key if available
    if (sessionApiKey) {
      queryParams.session_api_key = sessionApiKey;
    }

    return {
      queryParams,
      reconnect: { enabled: true },
      onOpen: () => {
        setMainConnectionState("OPEN");
        hasConnectedRefMain.current = true; // Mark that we've successfully connected
        clearConnectionError(); // Clear a previous connection error; keep sticky conversation errors
      },
      onClose: () => {
        setMainConnectionState("CLOSED");
      },
      onError: () => {
        setMainConnectionState("CLOSED");
        // Only show error message if we've previously connected successfully
        if (hasConnectedRefMain.current) {
          setErrorMessage(SERVER_CONNECTION_ERROR_MESSAGE, "connection");
        }
      },
      onMessage: handleMainMessage,
    };
  }, [
    handleMainMessage,
    setErrorMessage,
    clearConnectionError,
    sessionApiKey,
    initialAfterTimestamp,
  ]);

  // Separate WebSocket options for planning agent connection
  const planningWebsocketOptions: WebSocketHookOptions = useMemo(() => {
    const queryParams: Record<string, string | boolean> = {
      resend_all: true,
    };

    // Add session_api_key if available
    if (sessionApiKey) {
      queryParams.session_api_key = sessionApiKey;
    }

    const planningAgentConversation = subConversations?.[0];

    return {
      queryParams,
      reconnect: { enabled: true },
      onOpen: async () => {
        setPlanningConnectionState("OPEN");
        hasConnectedRefPlanning.current = true; // Mark that we've successfully connected
        clearConnectionError(); // Clear a previous connection error; keep sticky conversation errors

        // Fetch expected event count for history loading detection
        if (
          planningAgentConversation?.id &&
          planningAgentConversation.conversation_url
        ) {
          try {
            const count = await EventService.getEventCount(
              planningAgentConversation.id,
              planningAgentConversation.conversation_url,
              planningAgentConversation.session_api_key,
            );
            setExpectedEventCountPlanning(count);

            // If no events expected, mark as loaded immediately
            if (count === 0) {
              setIsLoadingHistoryPlanning(false);
            }
          } catch (error) {
            // Fall back to marking as loaded to avoid infinite loading state
            setIsLoadingHistoryPlanning(false);
          }
        }
      },
      onClose: () => {
        setPlanningConnectionState("CLOSED");
      },
      onError: () => {
        setPlanningConnectionState("CLOSED");
        // Only show error message if we've previously connected successfully
        if (hasConnectedRefPlanning.current) {
          setErrorMessage(SERVER_CONNECTION_ERROR_MESSAGE, "connection");
        }
      },
      onMessage: handlePlanningMessage,
    };
  }, [
    handlePlanningMessage,
    setErrorMessage,
    clearConnectionError,
    sessionApiKey,
    subConversations,
  ]);

  // Only attempt WebSocket connection when we have a valid URL
  // This prevents connection attempts during task polling phase
  const websocketUrl = wsUrl;
  const { socket: mainSocket, reconnect: reconnectMain } = useWebSocket(
    websocketUrl || "",
    mainWebsocketOptions,
  );

  const { socket: planningAgentSocket, reconnect: reconnectPlanning } =
    useWebSocket(planningAgentWsUrl || "", planningWebsocketOptions);

  const reconnect = useCallback(() => {
    removeErrorMessage();
    const currentMode = useConversationStore.getState().conversationMode;
    if (currentMode === "plan" && planningAgentWsUrl) {
      reconnectPlanning();
      return;
    }
    reconnectMain();
  }, [
    planningAgentWsUrl,
    reconnectMain,
    reconnectPlanning,
    removeErrorMessage,
  ]);

  // V1 send message function via WebSocket
  // Falls back to REST API queue when WebSocket is not connected
  const sendMessage = useCallback(
    async (message: SendMessageRequest): Promise<SendMessageResult> => {
      const currentMode = useConversationStore.getState().conversationMode;
      const currentSocket =
        currentMode === "plan" ? planningAgentSocket : mainSocket;

      if (currentSocket?.readyState !== WebSocket.OPEN) {
        // WebSocket not connected - queue message via REST API
        // Message will be delivered automatically when conversation becomes ready
        if (!conversationId) {
          const error = new Error("No conversation ID available");
          setErrorMessage(error.message);
          throw error;
        }

        try {
          await new ConversationClient(getAgentServerClientOptions()).sendEvent(
            conversationId,
            {
              role: "user",
              content: message.content,
            },
            { run: true },
          );
          // Message queued successfully - it will be delivered when ready
          // Return queued: true so caller knows not to show optimistic UI
          return { queued: true };
        } catch (error) {
          const errorMessage =
            error instanceof Error
              ? error.message
              : "Failed to queue message for delivery";
          setErrorMessage(errorMessage);
          throw error;
        }
      }

      try {
        // Send message through WebSocket as JSON with run: true so the
        // agent loop starts automatically in async mode.
        currentSocket.send(JSON.stringify({ ...message, run: true }));
        return { queued: false };
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Failed to send message";
        setErrorMessage(errorMessage);
        throw error;
      }
    },
    [mainSocket, planningAgentSocket, setErrorMessage, conversationId],
  );

  // Track main socket state changes
  useEffect(() => {
    // Only process socket updates if we have a valid URL and socket
    if (mainSocket && wsUrl) {
      // Update state based on socket readyState
      const updateState = () => {
        switch (mainSocket.readyState) {
          case WebSocket.CONNECTING:
            setMainConnectionState("CONNECTING");
            break;
          case WebSocket.OPEN:
            setMainConnectionState("OPEN");
            break;
          case WebSocket.CLOSING:
            setMainConnectionState("CLOSING");
            break;
          case WebSocket.CLOSED:
            setMainConnectionState("CLOSED");
            break;
          default:
            setMainConnectionState("CLOSED");
            break;
        }
      };

      updateState();
    }
  }, [mainSocket, wsUrl]);

  // Track planning agent socket state changes
  useEffect(() => {
    // Only process socket updates if we have a valid URL and socket
    if (planningAgentSocket && planningAgentWsUrl) {
      // Update state based on socket readyState
      const updateState = () => {
        switch (planningAgentSocket.readyState) {
          case WebSocket.CONNECTING:
            setPlanningConnectionState("CONNECTING");
            break;
          case WebSocket.OPEN:
            setPlanningConnectionState("OPEN");
            break;
          case WebSocket.CLOSING:
            setPlanningConnectionState("CLOSING");
            break;
          case WebSocket.CLOSED:
            setPlanningConnectionState("CLOSED");
            break;
          default:
            setPlanningConnectionState("CLOSED");
            break;
        }
      };

      updateState();
    }
  }, [planningAgentSocket, planningAgentWsUrl]);

  const contextValue = useMemo(
    () => ({ connectionState, sendMessage, isLoadingHistory, reconnect }),
    [connectionState, sendMessage, isLoadingHistory, reconnect],
  );

  return (
    <ConversationWebSocketContext.Provider value={contextValue}>
      {children}
    </ConversationWebSocketContext.Provider>
  );
}

export const useConversationWebSocket =
  (): ConversationWebSocketContextType | null => {
    const context = useContext(ConversationWebSocketContext);
    // Return null instead of throwing when not in provider
    // This allows the hook to be called conditionally based on conversation version
    return context || null;
  };
