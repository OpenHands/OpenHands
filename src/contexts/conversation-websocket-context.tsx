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
  isClassifyAndSwitchLLMObservationEvent,
  isCanvasUIActionEvent,
  isLaunchChildConversationActionEvent,
} from "#/types/agent-server/type-guards";
import {
  asSessionFrame,
  type SessionFrame,
} from "#/types/agent-server/session-frames";
import {
  createStreamingDeltaBatcher,
  StreamingDeltaBatcher,
} from "#/utils/streaming-delta-batcher";
import { handleCanvasUIAction } from "#/services/canvas-ui";
import { handleLaunchChildConversationAction } from "#/services/child-conversation-launch";
import { ConversationStateUpdateEventStats } from "#/types/agent-server/core/events/conversation-state-event";
import type {
  ConversationErrorEvent,
  ServerErrorEvent,
} from "#/types/agent-server/core/events/conversation-state-event";
import { handleActionEventCacheInvalidation } from "#/utils/cache-utils";
import { buildWebSocketUrl } from "#/utils/websocket-url";
import { createSeqCursor, type SeqCursor } from "#/utils/session-seq-cursor";
import type {
  AppConversation,
  SendMessageRequest,
} from "#/api/conversation-service/agent-server-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { useConversationStore } from "#/stores/conversation-store";
import { isPlanningMode } from "#/utils/conversation-mode";
import { trackError } from "#/utils/error-handler";
import { useReadConversationFile } from "#/hooks/mutation/use-read-conversation-file";
import useMetricsStore, { type MetricsState } from "#/stores/metrics-store";
import { useConversationHistory } from "#/hooks/query/use-conversation-history";
import { setConversationState } from "#/utils/conversation-local-storage";
import {
  recordModelSwitchMessage,
  seedModelSwitchesFromHistory,
  stampActiveLlmProfile,
} from "#/hooks/chat/record-model-switch-message";
import {
  invalidateConversationQueries,
  updateConversationLlmModelInCache,
} from "#/hooks/mutation/conversation-mutation-utils";
import {
  findPhasePlannerConversationId,
  isPlanFilePath,
  buildPhasePlanPath,
} from "#/utils/plan-file";
import {
  matchDeepPlanDocumentFile,
  type DeepPlanPhaseId,
} from "#/utils/deep-plan";
import { deepPlanUnavailableDocuments } from "#/utils/deep-plan-machine";

export type WebSocketConnectionState =
  | "CONNECTING"
  | "OPEN"
  | "CLOSED"
  | "CLOSING";

/** Which of the two session sockets a connection error came from. */
type ConnectionErrorSource = "main" | "planning";

interface SendMessageResult {
  queued: boolean; // true if message was queued for later delivery, false if sent immediately
}

interface ConversationWebSocketContextType {
  connectionState: WebSocketConnectionState;
  /**
   * The main connection's own state, unmerged with the planning connection —
   * see `useMainWebSocketStatus`.
   */
  mainConnectionState: WebSocketConnectionState;
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

/**
 * Route one frame of `/sockets/session/{id}`.
 *
 * Progress frames (`item_started` / `delta` / `item_aborted`) are applied to
 * the streaming slot here and produce no event. Durable and transient frames
 * are unwrapped and returned for the caller's normal event handling; a durable
 * frame also advances the resume cursor and stamps its `seq` on the event, so
 * the UI can place a slot relative to it.
 */
const routeSessionFrame = (
  raw: unknown,
  batcher: StreamingDeltaBatcher | null,
  /** Report a durable frame's `seq` to this socket's resume cursor. */
  advanceCursor: (seq: number) => void,
  slotMeta: { isFromPlanningAgent?: boolean } = {},
): unknown | null => {
  const frame: SessionFrame | null = asSessionFrame(raw);
  if (!frame) {
    return null;
  }

  if (frame.type === "delta") {
    batcher?.enqueue(frame);
    return null;
  }

  // Nothing else may overtake text that was streamed ahead of it.
  batcher?.flush();

  switch (frame.type) {
    case "item_started":
      useEventStore.getState().openStreamingSlot(frame, slotMeta);
      return null;
    case "item_aborted":
      // Nothing durable is coming to supersede the provisional text.
      useEventStore.getState().abortStreamingSlot(frame.item_id, frame.attempt);
      return null;
    case "durable":
      advanceCursor(frame.seq);
      return { ...frame.event, seq: frame.seq };
    case "transient":
      return frame.event;
    case "error":
      // The server answers a rejected inbound message with an ErrorFrame and
      // keeps the socket open ("not worth dropping over"), so there is no
      // reconnect and nothing else will ever surface this to the user.
      useErrorMessageStore
        .getState()
        .setErrorMessage(frame.detail, "conversation", frame.code);
      return null;
    default:
      // `sync` needs nothing: the cursor advances on durable frames.
      return null;
  }
};

export function ConversationWebSocketProvider({
  children,
  conversationId,
  conversationUrl,
  sessionApiKey,
  subConversations,
  subConversationIds,
  deepPlanWorkingDir,
}: {
  children: React.ReactNode;
  conversationId?: string;
  conversationUrl?: string | null;
  sessionApiKey?: string | null;
  subConversations?: AppConversation[];
  subConversationIds?: string[];
  /**
   * The parent conversation's resolved workspace dir. Used to rebuild a phase
   * document's absolute path when the file must be re-read during a restore and
   * no live file observation named it. Mirrors the local read path's default
   * (`<workingDir>/.agents_tmp/<outputFile>`).
   */
  deepPlanWorkingDir?: string | null;
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
  // Sockets that are currently down. Both sockets raise the same banner, so a
  // single "last error source" cannot represent a failure of both — retrying it
  // would leave the other one closed. `reconnect` retries every entry here, and
  // each socket's own `onOpen` removes itself so a healthy socket is not
  // reconnected on the next Retry.
  const failedSocketsRef = useRef<Set<ConnectionErrorSource>>(new Set());

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

  // Coalesce streaming deltas to ≤1 store commit/render per frame.
  // Separate batchers keep the main and planning streams from ever merging.
  const mainDeltaBatcherRef = useRef<StreamingDeltaBatcher | null>(null);
  if (mainDeltaBatcherRef.current === null) {
    mainDeltaBatcherRef.current = createStreamingDeltaBatcher((frames) => {
      useEventStore.getState().appendStreamingDeltas(frames);
      // A delta means connectivity recovered — mirror handleNonErrorEvent.
      useErrorMessageStore.getState().clearConnectionError();
    });
  }
  const planningDeltaBatcherRef = useRef<StreamingDeltaBatcher | null>(null);
  if (planningDeltaBatcherRef.current === null) {
    planningDeltaBatcherRef.current = createStreamingDeltaBatcher((frames) => {
      useEventStore
        .getState()
        .appendStreamingDeltas(frames, { isFromPlanningAgent: true });
      useErrorMessageStore.getState().clearConnectionError();
    });
  }

  // Resume cursors, one per socket. Started at connect time (see the lazy
  // `queryParams` below) and advanced per durable frame without re-rendering.
  const mainCursorRef = useRef<SeqCursor>(createSeqCursor());
  const planningCursorRef = useRef<SeqCursor>(createSeqCursor());

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

  const {
    setPlanContent,
    setDeepPlanDocument,
    failDeepPlanDocumentRestore,
    conversationMode,
  } = useConversationStore();
  const deepPlan = useConversationStore((state) => state.deepPlan);

  useEffect(() => {
    setPlanContent(null);
  }, [conversationId, setPlanContent]);

  const { mutate: readConversationFile } = useReadConversationFile();

  // Track planning-agent received events (still WS-driven).
  const receivedEventCountRefPlanning = useRef(0);

  // Track the latest PlanningFileEditorObservation for Plan.md during history replay
  const latestPlanningFileEventRef = useRef<{
    path: string;
    conversationId: string;
  } | null>(null);

  // Deep-plan phase documents the planner has written, keyed by phase. The
  // reference validator reads these at a checkpoint, so a document produced
  // during history replay has to be re-read from disk afterwards.
  const latestDeepPlanFileEventsRef = useRef<
    Map<DeepPlanPhaseId, { path: string; conversationId: string }>
  >(new Map());

  // Phases whose disk re-read has already been kicked off for the current
  // conversation. Guards the restore effect against re-issuing a read while the
  // first one is still in flight (the store update that clears `pending`
  // arrives only on success). Keyed to the conversation it was populated for so
  // a switch starts fresh.
  const restoreAttemptedRef = useRef<Set<DeepPlanPhaseId>>(new Set());
  const restoreAttemptedConversationRef = useRef<string | undefined>(undefined);

  // Resolve the planner conversation the phase document should be read from.
  // Deep Planning runs one planner per phase, each pinned to that phase's
  // document; every planner shares the parent's workspace, so the file can be
  // fetched through any of them — but resolving the phase's own planner keeps
  // the read scoped to a conversation that legitimately owns the file and
  // avoids assuming the phase-tagged planner is always first in the list.
  const resolvePhaseConversationId = useCallback(
    (phase: DeepPlanPhaseId, fallbackId: string): string => {
      return (
        findPhasePlannerConversationId(
          subConversations,
          conversationId,
          phase,
        ) ?? fallbackId
      );
    },
    [subConversations, conversationId],
  );

  const handleNonErrorEvent = useCallback(() => {
    // A normal event means connectivity recovered: clear a transient connection
    // error, but keep sticky conversation errors (e.g. a wrong API key).
    clearConnectionError();
  }, [clearConnectionError]);

  // Helper function to update metrics from stats event
  const updateMetricsFromStats = useCallback(
    (event: ConversationStateUpdateEventStats) => {
      const usageToMetrics = event.value.usage_to_metrics;
      if (!usageToMetrics) {
        return;
      }

      // usage_to_metrics is keyed by arbitrary LLM usage ids ("default",
      // "condenser", "profile:<name>:<uuid>", …) — combine across all of
      // them, mirroring getCombinedMetrics on the REST path.
      const combined = Object.values(usageToMetrics).reduce<{
        cost: number;
        maxBudgetPerTask: number | null;
        usage: MetricsState["usage"];
      }>(
        (acc, metrics) => {
          acc.cost += metrics.accumulated_cost;
          if (
            acc.maxBudgetPerTask === null &&
            metrics.max_budget_per_task !== null
          ) {
            acc.maxBudgetPerTask = metrics.max_budget_per_task;
          }
          const tokenUsage = metrics.accumulated_token_usage;
          if (tokenUsage) {
            acc.usage = {
              prompt_tokens:
                (acc.usage?.prompt_tokens ?? 0) + tokenUsage.prompt_tokens,
              completion_tokens:
                (acc.usage?.completion_tokens ?? 0) +
                tokenUsage.completion_tokens,
              cache_read_tokens:
                (acc.usage?.cache_read_tokens ?? 0) +
                tokenUsage.cache_read_tokens,
              cache_write_tokens:
                (acc.usage?.cache_write_tokens ?? 0) +
                tokenUsage.cache_write_tokens,
              context_window: Math.max(
                acc.usage?.context_window ?? 0,
                tokenUsage.context_window,
              ),
              per_turn_token: Math.max(
                acc.usage?.per_turn_token ?? 0,
                tokenUsage.per_turn_token,
              ),
            };
          }
          return acc;
        },
        { cost: 0, maxBudgetPerTask: null, usage: null },
      );

      useMetricsStore.getState().setMetrics({
        cost: combined.cost,
        max_budget_per_task: combined.maxBudgetPerTask,
        usage: combined.usage,
      });
    },
    [],
  );

  // Initial REST history load: fetch the most recent events and seed the
  // store. Older events are paginated in via `useLoadOlderEvents` when the
  // user scrolls to the top of the chat. The WebSocket connection waits for
  // this query so it can subscribe with `resend_mode='since'` and avoid
  // re-streaming everything REST already returned.
  const { data: preloadedHistory, isPending: isPreloadingHistory } =
    useConversationHistory(conversationId);

  // Skeleton only on the genuine first load (no cached data yet). On return the
  // cached page is present, so `isPending` is false and we render the
  // last-known discussion immediately while the tail refetch runs in the
  // background — the socket gate below also keys on `isPending`, so that
  // refetch never drops a live socket.
  const isLoadingHistoryMain = !!conversationId && isPreloadingHistory;

  // First-connect cursor from the REST page (see `afterSeq`). Read lazily by
  // the socket's `queryParams`; after the first connect the socket's own
  // cursor takes over.
  const historyAfterSeqRef = useRef<number | null>(null);
  useEffect(() => {
    historyAfterSeqRef.current = preloadedHistory?.afterSeq ?? null;
  }, [preloadedHistory]);

  // The planner has its own log: reset its cursor when it is a different one.
  const planningSocketConversationId = subConversations?.[0]?.id ?? null;
  useEffect(() => {
    planningCursorRef.current.clear();
  }, [planningSocketConversationId]);

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
    // The metrics store is conversation-scoped state too: without a reset the
    // previous conversation's usage/cost keeps rendering in the new
    // conversation's meter until fresh WS stats arrive — and a brand-new
    // conversation sends none, so the stale figure stuck indefinitely.
    useMetricsStore.getState().resetMetrics();
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
            event,
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

  // Build WebSocket URL from props.
  //
  // We deliberately wait for the FIRST history load (`isPending`: no data for
  // this query key yet) before opening the socket, so the WS subscription can
  // use `resend_mode='since'` with a meaningful `after_timestamp` instead of
  // falling back to `resend_mode='all'`. The gate is intentionally NOT on
  // `isFetching`: background refetches (e.g. the `refetchOnMount` fired when
  // returning to a conversation) must never tear a live socket down — on a
  // flaky link that caused a refetch → teardown → reconnect → refetch loop
  // that kept the conversation stuck at "Connecting" for minutes. Connecting
  // during a background refetch anchors `since` to the cached tail; the
  // overlap with the refetched page is deduped by the event store and the
  // `isDuplicateEvent` guards in the message handlers. A query-key reset
  // (backend swap / new session key) makes `isPending` true again, so a
  // genuine reset still re-gates. If the initial load errors, `isPending`
  // flips false and we fall through to connect with `resend_mode='all'` so
  // the user still sees live events.
  const wsUrl = useMemo(() => {
    if (!conversationId || !conversationUrl) {
      return null;
    }
    if (isPreloadingHistory) {
      return null;
    }
    return buildWebSocketUrl(conversationId, conversationUrl);
  }, [conversationId, conversationUrl, isPreloadingHistory]);

  // Derived from `subConversationIds` (the pre-filtered, tag-verified id
  // list) rather than the resolved `subConversations` entry, which lands a
  // tick later — routing on it would leave a window where the first prompt
  // sent in plan mode fell through to the parent (the code agent) instead of
  // the planner.
  const planningConversationId = useMemo(
    () => subConversationIds?.[0] ?? null,
    [subConversationIds],
  );

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

  // Same for deep-plan phase documents: re-read the latest write of each so the
  // reference validator has the chain after a reload, not just for writes that
  // happen to stream in live.
  useEffect(() => {
    if (isLoadingHistoryPlanning) return;
    const pending = latestDeepPlanFileEventsRef.current;
    if (pending.size === 0) return;
    latestDeepPlanFileEventsRef.current = new Map();
    for (const [phase, { path, conversationId: fallbackId }] of pending) {
      readConversationFile(
        {
          conversationId: resolvePhaseConversationId(phase, fallbackId),
          filePath: path,
        },
        {
          onSuccess: (fileContent) => setDeepPlanDocument(phase, fileContent),
          onError: (error) => {
            console.warn("Failed to read deep-plan document:", error);
          },
        },
      );
    }
  }, [
    isLoadingHistoryPlanning,
    readConversationFile,
    setDeepPlanDocument,
    resolvePhaseConversationId,
  ]);

  // Restore the persisted phase documents after a reload or an in-app
  // conversation switch. The slim persisted state keeps only hashes, so the
  // bodies start empty; history replay only fills the *active* planner, and a
  // completed phase's planner never re-emits its file. Every vouched-for phase
  // that is still missing is therefore read from disk here, so the checkpoint
  // can validate upstream citations instead of seeing each as a missing
  // document. Reads are tracked per phase so a phase already filled by live
  // history is not re-read, and a read that lands late cannot overwrite a live
  // write (setDeepPlanDocument rejects identical bytes and invalidates on a
  // genuine change).
  useEffect(() => {
    if (!conversationId) return;
    if (!conversationUrl || isLoadingHistoryPlanning) return;
    if (conversationMode !== "deep-plan") return;

    const targetIds = subConversationIds ?? [];
    if (targetIds.length === 0) return;

    // A conversation switch swaps the store contents; drop the previous
    // conversation's in-flight tracking so its documents are restored anew.
    if (restoreAttemptedConversationRef.current !== conversationId) {
      restoreAttemptedRef.current = new Set();
      restoreAttemptedConversationRef.current = conversationId;
    }

    const { pending } = deepPlanUnavailableDocuments(deepPlan);
    if (pending.length === 0) return;

    const toRead = pending.filter(
      (phase) => !restoreAttemptedRef.current.has(phase),
    );

    for (const phase of toRead) {
      const path =
        latestDeepPlanFileEventsRef.current.get(phase)?.path ??
        (deepPlanWorkingDir
          ? buildPhasePlanPath(deepPlanWorkingDir, phase)
          : null);
      // The workspace dir is resolved with the conversation query; if it is not
      // in yet, leave the phase unattempted so the effect retries on the render
      // that carries it — a premature fail would strand the checkpoint.
      if (!path) continue;
      restoreAttemptedRef.current.add(phase);
      const fallbackId = targetIds[0];
      readConversationFile(
        {
          conversationId: resolvePhaseConversationId(phase, fallbackId),
          filePath: path,
        },
        {
          onSuccess: (fileContent) => setDeepPlanDocument(phase, fileContent),
          onError: (error) => {
            console.warn("Failed to restore deep-plan document:", error);
            failDeepPlanDocumentRestore(phase);
          },
        },
      );
    }
  }, [
    conversationId,
    conversationUrl,
    isLoadingHistoryPlanning,
    conversationMode,
    subConversationIds,
    deepPlan,
    deepPlanWorkingDir,
    readConversationFile,
    resolvePhaseConversationId,
    setDeepPlanDocument,
    failDeepPlanDocumentRestore,
  ]);

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
    // Failure state belongs to the previous conversation's sockets; a Retry in
    // the next conversation must not reconnect a socket that is already gone.
    failedSocketsRef.current.clear();
    // A cursor is a position in one conversation's log; carrying it into the
    // next would skip that conversation's events below it.
    mainCursorRef.current.clear();
    // Reset the tracked event ref when conversation changes
    latestPlanningFileEventRef.current = null;
    // Paths recorded for the previous conversation must not steer the next
    // conversation's document restore.
    latestDeepPlanFileEventsRef.current = new Map();
  }, [conversationId]);

  // Drop buffered deltas on conversation switch/unmount: the store is cleared on
  // switch, so flushing them would leak into the next conversation.
  useEffect(() => {
    const mainBatcher = mainDeltaBatcherRef.current;
    const planningBatcher = planningDeltaBatcherRef.current;
    return () => {
      mainBatcher?.reset();
      planningBatcher?.reset();
    };
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
        const event = routeSessionFrame(
          JSON.parse(messageEvent.data),
          mainDeltaBatcherRef.current,
          mainCursorRef.current.observe,
        );

        // History loading for the main conversation is REST-driven now;
        // every durable frame is a new event we add to the store.

        // Use type guard to validate v1 event structure
        if (isAgentServerEvent(event)) {
          // A reconnect replays the backlog from a stale anchor. The store
          // dedups by id, but the side-effects below aren't idempotent, so skip
          // them for replayed events (#1656).
          const isDuplicateEvent = useEventStore
            .getState()
            .eventIds.has(event.id ?? "");
          const switchLLMObservation = isSwitchLLMObservationEvent(event)
            ? event
            : null;
          const classifyAndSwitchLLMObservation =
            isClassifyAndSwitchLLMObservationEvent(event) ? event : null;
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
            const classification =
              "classification" in errorEvent ? errorEvent.classification : null;
            trackError({
              source: "conversation",
              metadata: {
                eventId: errorEvent.id,
                errorCode: errorEvent.code,
              },
              classification,
            });
            setErrorMessage(
              errorEvent.detail,
              "conversation",
              errorEvent.code,
              classification,
            );
          } else {
            handleNonErrorEvent();
          }

          // LLM errors render inline in the chat (see ErrorEventMessage); track
          // them for analytics but keep them out of the banner above the chat box.
          if (isAgentErrorEvent(event)) {
            trackError({
              source: "agent",
              metadata: {
                eventId: event.id,
                toolName: event.tool_name,
                toolCallId: event.tool_call_id,
              },
              classification: event.classification,
            });
          }

          // Clear optimistic user message when a user message is confirmed.
          // History and live delivery share the same timestamp/identity checks.
          if (isUserMessageEvent(event)) {
            if (conversationId) {
              consumeMatchingPendingMessage(
                conversationId,
                extractMessageEventText(event),
                event,
              );
              // Clear draft from localStorage - message was successfully delivered
              setConversationState(conversationId, { draftMessage: null });
            }
          }

          // Handle cache invalidation for ActionEvent. The main WebSocket only
          // opens when `conversationId` is present (see `wsUrl` below), so it
          // is always defined for events arriving over this socket.
          if (isActionEvent(event) && conversationId) {
            handleActionEventCacheInvalidation(
              event,
              conversationId,
              queryClient,
            );
          }

          // Handle conversation state updates
          if (isConversationStateUpdateEvent(event)) {
            if (
              isFullStateConversationStateUpdateEvent(event) &&
              conversationId
            ) {
              setExecutionStatus(conversationId, event.value.execution_status);
            }
            if (
              isAgentStatusConversationStateUpdateEvent(event) &&
              conversationId
            ) {
              setExecutionStatus(conversationId, event.value);
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
            // when several profiles share a model (#1082). Stamp with the
            // observation's own timestamp so a later history seed of this same
            // event can't roll it back (or needlessly rewrite it).
            stampActiveLlmProfile(
              conversationId,
              switchLLMObservation.observation.profile_name,
              switchLLMObservation.timestamp,
            );

            if (switchLLMObservation.observation.active_model) {
              updateConversationLlmModelInCache(
                queryClient,
                conversationId,
                switchLLMObservation.observation.active_model,
              );
            }

            invalidateConversationQueries(queryClient, conversationId);
          }

          // Router-driven model switch (Router/meta-profile classifier).
          // Same UI semantics as SwitchLLMObservation: update the combobox,
          // stamp the active profile, record the inline "Switched to"
          // message. Per the SDK wire contract, `model` is the saved LLM
          // profile name that was activated and `active_model` is the
          // underlying model string — so the profile stamp and inline
          // message use `model` (mirroring SwitchLLMObservation.profile_name),
          // while the combobox cache update uses `active_model`.
          if (
            conversationId &&
            classifyAndSwitchLLMObservation &&
            !classifyAndSwitchLLMObservation.observation.is_error &&
            classifyAndSwitchLLMObservation.observation.model
          ) {
            const profileName =
              classifyAndSwitchLLMObservation.observation.model;

            recordModelSwitchMessage(conversationId, profileName);

            stampActiveLlmProfile(
              conversationId,
              profileName,
              classifyAndSwitchLLMObservation.timestamp,
            );

            if (classifyAndSwitchLLMObservation.observation.active_model) {
              updateConversationLlmModelInCache(
                queryClient,
                conversationId,
                classifyAndSwitchLLMObservation.observation.active_model,
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

          // Same client-tool pattern, but the work is a network call: launch
          // the requested child conversation and post the outcome back so the
          // agent learns the id the server-side acknowledgement can't carry.
          if (conversationId && isLaunchChildConversationActionEvent(event)) {
            void handleLaunchChildConversationAction(
              event.action,
              conversationId,
              event.tool_call_id,
            );
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
        const event = routeSessionFrame(
          JSON.parse(messageEvent.data),
          planningDeltaBatcherRef.current,
          planningCursorRef.current.observe,
          { isFromPlanningAgent: true },
        );
        if (event === null) {
          return;
        }

        // Track received events for history loading. Only events count:
        // progress frames are not part of the replayed log.
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
            .eventIds.has(event.id ?? "");
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
            const classification =
              "classification" in errorEvent ? errorEvent.classification : null;
            trackError({
              source: "planning_conversation",
              metadata: {
                eventId: errorEvent.id,
                errorCode: errorEvent.code,
              },
              classification,
            });
            setErrorMessage(
              errorEvent.detail,
              "conversation",
              errorEvent.code,
              classification,
            );
          } else {
            handleNonErrorEvent();
          }

          // LLM errors render inline in the chat (see ErrorEventMessage); track
          // them for analytics but keep them out of the banner above the chat box.
          if (isAgentErrorEvent(event)) {
            trackError({
              source: "planning_agent",
              metadata: {
                eventId: event.id,
                toolName: event.tool_name,
                toolCallId: event.tool_call_id,
              },
              classification: event.classification,
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
                event,
              );
              setConversationState(conversationId, { draftMessage: null });
            }
          }

          // Handle cache invalidation for ActionEvent. The planning socket only
          // opens when the first sub-conversation has an id (see
          // `planningAgentWsUrl` below), so it is always defined here.
          if (isActionEvent(event)) {
            const planningAgentConversation = subConversations?.[0];
            if (planningAgentConversation?.id) {
              handleActionEventCacheInvalidation(
                event,
                planningAgentConversation.id,
                queryClient,
              );
            }
          }

          // Handle conversation state updates
          if (isConversationStateUpdateEvent(event)) {
            // Scope to the planning agent's own conversation id, not the main
            // `conversationId` — this socket reports the planning helper
            // conversation's run/idle transitions, which must never overwrite
            // the main conversation's status in the shared store.
            if (
              isFullStateConversationStateUpdateEvent(event) &&
              planningConversationId
            ) {
              setExecutionStatus(
                planningConversationId,
                event.value.execution_status,
              );
            }
            if (
              isAgentStatusConversationStateUpdateEvent(event) &&
              planningConversationId
            ) {
              setExecutionStatus(planningConversationId, event.value);
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

          // Handle PlanningFileEditorObservation - update the plan for Plan.md,
          // and the phase document for a deep-plan output file.
          if (isPlanningFileEditorObservationEvent(event)) {
            const { path } = event.observation;
            const planningAgentConversation = subConversations?.[0];
            const planningConversationId = planningAgentConversation?.id;
            const deepPlanPhase = matchDeepPlanDocumentFile(path);

            if (deepPlanPhase && planningConversationId && path) {
              // Read the phase's own document through the planner pinned to it,
              // not the first sub-conversation (Deep Planning has one planner
              // per phase).
              const readConversationId = resolvePhaseConversationId(
                deepPlanPhase,
                planningConversationId,
              );
              if (isLoadingHistoryPlanning) {
                // Only the newest write per phase matters.
                latestDeepPlanFileEventsRef.current.set(deepPlanPhase, {
                  path,
                  conversationId: readConversationId,
                });
              } else {
                readConversationFile(
                  { conversationId: readConversationId, filePath: path },
                  {
                    onSuccess: (fileContent) =>
                      setDeepPlanDocument(deepPlanPhase, fileContent),
                    onError: (error) => {
                      console.warn("Failed to read deep-plan document:", error);
                    },
                  },
                );
              }
            } else if (isPlanFilePath(path)) {
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
      planningConversationId,
      conversationId,
      setExecutionStatus,
      appendInput,
      appendOutput,
      readConversationFile,
      setPlanContent,
      setDeepPlanDocument,
      resolvePhaseConversationId,
      updateMetricsFromStats,
      handleNonErrorEvent,
    ],
  );

  // Separate WebSocket options for main connection
  const mainWebsocketOptions: WebSocketHookOptions = useMemo(() => {
    // `after_seq` replaces the legacy resend_mode/after_timestamp pair, which
    // compared naive local timestamps. Resolved lazily so a reconnect resumes
    // from the newest `seq` this socket actually saw rather than from a value
    // captured at render. The first connect has no cursor and asks for the
    // whole log (`-1`); the REST preload (`useConversationHistory`) still
    // renders instantly and the event store dedupes the overlap by id.
    const queryParams = () => {
      const cursor = mainCursorRef.current;
      cursor.start(cursor.value ?? historyAfterSeqRef.current ?? -1);
      return { after_seq: String(cursor.value) };
    };

    return {
      queryParams,
      sessionApiKey,
      reconnect: { enabled: true },
      onOpen: () => {
        setMainConnectionState("OPEN");
        hasConnectedRefMain.current = true; // Mark that we've successfully connected
        failedSocketsRef.current.delete("main"); // This socket recovered.
        clearConnectionError(); // Clear a previous connection error; keep sticky conversation errors
        // Progress frames are never replayed, so any slot left open across the
        // gap can never be retired. Discard and wait: the durable message is
        // coming on the cursor regardless.
        mainDeltaBatcherRef.current?.reset();
        useEventStore.getState().clearStreamingSlots();
      },
      onClose: () => {
        setMainConnectionState("CLOSED");
        mainDeltaBatcherRef.current?.reset();
        useEventStore.getState().clearStreamingSlots();
      },
      onError: () => {
        setMainConnectionState("CLOSED");
        // Only show error message if we've previously connected successfully
        if (hasConnectedRefMain.current) {
          failedSocketsRef.current.add("main");
          setErrorMessage(SERVER_CONNECTION_ERROR_MESSAGE, "connection");
        }
      },
      onMessage: handleMainMessage,
    };
  }, [handleMainMessage, setErrorMessage, clearConnectionError, sessionApiKey]);

  // Separate WebSocket options for planning agent connection
  const planningWebsocketOptions: WebSocketHookOptions = useMemo(() => {
    // The planner's history is not preloaded over REST, so it always replays
    // from the start on a first connect and from its cursor after that.
    const queryParams = () => {
      const cursor = planningCursorRef.current;
      cursor.start(cursor.value ?? -1);
      return { after_seq: String(cursor.value) };
    };

    const planningAgentConversation = subConversations?.[0];
    const planningApiKey =
      planningAgentConversation?.session_api_key ?? sessionApiKey;

    return {
      queryParams,
      sessionApiKey: planningApiKey,
      reconnect: { enabled: true },
      onOpen: async () => {
        setPlanningConnectionState("OPEN");
        hasConnectedRefPlanning.current = true; // Mark that we've successfully connected
        failedSocketsRef.current.delete("planning"); // This socket recovered.
        clearConnectionError(); // Clear a previous connection error; keep sticky conversation errors
        // See the main socket: an open slot cannot survive the gap.
        planningDeltaBatcherRef.current?.reset();
        useEventStore.getState().clearStreamingSlots(true);

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
        planningDeltaBatcherRef.current?.reset();
        useEventStore.getState().clearStreamingSlots(true);
      },
      onError: () => {
        setPlanningConnectionState("CLOSED");
        // Only show error message if we've previously connected successfully
        if (hasConnectedRefPlanning.current) {
          failedSocketsRef.current.add("planning");
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
    // Retry every socket that is currently down. Retrying only the one that
    // raised the banner last would leave the other closed when both failed,
    // which is exactly the state a single `last error source` cannot express.
    // The mode is the fallback for a banner raised before either socket
    // reported an error, so a Retry always acts on something.
    const failed = new Set(failedSocketsRef.current);
    if (failed.size === 0) {
      const source: ConnectionErrorSource = isPlanningMode(
        useConversationStore.getState().conversationMode,
        useConversationStore.getState().deepPlan.activePhase,
      )
        ? "planning"
        : "main";
      failed.add(source);
    }

    const retryPlanning = failed.has("planning") && planningAgentWsUrl;
    if (retryPlanning) {
      reconnectPlanning();
    }
    // Without a planner URL the fallback source cannot be retried, so fall back
    // to the main socket rather than making Retry a no-op.
    if (failed.has("main") || (failed.has("planning") && !retryPlanning)) {
      reconnectMain();
    }
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
      const currentPhase = useConversationStore.getState().deepPlan.activePhase;
      const routesToPlanner = isPlanningMode(currentMode, currentPhase);
      const currentSocket = routesToPlanner ? planningAgentSocket : mainSocket;
      const targetConversationId = routesToPlanner
        ? planningConversationId
        : conversationId;

      if (currentSocket?.readyState !== WebSocket.OPEN) {
        // WebSocket not connected - queue message via REST API
        // Message will be delivered automatically when conversation becomes ready
        if (!targetConversationId) {
          // Never fall back to the parent in plan mode: without a planner
          // target the message would run in the code agent, which is exactly
          // the boundary plan mode exists to enforce.
          const error = new Error(
            routesToPlanner
              ? "Planning conversation is not ready yet"
              : "No conversation ID available",
          );
          setErrorMessage(error.message);
          throw error;
        }

        try {
          await new ConversationClient(getAgentServerClientOptions()).sendEvent(
            targetConversationId,
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
    [
      mainSocket,
      planningAgentSocket,
      setErrorMessage,
      conversationId,
      planningConversationId,
    ],
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
    () => ({
      connectionState,
      mainConnectionState,
      sendMessage,
      isLoadingHistory,
      reconnect,
    }),
    [
      connectionState,
      mainConnectionState,
      sendMessage,
      isLoadingHistory,
      reconnect,
    ],
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
