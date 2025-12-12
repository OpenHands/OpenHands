import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useMemo,
  useRef,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useWebSocket, WebSocketHookOptions } from "#/hooks/use-websocket";
import { useEventStore } from "#/stores/use-event-store";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useV1ConversationStateStore } from "#/stores/v1-conversation-state-store";
import { useCommandStore } from "#/state/command-store";
import { useBrowserStore } from "#/stores/browser-store";
import {
  isV1Event,
  isAgentErrorEvent,
  isUserMessageEvent,
  isActionEvent,
  isConversationStateUpdateEvent,
  isFullStateConversationStateUpdateEvent,
  isAgentStatusConversationStateUpdateEvent,
  isStatsConversationStateUpdateEvent,
  isExecuteBashActionEvent,
  isExecuteBashObservationEvent,
  isConversationErrorEvent,
  isPlanningFileEditorObservationEvent,
  isBrowserObservationEvent,
  isBrowserNavigateActionEvent,
} from "#/types/v1/type-guards";
import { ConversationStateUpdateEventStats } from "#/types/v1/core/events/conversation-state-event";
import { handleActionEventCacheInvalidation } from "#/utils/cache-utils";
import { buildWebSocketUrl } from "#/utils/websocket-url";
import type {
  V1AppConversation,
  V1SendMessageRequest,
} from "#/api/conversation-service/v1-conversation-service.types";
import EventService from "#/api/event-service/event-service.api";
import { useConversationStore } from "#/state/conversation-store";
import { isBudgetOrCreditError } from "#/utils/error-handler";
import { useTracking } from "#/hooks/use-tracking";
import { useReadConversationFile } from "#/hooks/mutation/use-read-conversation-file";
import useMetricsStore from "#/stores/metrics-store";

// eslint-disable-next-line @typescript-eslint/naming-convention
export type V1_WebSocketConnectionState =
  | "CONNECTING"
  | "OPEN"
  | "CLOSED"
  | "CLOSING";

interface ConversationWebSocketContextType {
  connectionState: V1_WebSocketConnectionState;
  sendMessage: (message: V1SendMessageRequest) => Promise<void>;
  isLoadingHistory: boolean;
}

const ConversationWebSocketContext = createContext<
  ConversationWebSocketContextType | undefined
>(undefined);

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
  subConversations?: V1AppConversation[];
  subConversationIds?: string[];
}) {
  // Separate connection state tracking for each WebSocket
  const [mainConnectionState, setMainConnectionState] =
    useState<V1_WebSocketConnectionState>("CONNECTING");
  const [planningConnectionState, setPlanningConnectionState] =
    useState<V1_WebSocketConnectionState>("CONNECTING");

  // Track if we've ever successfully connected for each connection
  // Don't show errors until after first successful connection
  const hasConnectedRefMain = React.useRef(false);
  const hasConnectedRefPlanning = React.useRef(false);

  const queryClient = useQueryClient();
  const { addEvent } = useEventStore();
  const { setErrorMessage, removeErrorMessage } = useErrorMessageStore();
  const { removeOptimisticUserMessage } = useOptimisticUserMessageStore();
  const { setExecutionStatus } = useV1ConversationStateStore();
  const { appendInput, appendOutput } = useCommandStore();
  const { trackCreditLimitReached } = useTracking();

  // History loading state - separate per connection
  const [isLoadingHistoryMain, setIsLoadingHistoryMain] = useState(true);
  const [isLoadingHistoryPlanning, setIsLoadingHistoryPlanning] =
    useState(true);
  const [expectedEventCountMain, setExpectedEventCountMain] = useState<
    number | null
  >(null);
  const [expectedEventCountPlanning, setExpectedEventCountPlanning] = useState<
    number | null
  >(null);

  const { conversationMode, setPlanContent } = useConversationStore();

  // Hook for reading conversation file
  const { mutate: readConversationFile } = useReadConversationFile();

  // Separate received event count tracking per connection
  const receivedEventCountRefMain = useRef(0);
  const receivedEventCountRefPlanning = useRef(0);

  // Track the latest PlanningFileEditorObservation event during history replay
  // We'll only call the API once after history loading completes
  const latestPlanningFileEventRef = useRef<{
    path: string;
    conversationId: string;
  } | null>(null);

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

  // Build WebSocket URL from props
  // Only build URL if we have both conversationId and conversationUrl
  // This prevents connection attempts during task polling phase
  const wsUrl = useMemo(() => {
    // Don't attempt connection if we're missing required data
    if (!conversationId || !conversationUrl) {
      return null;
    }
    return buildWebSocketUrl(conversationId, conversationUrl);
  }, [conversationId, conversationUrl]);

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
  const connectionState = useMemo<V1_WebSocketConnectionState>(() => {
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
      expectedEventCountMain !== null &&
      receivedEventCountRefMain.current >= expectedEventCountMain &&
      isLoadingHistoryMain
    ) {
      setIsLoadingHistoryMain(false);
    }
  }, [expectedEventCountMain, isLoadingHistoryMain, receivedEventCountRefMain]);

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
            // eslint-disable-next-line no-console
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

  // Merged loading history state - true if either connection is still loading
  const isLoadingHistory = useMemo(
    () => isLoadingHistoryMain || isLoadingHistoryPlanning,
    [isLoadingHistoryMain, isLoadingHistoryPlanning],
  );

  // Reset hasConnected flags and history loading state when conversation changes
  useEffect(() => {
    hasConnectedRefPlanning.current = false;
    setIsLoadingHistoryMain(true);
    setExpectedEventCountMain(null);
    receivedEventCountRefMain.current = 0;
    // Reset the tracked event ref when conversation changes
    latestPlanningFileEventRef.current = null;
  }, [conversationId]);

  // Separate message handlers for each connection
  const handleMainMessage = useCallback(
    (messageEvent: MessageEvent) => {
      try {
        const event = JSON.parse(messageEvent.data);

        // Track received events for history loading (count ALL events from WebSocket)
        // Always count when loading, even if we don't have the expected count yet
        if (isLoadingHistoryMain) {
          receivedEventCountRefMain.current += 1;

          if (
            expectedEventCountMain !== null &&
            receivedEventCountRefMain.current >= expectedEventCountMain
          ) {
            setIsLoadingHistoryMain(false);
          }
        }

        // Use type guard to validate v1 event structure
        if (isV1Event(event)) {
          addEvent(event);

          // Handle ConversationErrorEvent specifically
          if (isConversationErrorEvent(event)) {
            setErrorMessage(event.detail);
          }

          // Handle AgentErrorEvent specifically
          if (isAgentErrorEvent(event)) {
            setErrorMessage(event.error);

            // Track credit limit reached if the error is budget-related
            if (isBudgetOrCreditError(event.error)) {
              trackCreditLimitReached({
                conversationId: conversationId || "unknown",
              });
            }
          }

          // Clear optimistic user message when a user message is confirmed
          if (isUserMessageEvent(event)) {
            removeOptimisticUserMessage();
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
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn("Failed to parse WebSocket message as JSON:", error);
      }
    },
    [
      addEvent,
      isLoadingHistoryMain,
      expectedEventCountMain,
      setErrorMessage,
      removeOptimisticUserMessage,
      queryClient,
      conversationId,
      setExecutionStatus,
      appendInput,
      appendOutput,
      updateMetricsFromStats,
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
        if (isV1Event(event)) {
          // Mark this event as coming from the planning agent
          const eventWithPlanningFlag = {
            ...event,
            isFromPlanningAgent: true,
          };
          addEvent(eventWithPlanningFlag);

          // Handle AgentErrorEvent specifically
          if (isAgentErrorEvent(event)) {
            setErrorMessage(event.error);
          }

          // Clear optimistic user message when a user message is confirmed
          if (isUserMessageEvent(event)) {
            removeOptimisticUserMessage();
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

          // Handle PlanningFileEditorObservation events - read and update plan content
          if (isPlanningFileEditorObservationEvent(event)) {
            const planningAgentConversation = subConversations?.[0];
            const planningConversationId = planningAgentConversation?.id;

            if (planningConversationId && event.observation.path) {
              // During history replay, track the latest event but don't call API
              // After history loading completes, we'll call the API once with the latest event
              if (isLoadingHistoryPlanning) {
                latestPlanningFileEventRef.current = {
                  path: event.observation.path,
                  conversationId: planningConversationId,
                };
              } else {
                // History loading is complete - this is a new real-time event
                // Call the API immediately for real-time updates
                readConversationFile(
                  {
                    conversationId: planningConversationId,
                    filePath: event.observation.path,
                  },
                  {
                    onSuccess: (fileContent) => {
                      setPlanContent(fileContent);
                    },
                    onError: (error) => {
                      // eslint-disable-next-line no-console
                      console.warn("Failed to read conversation file:", error);
                    },
                  },
                );
              }
            }
          }
        }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn("Failed to parse WebSocket message as JSON:", error);
      }
    },
    [
      addEvent,
      isLoadingHistoryPlanning,
      expectedEventCountPlanning,
      setErrorMessage,
      removeOptimisticUserMessage,
      queryClient,
      subConversations,
      setExecutionStatus,
      appendInput,
      appendOutput,
      readConversationFile,
      setPlanContent,
      updateMetricsFromStats,
    ],
  );

  // Separate WebSocket options for main connection
  const mainWebsocketOptions: WebSocketHookOptions = useMemo(() => {
    const queryParams: Record<string, string | boolean> = {
      resend_all: true,
    };

    // Add session_api_key if available
    if (sessionApiKey) {
      queryParams.session_api_key = sessionApiKey;
    }

    return {
      queryParams,
      reconnect: { enabled: true },
      onOpen: async () => {
        setMainConnectionState("OPEN");
        hasConnectedRefMain.current = true; // Mark that we've successfully connected
        removeErrorMessage(); // Clear any previous error messages on successful connection

        // Fetch expected event count for history loading detection
        if (conversationId && conversationUrl) {
          try {
            const count = await EventService.getEventCount(
              conversationId,
              conversationUrl,
              sessionApiKey,
            );
            setExpectedEventCountMain(count);

            // If no events expected, mark as loaded immediately
            if (count === 0) {
              setIsLoadingHistoryMain(false);
            }
          } catch (error) {
            // Fall back to marking as loaded to avoid infinite loading state
            setIsLoadingHistoryMain(false);
          }
        }
      },
      onClose: (event: CloseEvent) => {
        setMainConnectionState("CLOSED");
        // Only show error message if we've previously connected successfully
        // This prevents showing errors during initial connection attempts (e.g., when auto-starting a conversation)
        if (event.code !== 1000 && hasConnectedRefMain.current) {
          setErrorMessage(
            `Connection lost: ${event.reason || "Unexpected disconnect"}`,
          );
        }
      },
      onError: () => {
        setMainConnectionState("CLOSED");
        // Only show error message if we've previously connected successfully
        if (hasConnectedRefMain.current) {
          setErrorMessage("Failed to connect to server");
        }
      },
      onMessage: handleMainMessage,
    };
  }, [
    handleMainMessage,
    setErrorMessage,
    removeErrorMessage,
    sessionApiKey,
    conversationId,
    conversationUrl,
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
        removeErrorMessage(); // Clear any previous error messages on successful connection

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
      onClose: (event: CloseEvent) => {
        setPlanningConnectionState("CLOSED");
        // Only show error message if we've previously connected successfully
        // This prevents showing errors during initial connection attempts (e.g., when auto-starting a conversation)
        if (event.code !== 1000 && hasConnectedRefPlanning.current) {
          setErrorMessage(
            `Connection lost: ${event.reason || "Unexpected disconnect"}`,
          );
        }
      },
      onError: () => {
        setPlanningConnectionState("CLOSED");
        // Only show error message if we've previously connected successfully
        if (hasConnectedRefPlanning.current) {
          setErrorMessage("Failed to connect to server");
        }
      },
      onMessage: handlePlanningMessage,
    };
  }, [
    handlePlanningMessage,
    setErrorMessage,
    removeErrorMessage,
    sessionApiKey,
    subConversations,
  ]);

  // Only attempt WebSocket connection when we have a valid URL
  // This prevents connection attempts during task polling phase
  const websocketUrl = wsUrl;
  const { socket: mainSocket } = useWebSocket(
    websocketUrl || "",
    mainWebsocketOptions,
  );

  const { socket: planningAgentSocket } = useWebSocket(
    planningAgentWsUrl || "",
    planningWebsocketOptions,
  );

  const socket = useMemo(
    () => (conversationMode === "plan" ? planningAgentSocket : mainSocket),
    [conversationMode, planningAgentSocket, mainSocket],
  );

  // V1 send message function via WebSocket
  const sendMessage = useCallback(
    async (message: V1SendMessageRequest) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        const error = "WebSocket is not connected";
        setErrorMessage(error);
        throw new Error(error);
      }

      try {
        // Send message through WebSocket as JSON
        socket.send(JSON.stringify(message));
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Failed to send message";
        setErrorMessage(errorMessage);
        throw error;
      }
    },
    [socket, setErrorMessage],
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
    () => ({ connectionState, sendMessage, isLoadingHistory }),
    [connectionState, sendMessage, isLoadingHistory],
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
