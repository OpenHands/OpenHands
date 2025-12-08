import React from "react";
import { io, Socket } from "socket.io-client";
import { useQueryClient } from "@tanstack/react-query";
import { usePostHog } from "posthog-js/react";
import EventLogger from "#/utils/event-logger";
import { handleAssistantMessage } from "#/services/actions";
import { showChatError, trackError } from "#/utils/error-handler";
import { useRate } from "#/hooks/use-rate";
import { OpenHandsParsedEvent } from "#/types/core";
import {
  AssistantMessageAction,
  CommandAction,
  FileEditAction,
  FileWriteAction,
  UserMessageAction,
} from "#/types/core/actions";
import { Conversation } from "#/api/open-hands.types";
import { useUserProviders } from "#/hooks/use-user-providers";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import {
  isAgentStateChangeObservation,
  isErrorObservation,
  isOpenHandsAction,
  isOpenHandsObservation,
  isStatusUpdate,
  isUserMessage,
} from "#/types/core/guards";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { useEventStore } from "#/stores/use-event-store";

/**
 * @deprecated Use `V1_WebSocketConnectionState` from `conversation-websocket-context.tsx` instead.
 * This type is for legacy V0 conversations only.
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
export type V0_WebSocketStatus = "CONNECTING" | "CONNECTED" | "DISCONNECTED";

const hasValidMessageProperty = (obj: unknown): obj is { message: string } =>
  typeof obj === "object" &&
  obj !== null &&
  "message" in obj &&
  typeof obj.message === "string";

const isOpenHandsEvent = (event: unknown): event is OpenHandsParsedEvent =>
  typeof event === "object" &&
  event !== null &&
  "id" in event &&
  "source" in event &&
  "message" in event &&
  "timestamp" in event;

const isFileWriteAction = (
  event: OpenHandsParsedEvent,
): event is FileWriteAction => "action" in event && event.action === "write";

const isFileEditAction = (
  event: OpenHandsParsedEvent,
): event is FileEditAction => "action" in event && event.action === "edit";

const isCommandAction = (event: OpenHandsParsedEvent): event is CommandAction =>
  "action" in event && event.action === "run";

const isAssistantMessage = (
  event: OpenHandsParsedEvent,
): event is AssistantMessageAction =>
  "source" in event &&
  "type" in event &&
  event.source === "agent" &&
  event.type === "message";

const isMessageAction = (
  event: OpenHandsParsedEvent,
): event is UserMessageAction | AssistantMessageAction =>
  isUserMessage(event) || isAssistantMessage(event);

interface UseWsClient {
  webSocketStatus: V0_WebSocketStatus;
  isLoadingMessages: boolean;
  send: (event: Record<string, unknown>) => void;
}

const WsClientContext = React.createContext<UseWsClient>({
  webSocketStatus: "DISCONNECTED",
  isLoadingMessages: true,
  send: () => {
    throw new Error("not connected");
  },
});

interface WsClientProviderProps {
  conversationId: string;
}

interface ErrorArg {
  message?: string;
  data?: ErrorArgData | unknown;
}

interface ErrorArgData {
  msg_id: string;
}

export function updateStatusWhenErrorMessagePresent(
  data: ErrorArg | unknown,
  posthog?: ReturnType<typeof usePostHog>,
) {
  const isObject = (val: unknown): val is object =>
    !!val && typeof val === "object";
  const isString = (val: unknown): val is string => typeof val === "string";
  if (isObject(data) && "message" in data && isString(data.message)) {
    if (data.message === "websocket error" || data.message === "timeout") {
      return;
    }
    let msgId: string | undefined;
    let metadata: Record<string, unknown> = {};

    if ("data" in data && isObject(data.data)) {
      if ("msg_id" in data.data && isString(data.data.msg_id)) {
        msgId = data.data.msg_id;
      }
      metadata = data.data as Record<string, unknown>;
    }

    showChatError({
      message: data.message,
      source: "websocket",
      metadata,
      msgId,
      posthog,
    });
  }
}

export function WsClientProvider({
  conversationId,
  children,
}: React.PropsWithChildren<WsClientProviderProps>) {
  const posthog = usePostHog();
  const { setErrorMessage, removeErrorMessage } = useErrorMessageStore();
  const { removeOptimisticUserMessage } = useOptimisticUserMessageStore();
  const { addEvent, clearEvents } = useEventStore();
  const queryClient = useQueryClient();
  const sioRef = React.useRef<Socket | null>(null);
  const pendingEventsRef = React.useRef<Record<string, unknown>[]>([]);
  const [webSocketStatus, setWebSocketStatus] =
    React.useState<V0_WebSocketStatus>("DISCONNECTED");
  const lastEventRef = React.useRef<Record<string, unknown> | null>(null);
  const { providers } = useUserProviders();

  const messageRateHandler = useRate({ threshold: 250 });
  const { data: conversation, refetch: refetchConversation } =
    useActiveConversation();

  function flushPendingEvents(socket: Socket | null = sioRef.current) {
    if (!socket || pendingEventsRef.current.length === 0) {
      return;
    }

    pendingEventsRef.current.forEach((queuedEvent) => {
      socket.emit("oh_user_action", queuedEvent);
    });
    pendingEventsRef.current = [];
  }

  function send(event: Record<string, unknown>) {
    const socket = sioRef.current;

    if (!socket) {
      EventLogger.error("WebSocket is not connected, queuing message...");
      pendingEventsRef.current.push(event);
      return;
    }

    if (pendingEventsRef.current.length > 0) {
      flushPendingEvents(socket);
    }

    socket.emit("oh_user_action", event);
  }

  function handleConnect() {
    setWebSocketStatus("CONNECTED");
    removeErrorMessage();
    flushPendingEvents();
  }

  function handleMessage(event: Record<string, unknown>) {
    handleAssistantMessage(event);

    if (isOpenHandsEvent(event)) {
      const isStatusUpdateError =
        isStatusUpdate(event) && event.type === "error";

      const isAgentStateChangeError =
        isAgentStateChangeObservation(event) &&
        event.extras.agent_state === "error";

      if (isStatusUpdateError || isAgentStateChangeError) {
        const errorMessage = isStatusUpdate(event)
          ? event.message
          : event.extras.reason || "Unknown error";

        trackError({
          message: errorMessage,
          source: "chat",
          metadata: { msgId: event.id },
          posthog,
        });
        setErrorMessage(errorMessage);

        return;
      }

      if (isOpenHandsAction(event) || isOpenHandsObservation(event)) {
        addEvent(event); // Event is already OpenHandsParsedEvent
      }

      if (isErrorObservation(event)) {
        trackError({
          message: event.message,
          source: "chat",
          metadata: { msgId: event.id },
          posthog,
        });
      } else {
        removeErrorMessage();
      }

      if (isUserMessage(event)) {
        removeOptimisticUserMessage();
      }

      if (isMessageAction(event)) {
        messageRateHandler.record(new Date().getTime());
      }

      // Invalidate diffs cache when a file is edited or written
      if (
        isFileEditAction(event) ||
        isFileWriteAction(event) ||
        isCommandAction(event)
      ) {
        queryClient.invalidateQueries(
          {
            queryKey: ["file_changes", conversationId],
          },
          // Do not refetch if we are still receiving messages at a high rate (e.g., loading an existing conversation)
          // This prevents unnecessary refetches when the user is still receiving messages
          { cancelRefetch: false },
        );

        // Invalidate file diff cache when a file is edited or written
        if (!isCommandAction(event)) {
          const cachedConversaton = queryClient.getQueryData<Conversation>([
            "user",
            "conversation",
            conversationId,
          ]);
          const clonedRepositoryDirectory =
            cachedConversaton?.selected_repository?.split("/").pop();

          let fileToInvalidate = event.args.path.replace("/workspace/", "");
          if (clonedRepositoryDirectory) {
            fileToInvalidate = fileToInvalidate.replace(
              `${clonedRepositoryDirectory}/`,
              "",
            );
          }

          queryClient.invalidateQueries({
            queryKey: ["file_diff", conversationId, fileToInvalidate],
          });
        }
      }
    }

    if (!Number.isNaN(parseInt(event.id as string, 10))) {
      lastEventRef.current = event;
    }
  }

  function handleDisconnect(data: unknown) {
    setWebSocketStatus("DISCONNECTED");
    const sio = sioRef.current;
    if (!sio) {
      return;
    }
    sio.io.opts.query = sio.io.opts.query || {};
    sio.io.opts.query.latest_event_id = lastEventRef.current?.id;

    updateStatusWhenErrorMessagePresent(data, posthog);
    setErrorMessage(hasValidMessageProperty(data) ? data.message : "");
  }

  function handleError(data: unknown) {
    // set status
    setWebSocketStatus("DISCONNECTED");
    updateStatusWhenErrorMessagePresent(data, posthog);

    setErrorMessage(
      hasValidMessageProperty(data)
        ? data.message
        : "An unknown error occurred on the WebSocket connection.",
    );

    // check if something went wrong with the conversation.
    refetchConversation();
  }

  React.useEffect(() => {
    lastEventRef.current = null;

    clearEvents();
    setWebSocketStatus("CONNECTING");
    pendingEventsRef.current = [];
  }, [conversationId]);

  React.useEffect(() => {
    if (!conversationId) {
      throw new Error("No conversation ID provided");
    }

    // Clear error messages when conversation is intentionally stopped
    if (conversation && conversation.status === "STOPPED") {
      const existingSocket = sioRef.current;
      if (existingSocket) {
        existingSocket.disconnect();
      }
      sioRef.current = null;
      pendingEventsRef.current = [];
      removeErrorMessage();
      setWebSocketStatus("DISCONNECTED");
      return () => undefined; // conversation intentionally stopped
    }

    // Set connecting status when conversation is starting
    if (conversation && conversation.status === "STARTING") {
      removeErrorMessage();
      setWebSocketStatus("CONNECTING");
      return () => undefined; // conversation is starting, will connect when ready
    }

    // Only connect when conversation is fully loaded and running
    if (
      !conversation ||
      conversation.status !== "RUNNING" ||
      !conversation.runtime_status ||
      conversation.runtime_status === "STATUS$STOPPED"
    ) {
      if (sioRef.current) {
        sioRef.current.disconnect();
      }
      sioRef.current = null;
      return () => undefined; // conversation not ready for WebSocket connection
    }

    let sio = sioRef.current;

    if (sio?.connected) {
      sio.disconnect();
    }

    // Set initial status...
    setWebSocketStatus("CONNECTING");

    const lastEvent = lastEventRef.current;
    const query = {
      latest_event_id: lastEvent?.id ?? -1,
      conversation_id: conversationId,
      providers_set: providers,
      session_api_key: conversation.session_api_key, // Have to set here because socketio doesn't support custom headers. :(
    };

    let baseUrl: string | null = null;
    let socketPath: string;
    if (conversation.url && !conversation.url.startsWith("/")) {
      const u = new URL(conversation.url);
      baseUrl = u.host;
      const pathBeforeApi = u.pathname.split("/api/conversations")[0] || "/";
      // Socket.IO server default path is /socket.io; prefix with pathBeforeApi for path mode
      socketPath = `${pathBeforeApi.replace(/\/$/, "")}/socket.io`;
    } else {
      baseUrl =
        (import.meta.env.VITE_BACKEND_BASE_URL as string | undefined) ||
        window?.location.host;
      socketPath = "/socket.io";
    }

    sio = io(baseUrl, {
      transports: ["websocket"],
      path: socketPath,
      query,
    });

    sio.on("connect", handleConnect);
    sio.on("oh_event", handleMessage);
    sio.on("connect_error", handleError);
    sio.on("connect_failed", handleError);
    sio.on("disconnect", handleDisconnect);

    sioRef.current = sio;
    flushPendingEvents(sio);

    return () => {
      sio.off("connect", handleConnect);
      sio.off("oh_event", handleMessage);
      sio.off("connect_error", handleError);
      sio.off("connect_failed", handleError);
      sio.off("disconnect", handleDisconnect);
    };
  }, [
    conversationId,
    conversation?.url,
    conversation?.status,
    conversation?.runtime_status,
    providers,
  ]);

  React.useEffect(
    () => () => {
      const sio = sioRef.current;
      if (sio) {
        sio.off("disconnect", handleDisconnect);
        sio.disconnect();
      }
    },
    [],
  );

  const value = React.useMemo<UseWsClient>(
    () => ({
      webSocketStatus,
      isLoadingMessages: messageRateHandler.isUnderThreshold,
      send,
    }),
    [webSocketStatus, messageRateHandler.isUnderThreshold],
  );

  return <WsClientContext value={value}>{children}</WsClientContext>;
}

export function useWsClient() {
  const context = React.useContext(WsClientContext);
  return context;
}
