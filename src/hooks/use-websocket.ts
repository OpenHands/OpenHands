import React from "react";
import { sendWebSocketAuth } from "#/utils/websocket-auth";
import { startHandshakeWatchdog } from "#/utils/websocket-handshake";

export interface WebSocketHookOptions {
  /**
   * Query params for the socket URL. Pass a function when a value must be
   * read at connect time rather than at render time — a resume cursor that
   * advances on every frame would otherwise re-render the owner per event.
   */
  queryParams?:
    | Record<string, string | boolean>
    | (() => Record<string, string | boolean>);
  sessionApiKey?: string | null;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onMessage?: (event: MessageEvent) => void;
  onError?: (event: Event) => void;
  /** Refresh an expired session key before retrying an auth-rejected socket. */
  refreshSessionApiKey?: (
    rejectedSessionApiKey: string | null | undefined,
  ) => Promise<string | null | undefined>;
  reconnect?: {
    enabled?: boolean;
    maxAttempts?: number;
  };
}

// Reconnect backoff bounds: 1s, 2s, 4s, … capped at 30s.
const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_DELAY_MS = 30_000;

export const useWebSocket = (url: string, options?: WebSocketHookOptions) => {
  const [isConnected, setIsConnected] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);
  const [isReconnecting, setIsReconnecting] = React.useState(false);
  const wsRef = React.useRef<WebSocket | null>(null);
  const attemptCountRef = React.useRef(0);
  const reconnectTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = React.useRef(true); // Only set to false by disconnect()
  // Track which WebSocket instances are allowed to reconnect using a WeakSet
  const allowedToReconnectRef = React.useRef<WeakSet<WebSocket>>(new WeakSet());

  // Store options in a ref to avoid reconnecting when callbacks change
  const optionsRef = React.useRef(options);
  // A refreshed key is used immediately, even before React Query state has
  // propagated back through the provider props. Keep it until props catch up.
  const refreshedSessionApiKeyRef = React.useRef<string | undefined>(undefined);
  const sessionApiKeyBySocketRef = React.useRef(
    new WeakMap<WebSocket, string | null | undefined>(),
  );
  const previousSessionApiKeyRef = React.useRef(options?.sessionApiKey);
  React.useEffect(() => {
    optionsRef.current = options;
    if (options?.sessionApiKey !== previousSessionApiKeyRef.current) {
      previousSessionApiKeyRef.current = options?.sessionApiKey;
      refreshedSessionApiKeyRef.current = undefined;
    }
  }, [options]);

  const connectWebSocket = React.useCallback(() => {
    // Build URL with query parameters if provided
    let wsUrl = url;
    const { queryParams } = optionsRef.current ?? {};
    if (queryParams) {
      const resolved =
        typeof queryParams === "function" ? queryParams() : queryParams;
      const stringParams = Object.entries(resolved).reduce(
        (acc, [key, value]) => {
          acc[key] = String(value);
          return acc;
        },
        {} as Record<string, string>,
      );
      const params = new URLSearchParams(stringParams);
      wsUrl = `${url}?${params.toString()}`;
    }

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    // Mark this WebSocket instance as allowed to reconnect
    allowedToReconnectRef.current.add(ws);

    // Abort a socket stuck in CONNECTING so it can't hold Chrome's per-host
    // handshake lock indefinitely; its close flows into the reconnect path
    // below.
    const cancelHandshakeWatchdog = startHandshakeWatchdog(ws);

    ws.onopen = (event) => {
      cancelHandshakeWatchdog();
      const sessionApiKey =
        refreshedSessionApiKeyRef.current ?? optionsRef.current?.sessionApiKey;
      sessionApiKeyBySocketRef.current.set(ws, sessionApiKey);
      sendWebSocketAuth(ws, sessionApiKey);
      setIsConnected(true);
      setError(null); // Clear any previous errors
      setIsReconnecting(false);
      attemptCountRef.current = 0; // Reset attempt count on successful connection
      optionsRef.current?.onOpen?.(event);
    };

    ws.onmessage = (event) => {
      // Deliberately no `lastMessage` state here: nothing reads it, and a
      // React state write per frame re-renders this hook's owner on every
      // streamed token. Consumers subscribe via `onMessage`.
      optionsRef.current?.onMessage?.(event);
    };

    ws.onclose = (event) => {
      cancelHandshakeWatchdog();
      // Check if this specific WebSocket instance is allowed to reconnect
      const canReconnect = allowedToReconnectRef.current.has(ws);
      // A socket replaced by reconnect() or a URL change can close after its
      // replacement has opened; ignore it so it cannot touch the live state.
      const wasReplaced = wsRef.current !== null && wsRef.current !== ws;
      if (wasReplaced) {
        return;
      }

      setIsConnected(false);
      // If the connection closes with an error code, treat it as an error
      if (event.code !== 1000) {
        // 1000 is normal closure
        setError(
          new Error(
            `WebSocket closed with code ${event.code}: ${event.reason || "Connection closed unexpectedly"}`,
          ),
        );
        // Also call onError handler for error closures (only if allowed to reconnect)
        if (canReconnect) {
          optionsRef.current?.onError?.(event);
        }
      }
      optionsRef.current?.onClose?.(event);

      // Attempt reconnection if enabled and allowed. Authentication failures
      // on Cloud need a fresh runtime key before another socket can be opened.
      const reconnectEnabled = optionsRef.current?.reconnect?.enabled ?? false;
      const maxAttempts =
        optionsRef.current?.reconnect?.maxAttempts ?? Infinity;

      const scheduleReconnect = () => {
        // A manual disconnect/unmount can happen while a session-key refresh
        // is in flight, so re-check this socket's permission before scheduling.
        if (
          !shouldReconnectRef.current ||
          !allowedToReconnectRef.current.has(ws) ||
          attemptCountRef.current >= maxAttempts
        ) {
          setIsReconnecting(false);
          return;
        }

        setIsReconnecting(true);
        attemptCountRef.current += 1;

        // Exponential backoff with up to 30% random jitter so parallel
        // sockets (main + planning) don't retry in lockstep and hammer an
        // already-struggling server every few seconds forever.
        const baseDelay = Math.min(
          RECONNECT_BASE_DELAY_MS * 2 ** (attemptCountRef.current - 1),
          RECONNECT_MAX_DELAY_MS,
        );
        const delay = baseDelay + Math.random() * baseDelay * 0.3;

        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, delay);
      };

      if (
        reconnectEnabled &&
        canReconnect &&
        shouldReconnectRef.current &&
        attemptCountRef.current < maxAttempts
      ) {
        const refreshSessionApiKey =
          optionsRef.current?.refreshSessionApiKey;

        if (event.code === 4001 && refreshSessionApiKey) {
          // Do not enter the normal retry loop with a key the server has just
          // rejected. A missing, unchanged, or failed refresh is terminal for
          // this connection attempt; onError has already surfaced the failure.
          setIsReconnecting(true);
          const rejectedSessionApiKey = sessionApiKeyBySocketRef.current.has(ws)
            ? sessionApiKeyBySocketRef.current.get(ws)
            : refreshedSessionApiKeyRef.current ??
              optionsRef.current?.sessionApiKey;

          void Promise.resolve()
            .then(() => refreshSessionApiKey(rejectedSessionApiKey))
            .then((refreshedSessionApiKey) => {
              if (
                !shouldReconnectRef.current ||
                !allowedToReconnectRef.current.has(ws)
              ) {
                return;
              }

              if (
                !refreshedSessionApiKey ||
                refreshedSessionApiKey === rejectedSessionApiKey
              ) {
                setIsReconnecting(false);
                return;
              }

              refreshedSessionApiKeyRef.current = refreshedSessionApiKey;
              scheduleReconnect();
            })
            .catch(() => {
              if (
                shouldReconnectRef.current &&
                allowedToReconnectRef.current.has(ws)
              ) {
                setIsReconnecting(false);
              }
            });
        } else {
          scheduleReconnect();
        }
      } else {
        setIsReconnecting(false);
      }
    };

    ws.onerror = (event) => {
      // Ignore errors from sockets we've deliberately replaced or closed —
      // aborting a mid-handshake socket fires `error`, and it must not
      // surface as a connection failure for the replacement socket.
      if (!allowedToReconnectRef.current.has(ws)) {
        return;
      }
      setIsConnected(false);
      optionsRef.current?.onError?.(event);
    };
  }, [url]);

  React.useEffect(() => {
    // Reset shouldReconnect flag and attempt count when creating a new connection
    shouldReconnectRef.current = true;
    attemptCountRef.current = 0;

    // Only attempt connection if we have a valid URL
    if (url && url.trim() !== "") {
      connectWebSocket();
    }

    return () => {
      // Disable reconnection on unmount to prevent reconnection attempts
      // This must be set BEFORE closing the socket, so the onclose handler sees it
      shouldReconnectRef.current = false;
      // Clear any pending reconnection timeouts
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      // Close the WebSocket connection
      if (wsRef.current) {
        const { readyState } = wsRef.current;
        // Remove this WebSocket from the allowed list BEFORE closing
        // so its onclose handler won't try to reconnect
        allowedToReconnectRef.current.delete(wsRef.current);
        // Only close if not already closed/closing
        if (
          readyState === WebSocket.CONNECTING ||
          readyState === WebSocket.OPEN
        ) {
          wsRef.current.close();
        }
        wsRef.current = null;
      }
    };
  }, [url, connectWebSocket]);

  const sendMessage = React.useCallback(
    (data: string | Blob | BufferSource) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(data);
      }
    },
    [],
  );

  const disconnect = React.useCallback(() => {
    shouldReconnectRef.current = false;
    setIsReconnecting(false);
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      // Remove from allowed list before closing
      allowedToReconnectRef.current.delete(wsRef.current);
      wsRef.current.close();
    }
  }, []);

  const reconnect = React.useCallback(() => {
    shouldReconnectRef.current = true;
    attemptCountRef.current = 0;
    setIsReconnecting(true);
    setError(null);

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      const ws = wsRef.current;
      allowedToReconnectRef.current.delete(ws);
      wsRef.current = null;
      if (
        ws.readyState === WebSocket.CONNECTING ||
        ws.readyState === WebSocket.OPEN
      ) {
        ws.close();
      }
    }

    connectWebSocket();
  }, [connectWebSocket]);

  return {
    isConnected,
    error,
    socket: wsRef.current,
    sendMessage,
    isReconnecting,
    attemptCount: attemptCountRef.current,
    disconnect,
    reconnect,
  };
};
