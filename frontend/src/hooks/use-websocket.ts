import React from "react";

export interface WebSocketHookOptions {
  queryParams?: Record<string, string | boolean>;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onMessage?: (event: MessageEvent) => void;
  onError?: (event: Event) => void;
  reconnect?: {
    enabled?: boolean;
    maxAttempts?: number;
  };
}

export const useWebSocket = <T = string>(
  url: string,
  options?: WebSocketHookOptions,
) => {
  const [isConnected, setIsConnected] = React.useState(false);
  const [lastMessage, setLastMessage] = React.useState<T | null>(null);
  const [messages, setMessages] = React.useState<T[]>([]);
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
  React.useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const connectWebSocket = React.useCallback(() => {
    // Build URL with query parameters if provided
    let wsUrl = url;
    if (optionsRef.current?.queryParams) {
      const stringParams = Object.entries(
        optionsRef.current.queryParams,
      ).reduce(
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

    ws.onopen = (event) => {
      setIsConnected(true);
      setError(null); // Clear any previous errors
      setIsReconnecting(false);
      attemptCountRef.current = 0; // Reset attempt count on successful connection
      optionsRef.current?.onOpen?.(event);
    };

    ws.onmessage = (event) => {
      setLastMessage(event.data);
      setMessages((prev) => [...prev, event.data]);
      optionsRef.current?.onMessage?.(event);
    };

    ws.onclose = (event) => {
      // Check if this specific WebSocket instance is allowed to reconnect
      const canReconnect = allowedToReconnectRef.current.has(ws);
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

      // Attempt reconnection if enabled and allowed
      // IMPORTANT: Only reconnect if this specific instance is allowed to reconnect
      const reconnectEnabled = optionsRef.current?.reconnect?.enabled ?? false;
      const maxAttempts =
        optionsRef.current?.reconnect?.maxAttempts ?? Infinity;

      if (
        reconnectEnabled &&
        canReconnect &&
        shouldReconnectRef.current &&
        attemptCountRef.current < maxAttempts
      ) {
        setIsReconnecting(true);
        attemptCountRef.current += 1;

        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000); // 3 second delay
      } else {
        setIsReconnecting(false);
      }
    };

    ws.onerror = (event) => {
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
    (data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
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

  return {
    isConnected,
    lastMessage,
    messages,
    error,
    socket: wsRef.current,
    sendMessage,
    isReconnecting,
    attemptCount: attemptCountRef.current,
    disconnect,
  };
};
