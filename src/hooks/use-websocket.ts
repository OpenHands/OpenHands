import React from "react";
import {
  ConversationEventStream,
  type ConversationEventStreamOptions,
  type ConversationEventStreamState,
} from "@openhands/typescript-client/clients";

export type WebSocketHookOptions = Omit<
  ConversationEventStreamOptions,
  "url" | "onStateChange"
>;

const INITIAL_STATE: ConversationEventStreamState = {
  isConnected: false,
  isReconnecting: false,
  attemptCount: 0,
  error: null,
};

// React owns presentation state; the SDK owns the connection and wire protocol.
export const useWebSocket = (url: string, options?: WebSocketHookOptions) => {
  const [state, setState] = React.useState(INITIAL_STATE);
  const streamRef = React.useRef<ConversationEventStream | null>(null);
  const optionsRef = React.useRef(options);

  React.useEffect(() => {
    optionsRef.current = options;
    streamRef.current?.updateOptions({
      ...options,
      url,
      onStateChange: setState,
    });
  }, [options, url]);

  React.useEffect(() => {
    if (!url.trim()) {
      setState(INITIAL_STATE);
      return;
    }
    const stream = new ConversationEventStream({
      ...optionsRef.current,
      url,
      onStateChange: setState,
    });
    streamRef.current = stream;
    stream.start();
    return () => {
      stream.stop();
      streamRef.current = null;
    };
  }, [url]);

  const sendMessage = React.useCallback(
    (data: string | Blob | BufferSource) => {
      if (streamRef.current?.readyState === WebSocket.OPEN)
        streamRef.current.send(data);
    },
    [],
  );
  const disconnect = React.useCallback(() => streamRef.current?.stop(), []);
  const reconnect = React.useCallback(() => streamRef.current?.reconnect(), []);

  return {
    ...state,
    socket: streamRef.current,
    sendMessage,
    disconnect,
    reconnect,
  };
};
