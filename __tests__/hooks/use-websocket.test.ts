import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useWebSocket, type WebSocketHookOptions } from "#/hooks/use-websocket";

class BrowserWebSocketDouble {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: BrowserWebSocketDouble[] = [];

  readonly url: string;
  readyState = BrowserWebSocketDouble.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  readonly send = vi.fn();
  readonly close = vi.fn(() => {
    if (
      this.readyState === BrowserWebSocketDouble.CLOSING ||
      this.readyState === BrowserWebSocketDouble.CLOSED
    ) {
      return;
    }

    this.emitClose(1000, "Normal closure");
  });

  constructor(url: string) {
    this.url = url;
    BrowserWebSocketDouble.instances.push(this);
  }

  open() {
    this.readyState = BrowserWebSocketDouble.OPEN;
    this.onopen?.(new Event("open"));
  }

  receive(data: unknown) {
    this.onmessage?.(new MessageEvent("message", { data }));
  }

  emitError() {
    this.onerror?.(new Event("error"));
  }

  emitClose(code: number, reason = "") {
    this.readyState = BrowserWebSocketDouble.CLOSED;
    this.onclose?.(
      new CloseEvent("close", {
        code,
        reason,
        wasClean: code === 1000,
      }),
    );
  }

  static reset() {
    BrowserWebSocketDouble.instances = [];
  }
}

const renderWebSocket = <T = string>(
  url: string,
  options?: WebSocketHookOptions,
) => {
  vi.stubGlobal("WebSocket", BrowserWebSocketDouble);
  return renderHook(() => useWebSocket<T>(url, options));
};

const getSocket = (index = BrowserWebSocketDouble.instances.length - 1) => {
  const socket = BrowserWebSocketDouble.instances[index];
  if (!socket) {
    throw new Error(`Expected WebSocket instance at index ${index}`);
  }
  return socket;
};

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  BrowserWebSocketDouble.reset();
});

describe("useWebSocket connection lifecycle", () => {
  it("connects with serialized query parameters and reports an open connection", () => {
    const onOpen = vi.fn();
    const { result } = renderWebSocket("ws://acme.test/events", {
      queryParams: { token: "abc 123", includeDrafts: false },
      onOpen,
    });
    const socket = getSocket();

    expect(socket.url).toBe(
      "ws://acme.test/events?token=abc+123&includeDrafts=false",
    );
    expect(result.current.isConnected).toBe(false);
    expect(result.current.socket).toBe(null);

    act(() => socket.open());

    expect(result.current.isConnected).toBe(true);
    expect(result.current.socket).toBe(socket);
    expect(result.current.error).toBe(null);
    expect(result.current.isReconnecting).toBe(false);
    expect(result.current.attemptCount).toBe(0);
    expect(onOpen).toHaveBeenCalledWith(
      expect.objectContaining({ type: "open" }),
    );
  });

  it("does not connect until the URL becomes non-blank", () => {
    vi.stubGlobal("WebSocket", BrowserWebSocketDouble);
    const { result, rerender } = renderHook(
      ({ url }: { url: string }) => useWebSocket(url),
      { initialProps: { url: "" } },
    );

    expect(BrowserWebSocketDouble.instances).toHaveLength(0);
    expect(result.current.socket).toBe(null);

    rerender({ url: "   " });
    expect(BrowserWebSocketDouble.instances).toHaveLength(0);

    rerender({ url: "ws://acme.test/events" });
    expect(BrowserWebSocketDouble.instances).toHaveLength(1);
    expect(getSocket().url).toBe("ws://acme.test/events");
  });

  it("uses the latest callbacks without replacing the active connection", () => {
    vi.stubGlobal("WebSocket", BrowserWebSocketDouble);
    const originalOnMessage = vi.fn();
    const latestOnMessage = vi.fn();
    const { result, rerender } = renderHook(
      ({ onMessage }: { onMessage: (event: MessageEvent) => void }) =>
        useWebSocket<{ sequence: number }>("ws://acme.test/events", {
          onMessage,
        }),
      { initialProps: { onMessage: originalOnMessage } },
    );
    const socket = getSocket();

    act(() => socket.open());
    rerender({ onMessage: latestOnMessage });
    act(() => socket.receive({ sequence: 7 }));

    expect(BrowserWebSocketDouble.instances).toHaveLength(1);
    expect(originalOnMessage).not.toHaveBeenCalled();
    expect(latestOnMessage).toHaveBeenCalledWith(
      expect.objectContaining({ data: { sequence: 7 } }),
    );
    expect("messages" in result.current).toBe(false);
    // The hook intentionally does not expose per-message state; consumers
    // read messages through the onMessage callback only.
    expect("lastMessage" in result.current).toBe(false);
  });

  it("handles messages and native errors without optional callbacks", () => {
    const { result } = renderWebSocket<{ sequence: number }>(
      "ws://acme.test/events",
    );
    const socket = getSocket();

    act(() => socket.open());
    // Delivering a message without any onMessage callback must not throw or
    // disturb the connection state.
    act(() => socket.receive({ sequence: 8 }));
    expect(result.current.isConnected).toBe(true);

    act(() => socket.emitError());
    expect(result.current.isConnected).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it("handles messages and native errors when options omit callbacks", () => {
    const { result } = renderWebSocket<{ sequence: number }>(
      "ws://acme.test/events",
      {},
    );
    const socket = getSocket();

    act(() => socket.open());
    // An empty options object still omits onMessage; delivery must be a no-op
    // that leaves the connection intact.
    act(() => socket.receive({ sequence: 9 }));
    expect(result.current.isConnected).toBe(true);

    act(() => socket.emitError());
    expect(result.current.isConnected).toBe(false);
    expect(result.current.error).toBe(null);
  });

  it("reports native socket errors through the current error callback", () => {
    const onError = vi.fn();
    const { result } = renderWebSocket("ws://acme.test/events", { onError });
    const socket = getSocket();

    act(() => socket.open());
    expect(result.current.isConnected).toBe(true);

    act(() => socket.emitError());

    expect(result.current.isConnected).toBe(false);
    expect(result.current.error).toBe(null);
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ type: "error" }),
    );
  });

  it("reports a normal close without treating it as an error", () => {
    const onClose = vi.fn();
    const onError = vi.fn();
    const { result } = renderWebSocket("ws://acme.test/events", {
      onClose,
      onError,
    });
    const socket = getSocket();

    act(() => socket.open());
    act(() => socket.emitClose(1000, "Finished"));

    expect(result.current.isConnected).toBe(false);
    expect(result.current.error).toBe(null);
    expect(result.current.isReconnecting).toBe(false);
    expect(onClose).toHaveBeenCalledWith(
      expect.objectContaining({ code: 1000, reason: "Finished" }),
    );
    expect(onError).not.toHaveBeenCalled();
  });

  it("turns an unexpected close into an error for consumers", () => {
    const onClose = vi.fn();
    const onError = vi.fn();
    const { result } = renderWebSocket("ws://acme.test/events", {
      onClose,
      onError,
    });
    const socket = getSocket();

    act(() => socket.emitClose(4001, "Authentication expired"));

    expect(result.current.error).toEqual(
      new Error("WebSocket closed with code 4001: Authentication expired"),
    );
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ code: 4001 }),
    );
    expect(onClose).toHaveBeenCalledWith(
      expect.objectContaining({ code: 4001 }),
    );
  });

  it("provides a useful fallback when an unexpected close has no reason", () => {
    const { result } = renderWebSocket("ws://acme.test/events");

    act(() => getSocket().emitClose(1006));

    expect(result.current.error?.message).toBe(
      "WebSocket closed with code 1006: Connection closed unexpectedly",
    );
  });
});

describe("useWebSocket messaging", () => {
  it("sends data only while the connection is open", () => {
    const { result } = renderWebSocket("ws://acme.test/events");
    const socket = getSocket();
    const payload = new Blob(["payload"]);

    act(() => result.current.sendMessage("too early"));
    expect(socket.send).not.toHaveBeenCalled();

    act(() => socket.open());
    act(() => result.current.sendMessage(payload));
    expect(socket.send).toHaveBeenCalledWith(payload);

    act(() => socket.emitClose(1000));
    act(() => result.current.sendMessage("too late"));
    expect(socket.send).toHaveBeenCalledOnce();
  });

  it("ignores sends before a socket exists", () => {
    const { result } = renderWebSocket("");

    act(() => result.current.sendMessage("not connected"));

    expect(BrowserWebSocketDouble.instances).toHaveLength(0);
  });
});

describe("useWebSocket reconnection", () => {
  it("automatically retries after the reconnect delay and resets state on success", () => {
    vi.useFakeTimers();
    // Pin the jitter to zero so the first backoff is exactly the 1s base delay.
    vi.spyOn(Math, "random").mockReturnValue(0);
    const onError = vi.fn();
    const { result } = renderWebSocket("ws://acme.test/events", {
      reconnect: { enabled: true },
      onError,
    });
    const firstSocket = getSocket();

    act(() => firstSocket.emitClose(1011, "Server restarted"));

    expect(result.current.isReconnecting).toBe(true);
    expect(result.current.attemptCount).toBe(1);
    expect(result.current.error).toEqual(
      new Error("WebSocket closed with code 1011: Server restarted"),
    );

    act(() => vi.advanceTimersByTime(999));
    expect(BrowserWebSocketDouble.instances).toHaveLength(1);

    act(() => vi.advanceTimersByTime(1));
    expect(BrowserWebSocketDouble.instances).toHaveLength(2);

    const replacementSocket = getSocket();
    act(() => replacementSocket.open());

    expect(result.current.isConnected).toBe(true);
    expect(result.current.error).toBe(null);
    expect(result.current.isReconnecting).toBe(false);
    expect(result.current.attemptCount).toBe(0);
    expect(onError).toHaveBeenCalledOnce();
  });

  it("stops retrying after the configured maximum number of attempts", () => {
    vi.useFakeTimers();
    const { result } = renderWebSocket("ws://acme.test/events", {
      reconnect: { enabled: true, maxAttempts: 1 },
    });

    act(() => getSocket().emitClose(1000));
    expect(result.current.attemptCount).toBe(1);

    act(() => vi.advanceTimersByTime(3000));
    expect(BrowserWebSocketDouble.instances).toHaveLength(2);

    act(() => getSocket().emitClose(1000));
    expect(result.current.isReconnecting).toBe(false);
    expect(result.current.attemptCount).toBe(1);

    act(() => vi.advanceTimersByTime(3000));
    expect(BrowserWebSocketDouble.instances).toHaveLength(2);
  });

  it("disconnects an active connection without scheduling a retry", () => {
    vi.useFakeTimers();
    const { result } = renderWebSocket("ws://acme.test/events", {
      reconnect: { enabled: true, maxAttempts: 3 },
    });
    const socket = getSocket();

    act(() => socket.open());
    act(() => result.current.disconnect());

    expect(socket.close).toHaveBeenCalledOnce();
    expect(result.current.isConnected).toBe(false);
    expect(result.current.isReconnecting).toBe(false);

    act(() => vi.advanceTimersByTime(3000));
    expect(BrowserWebSocketDouble.instances).toHaveLength(1);
  });

  it("does not report a disconnected socket's late close as a reconnectable error", () => {
    vi.useFakeTimers();
    const onError = vi.fn();
    const { result } = renderWebSocket("ws://acme.test/events", {
      reconnect: { enabled: true },
      onError,
    });
    const socket = getSocket();

    act(() => socket.open());
    act(() => result.current.disconnect());
    act(() => socket.emitClose(1006, "Late network failure"));

    expect(result.current.error?.message).toBe(
      "WebSocket closed with code 1006: Late network failure",
    );
    expect(onError).not.toHaveBeenCalled();
    expect(result.current.isReconnecting).toBe(false);

    act(() => vi.advanceTimersByTime(3000));
    expect(BrowserWebSocketDouble.instances).toHaveLength(1);
  });

  it("cancels a pending automatic retry when disconnected", () => {
    vi.useFakeTimers();
    const { result } = renderWebSocket("ws://acme.test/events", {
      reconnect: { enabled: true },
    });

    act(() => getSocket().emitClose(1000));
    expect(result.current.isReconnecting).toBe(true);

    act(() => result.current.disconnect());
    expect(result.current.isReconnecting).toBe(false);

    act(() => vi.advanceTimersByTime(3000));
    expect(BrowserWebSocketDouble.instances).toHaveLength(1);
  });

  it("manually replaces an active connection immediately", () => {
    const { result } = renderWebSocket("ws://acme.test/events");
    const firstSocket = getSocket();

    act(() => firstSocket.open());
    act(() => result.current.reconnect());

    expect(firstSocket.close).toHaveBeenCalledOnce();
    expect(BrowserWebSocketDouble.instances).toHaveLength(2);
    expect(result.current.isConnected).toBe(false);
    expect(result.current.isReconnecting).toBe(false);
    expect(result.current.error).toBe(null);
    expect(result.current.attemptCount).toBe(0);
  });

  it("manual reconnect supersedes a pending retry without creating a duplicate", () => {
    vi.useFakeTimers();
    const { result } = renderWebSocket("ws://acme.test/events", {
      reconnect: { enabled: true },
    });
    const firstSocket = getSocket();

    act(() => firstSocket.emitClose(1012, "Restarting"));
    expect(result.current.error).not.toBe(null);

    act(() => result.current.reconnect());
    expect(BrowserWebSocketDouble.instances).toHaveLength(2);
    expect(firstSocket.close).not.toHaveBeenCalled();
    expect(result.current.error).toBe(null);
    expect(result.current.attemptCount).toBe(0);

    act(() => vi.advanceTimersByTime(3000));
    expect(BrowserWebSocketDouble.instances).toHaveLength(2);
  });

  it("keeps automatic retries enabled after a manual reconnect", () => {
    vi.useFakeTimers();
    const { result } = renderWebSocket("ws://acme.test/events", {
      reconnect: { enabled: true, maxAttempts: 2 },
    });
    const firstSocket = getSocket();

    act(() => result.current.reconnect());
    expect(firstSocket.close).toHaveBeenCalledOnce();
    expect(BrowserWebSocketDouble.instances).toHaveLength(2);

    act(() => getSocket().emitClose(1000));
    expect(result.current.isReconnecting).toBe(true);
    expect(result.current.attemptCount).toBe(1);

    act(() => vi.advanceTimersByTime(3000));
    expect(BrowserWebSocketDouble.instances).toHaveLength(3);
  });

  it("manually reconnects with the latest URL", () => {
    vi.stubGlobal("WebSocket", BrowserWebSocketDouble);
    const { result, rerender } = renderHook(
      ({ url }: { url: string }) => useWebSocket(url),
      { initialProps: { url: "ws://acme.test/first" } },
    );

    rerender({ url: "ws://acme.test/second" });
    const secondSocket = getSocket();
    expect(secondSocket.url).toBe("ws://acme.test/second");

    act(() => result.current.reconnect());

    expect(secondSocket.close).toHaveBeenCalledOnce();
    expect(BrowserWebSocketDouble.instances).toHaveLength(3);
    expect(getSocket().url).toBe("ws://acme.test/second");
  });

  it("can be explicitly connected after starting without a URL", () => {
    const { result } = renderWebSocket("");

    act(() => result.current.disconnect());
    expect(BrowserWebSocketDouble.instances).toHaveLength(0);
    expect(result.current.isReconnecting).toBe(false);

    act(() => result.current.reconnect());
    expect(BrowserWebSocketDouble.instances).toHaveLength(1);
    expect(getSocket().url).toBe("");
    expect(result.current.isReconnecting).toBe(true);
  });
});

describe("useWebSocket cleanup", () => {
  it("closes a connecting socket on unmount", () => {
    const { unmount } = renderWebSocket("ws://acme.test/events");
    const socket = getSocket();

    unmount();

    expect(socket.close).toHaveBeenCalledOnce();
  });

  it("closes an open socket when the URL changes", () => {
    vi.stubGlobal("WebSocket", BrowserWebSocketDouble);
    const { rerender } = renderHook(
      ({ url }: { url: string }) => useWebSocket(url),
      { initialProps: { url: "ws://acme.test/first" } },
    );
    const firstSocket = getSocket();
    act(() => firstSocket.open());

    rerender({ url: "ws://acme.test/second" });

    expect(firstSocket.close).toHaveBeenCalledOnce();
    expect(BrowserWebSocketDouble.instances).toHaveLength(2);
    expect(getSocket().url).toBe("ws://acme.test/second");
  });

  it("does not close a socket that has already closed", () => {
    const { unmount } = renderWebSocket("ws://acme.test/events");
    const socket = getSocket();
    act(() => socket.emitClose(1000));

    unmount();

    expect(socket.close).not.toHaveBeenCalled();
  });

  it("cancels a pending retry on unmount", () => {
    vi.useFakeTimers();
    const { unmount } = renderWebSocket("ws://acme.test/events", {
      reconnect: { enabled: true },
    });
    act(() => getSocket().emitClose(1000));

    unmount();
    act(() => vi.advanceTimersByTime(3000));

    expect(BrowserWebSocketDouble.instances).toHaveLength(1);
  });
});
