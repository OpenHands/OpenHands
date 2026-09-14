/**
 * TODO: Fix flaky WebSocket tests (https://github.com/OpenHands/OpenHands/issues/11944)
 *
 * Several tests in this file are skipped because they fail intermittently in CI
 * but pass locally. The SUSPECTED root cause is that `wsLink.broadcast()` sends messages
 * to ALL connected clients across all tests, causing cross-test contamination
 * when tests run in parallel with Vitest v4.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  afterEach,
  vi,
} from "vitest";
import { ws } from "msw";
import { setupServer } from "msw/node";
import { useWebSocket } from "#/hooks/use-websocket";

describe("useWebSocket", () => {
  // MSW WebSocket mock setup
  const wsLink = ws.link("ws://acme.com/ws");

  const mswServer = setupServer(
    wsLink.addEventListener("connection", ({ client, server }) => {
      // Establish the connection
      server.connect();

      // Send a welcome message to confirm connection
      client.send("Welcome to the WebSocket!");
    }),
  );

  beforeAll(() =>
    mswServer.listen({
      onUnhandledRequest: "warn",
    }),
  );
  afterEach(() => mswServer.resetHandlers());
  afterAll(() => mswServer.close());

  const waitForConnection = async (result: {
    current: {
      isConnected: boolean;
    };
  }) => {
    await waitFor(
      () => {
        expect(result.current.isConnected).toBe(true);
      },
      { timeout: 5000 },
    );
  };

  it("should establish a WebSocket connection", async () => {
    const messages: string[] = [];
    const { result } = renderHook(() =>
      useWebSocket("ws://acme.com/ws", {
        onMessage: (event) => messages.push(event.data),
      }),
    );

    // Initially should not be connected
    expect(result.current.isConnected).toBe(false);

    // Wait for connection to be established
    await waitForConnection(result);

    // Should deliver the welcome message from our mock via onMessage
    await waitFor(() => {
      expect(messages).toContain("Welcome to the WebSocket!");
    });

    // Confirm that the WebSocket connection is established when the hook is used
    expect(result.current.socket).toBeTruthy();
  });

  it("should not retain an unbounded raw message history", async () => {
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;
      static instance: MockWebSocket | null = null;

      readonly url: string;
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockWebSocket.instance = this;
        queueMicrotask(() => {
          this.readyState = MockWebSocket.OPEN;
          this.onopen?.(new Event("open"));
        });
      }

      send() {}

      close() {
        this.readyState = MockWebSocket.CLOSED;
        this.onclose?.(
          new CloseEvent("close", {
            code: 1000,
            reason: "Normal closure",
            wasClean: true,
          }),
        );
      }
    }

    const originalWebSocket = globalThis.WebSocket;
    vi.stubGlobal("WebSocket", MockWebSocket);

    try {
      const messages: string[] = [];
      const { result, unmount } = renderHook(() =>
        useWebSocket("ws://acme.com/ws", {
          onMessage: (event) => messages.push(event.data),
        }),
      );

      await waitForConnection(result);

      act(() => {
        MockWebSocket.instance?.onmessage?.(
          new MessageEvent("message", { data: "first" }),
        );
        MockWebSocket.instance?.onmessage?.(
          new MessageEvent("message", { data: "second" }),
        );
        MockWebSocket.instance?.onmessage?.(
          new MessageEvent("message", { data: "third" }),
        );
      });

      // Every frame is delivered via onMessage, but the hook retains no raw
      // message history of its own — not even the latest.
      expect(messages).toEqual(["first", "second", "third"]);
      expect("lastMessage" in result.current).toBe(false);
      expect("messages" in result.current).toBe(false);

      unmount();
    } finally {
      globalThis.WebSocket = originalWebSocket;
      MockWebSocket.instance = null;
    }
  });

  it("should handle connection errors gracefully", async () => {
    // Create a mock that will simulate an error
    const errorLink = ws.link("ws://error-test.com/ws");
    mswServer.use(
      errorLink.addEventListener("connection", ({ client }) => {
        // Simulate an error by closing the connection immediately
        client.close(1006, "Connection failed");
      }),
    );

    const { result } = renderHook(() => useWebSocket("ws://error-test.com/ws"));

    // Initially should not be connected and no error
    expect(result.current.isConnected).toBe(false);
    expect(result.current.error).toBe(null);

    // Wait for the connection to fail
    await waitFor(() => {
      expect(result.current.isConnected).toBe(false);
    });

    // Should have error information (the close event should trigger error state)
    await waitFor(() => {
      expect(result.current.error).not.toBe(null);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    // Should have meaningful error message (could be from onerror or onclose)
    expect(
      result.current.error?.message.includes("WebSocket closed with code 1006"),
    ).toBe(true);

    // Should not crash the application
    expect(result.current.socket).toBeTruthy();
  });

  it.skip("should close the WebSocket connection on unmount", async () => {
    const { result, unmount } = renderHook(() =>
      useWebSocket("ws://acme.com/ws"),
    );

    // Wait for connection to be established
    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    // Verify connection is active
    expect(result.current.isConnected).toBe(true);
    expect(result.current.socket).toBeTruthy();

    const closeSpy = vi.spyOn(result.current.socket!, "close");

    // Unmount the component (this should trigger the useEffect cleanup)
    unmount();

    // Verify that WebSocket close was called during cleanup
    expect(closeSpy).toHaveBeenCalledOnce();
  });

  it("should support query parameters in WebSocket URL", async () => {
    // Stub WebSocket deterministically (mirrors the `onClose` test below).
    // The MSW-backed variant was flaky in CI — `wsLink.broadcast()` from
    // other tests leaks across the shared mock server, and this assertion
    // only needs to observe the constructed URL, not a real connection.
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;

      readonly url: string;
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        queueMicrotask(() => {
          this.readyState = MockWebSocket.OPEN;
          this.onopen?.(new Event("open"));
        });
      }

      send() {}

      close() {
        this.readyState = MockWebSocket.CLOSED;
        this.onclose?.(
          new CloseEvent("close", {
            code: 1000,
            reason: "Normal closure",
            wasClean: true,
          }),
        );
      }
    }

    const originalWebSocket = globalThis.WebSocket;
    vi.stubGlobal("WebSocket", MockWebSocket);

    try {
      const baseUrl = "ws://acme.com/ws";
      const queryParams = {
        token: "abc123",
        userId: "user456",
        version: "v1",
      };

      const { result, unmount } = renderHook(() =>
        useWebSocket(baseUrl, { queryParams }),
      );

      await waitForConnection(result);

      // Verify that the WebSocket was created with query parameters
      expect(result.current.socket).toBeTruthy();
      expect(result.current.socket!.url).toBe(
        "ws://acme.com/ws?token=abc123&userId=user456&version=v1",
      );

      unmount();
    } finally {
      globalThis.WebSocket = originalWebSocket;
    }
  });

  it("should send the session key before application messages on every connection", async () => {
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;
      static instance: MockWebSocket | null = null;
      static readonly instances: MockWebSocket[] = [];

      readonly url: string;
      readonly sent: string[] = [];
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockWebSocket.instance = this;
        MockWebSocket.instances.push(this);
        queueMicrotask(() => {
          this.readyState = MockWebSocket.OPEN;
          this.onopen?.(new Event("open"));
        });
      }

      send(data: string) {
        if (this.readyState !== MockWebSocket.OPEN) {
          throw new DOMException("WebSocket is not open", "InvalidStateError");
        }
        this.sent.push(data);
      }

      close() {
        this.readyState = MockWebSocket.CLOSED;
      }
    }

    const originalWebSocket = globalThis.WebSocket;
    vi.stubGlobal("WebSocket", MockWebSocket);
    const sessionApiKey = `sk-oh-${"a".repeat(64)}`;
    const expectedFrames = [
      JSON.stringify({ type: "auth", session_api_key: sessionApiKey }),
      "application-message",
    ];

    try {
      const { result, unmount } = renderHook(() =>
        useWebSocket("ws://acme.com/ws", {
          sessionApiKey,
          onOpen: () => MockWebSocket.instance?.send("application-message"),
        }),
      );
      const firstSocket = MockWebSocket.instance!;

      expect(firstSocket.readyState).toBe(MockWebSocket.CONNECTING);
      expect(firstSocket.sent).toEqual([]);
      await waitForConnection(result);

      expect(firstSocket.url).toBe("ws://acme.com/ws");
      expect(firstSocket.sent).toEqual(expectedFrames);

      act(() => {
        result.current.reconnect();
      });

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(2);
        expect(MockWebSocket.instances[1].sent).toEqual(expectedFrames);
      });

      unmount();
    } finally {
      globalThis.WebSocket = originalWebSocket;
      MockWebSocket.instance = null;
      MockWebSocket.instances.length = 0;
    }
  });

  // Skipped: flaky in CI - see comment at top of file
  it.skip("should call onOpen handler when WebSocket connection opens", async () => {
    const onOpenSpy = vi.fn();
    const options = { onOpen: onOpenSpy };

    const { result } = renderHook(() =>
      useWebSocket("ws://acme.com/ws", options),
    );

    // Initially should not be connected
    expect(result.current.isConnected).toBe(false);
    expect(onOpenSpy).not.toHaveBeenCalled();

    // Wait for connection to be established
    await waitForConnection(result);

    // onOpen handler should have been called
    expect(onOpenSpy).toHaveBeenCalledOnce();
  });

  it("should call onClose handler when WebSocket connection closes", async () => {
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;

      readonly url: string;
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        queueMicrotask(() => {
          this.readyState = MockWebSocket.OPEN;
          this.onopen?.(new Event("open"));
        });
      }

      send() {}

      close() {
        this.readyState = MockWebSocket.CLOSED;
        this.onclose?.(
          new CloseEvent("close", {
            code: 1000,
            reason: "Normal closure",
            wasClean: true,
          }),
        );
      }
    }

    const originalWebSocket = globalThis.WebSocket;
    vi.stubGlobal("WebSocket", MockWebSocket);

    const onCloseSpy = vi.fn();
    const options = { onClose: onCloseSpy };

    try {
      const { result, unmount } = renderHook(() =>
        useWebSocket("ws://acme.com/ws", options),
      );

      await waitForConnection(result);

      act(() => {
        result.current.disconnect();
      });

      await waitFor(() => {
        expect(onCloseSpy).toHaveBeenCalledOnce();
      });

      unmount();
    } finally {
      globalThis.WebSocket = originalWebSocket;
    }
  });

  it.skip("should call onMessage handler when WebSocket receives a message", async () => {
    const onMessageSpy = vi.fn();
    const options = { onMessage: onMessageSpy };

    const { result } = renderHook(() =>
      useWebSocket("ws://acme.com/ws", options),
    );

    // Wait for connection to be established
    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    // onMessage handler should have been called for the welcome message
    await waitFor(() => {
      expect(onMessageSpy).toHaveBeenCalledOnce();
    });

    // Send another message from the mock server
    wsLink.broadcast("Hello from server!");

    // onMessage handler should have been called twice now
    await waitFor(() => {
      expect(onMessageSpy).toHaveBeenCalledTimes(2);
    });
  });

  it("should call onError handler when WebSocket encounters an error", async () => {
    const onErrorSpy = vi.fn();
    const options = { onError: onErrorSpy };

    // Create a mock that will simulate an error
    const errorLink = ws.link("ws://error-test.com/ws");
    mswServer.use(
      errorLink.addEventListener("connection", ({ client }) => {
        // Simulate an error by closing the connection immediately
        client.close(1006, "Connection failed");
      }),
    );

    const { result } = renderHook(() =>
      useWebSocket("ws://error-test.com/ws", options),
    );

    // Initially should not be connected and no error
    expect(result.current.isConnected).toBe(false);
    expect(onErrorSpy).not.toHaveBeenCalled();

    // Wait for the connection to fail
    await waitFor(() => {
      expect(result.current.isConnected).toBe(false);
    });

    // Should have error information
    await waitFor(() => {
      expect(result.current.error).not.toBe(null);
    });

    // onError handler should have been called
    expect(onErrorSpy).toHaveBeenCalled();
  });

  it.skip("should provide sendMessage function to send messages to WebSocket", async () => {
    const { result } = renderHook(() => useWebSocket("ws://acme.com/ws"));

    // Wait for connection to be established
    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    // Should have a sendMessage function
    expect(result.current.sendMessage).toBeDefined();
    expect(typeof result.current.sendMessage).toBe("function");

    // Mock the WebSocket send method
    const sendSpy = vi.spyOn(result.current.socket!, "send");

    // Send a message
    result.current.sendMessage("Hello WebSocket!");

    // Verify that WebSocket.send was called with the correct message
    expect(sendSpy).toHaveBeenCalledOnce();
    expect(sendSpy).toHaveBeenCalledWith("Hello WebSocket!");
  });

  it("closes a handshake stuck in CONNECTING at the timeout and retries", async () => {
    // Arrange: a socket whose handshake never completes. Closing it while
    // CONNECTING fires error + close(1006), as browsers do.
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;
      static readonly instances: MockWebSocket[] = [];

      readonly url: string;
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockWebSocket.instances.push(this);
      }

      send() {}

      close() {
        this.readyState = MockWebSocket.CLOSED;
        this.onerror?.(new Event("error"));
        this.onclose?.(
          new CloseEvent("close", { code: 1006, reason: "", wasClean: false }),
        );
      }
    }

    const originalWebSocket = globalThis.WebSocket;
    vi.stubGlobal("WebSocket", MockWebSocket);
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0); // deterministic backoff

    try {
      const { unmount } = renderHook(() =>
        useWebSocket("ws://acme.com/ws", { reconnect: { enabled: true } }),
      );
      const firstSocket = MockWebSocket.instances[0];

      // Act/Assert: just before the timeout the handshake is still pending.
      await act(async () => {
        vi.advanceTimersByTime(9_999);
      });
      expect(firstSocket.readyState).toBe(MockWebSocket.CONNECTING);
      expect(MockWebSocket.instances).toHaveLength(1);

      // At the timeout the stuck socket is closed (releasing the browser's
      // per-host handshake lock)...
      await act(async () => {
        vi.advanceTimersByTime(1);
      });
      expect(firstSocket.readyState).toBe(MockWebSocket.CLOSED);

      // ...and a fresh attempt follows after the first backoff delay.
      await act(async () => {
        vi.advanceTimersByTime(1_000);
      });
      expect(MockWebSocket.instances).toHaveLength(2);

      unmount();
    } finally {
      vi.useRealTimers();
      globalThis.WebSocket = originalWebSocket;
      MockWebSocket.instances.length = 0;
    }
  });

  it("does not close a socket that finished its handshake in time", async () => {
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;
      static instance: MockWebSocket | null = null;

      readonly url: string;
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockWebSocket.instance = this;
      }

      send() {}

      close() {
        this.readyState = MockWebSocket.CLOSED;
      }
    }

    const originalWebSocket = globalThis.WebSocket;
    vi.stubGlobal("WebSocket", MockWebSocket);
    vi.useFakeTimers();

    try {
      const { unmount } = renderHook(() => useWebSocket("ws://acme.com/ws"));
      const socket = MockWebSocket.instance!;
      const closeSpy = vi.spyOn(socket, "close");

      // Act: the handshake completes, then the watchdog window elapses.
      await act(async () => {
        socket.readyState = MockWebSocket.OPEN;
        socket.onopen?.(new Event("open"));
      });
      await act(async () => {
        vi.advanceTimersByTime(60_000);
      });

      // Assert: the cleared watchdog never touched the healthy socket.
      expect(closeSpy).not.toHaveBeenCalled();
      expect(socket.readyState).toBe(MockWebSocket.OPEN);

      unmount();
    } finally {
      vi.useRealTimers();
      globalThis.WebSocket = originalWebSocket;
      MockWebSocket.instance = null;
    }
  });

  it("spaces reconnect attempts with exponential backoff capped at 30s", async () => {
    // Arrange: every connection attempt fails immediately.
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;
      static readonly instances: MockWebSocket[] = [];

      readonly url: string;
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockWebSocket.instances.push(this);
        queueMicrotask(() => {
          this.readyState = MockWebSocket.CLOSED;
          this.onclose?.(
            new CloseEvent("close", {
              code: 1006,
              reason: "",
              wasClean: false,
            }),
          );
        });
      }

      send() {}

      close() {
        this.readyState = MockWebSocket.CLOSED;
      }
    }

    const originalWebSocket = globalThis.WebSocket;
    vi.stubGlobal("WebSocket", MockWebSocket);
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0); // strip the jitter

    const expectInstancesAfter = async (
      advanceMs: number,
      expected: number,
    ) => {
      await act(async () => {
        vi.advanceTimersByTime(advanceMs);
      });
      expect(MockWebSocket.instances).toHaveLength(expected);
    };

    try {
      const { unmount } = renderHook(() =>
        useWebSocket("ws://acme.com/ws", { reconnect: { enabled: true } }),
      );
      // Flush the first attempt's immediate failure.
      await act(async () => {});
      expect(MockWebSocket.instances).toHaveLength(1);

      // Act/Assert: retries land at 1s, then 2s, then 4s after each failure.
      await expectInstancesAfter(999, 1);
      await expectInstancesAfter(1, 2);
      await expectInstancesAfter(1_999, 2);
      await expectInstancesAfter(1, 3);
      await expectInstancesAfter(3_999, 3);
      await expectInstancesAfter(1, 4);
      await expectInstancesAfter(7_999, 4);
      await expectInstancesAfter(1, 5);
      await expectInstancesAfter(15_999, 5);
      await expectInstancesAfter(1, 6);

      // The sixth failure would double to 32s; the cap holds it at 30s...
      await expectInstancesAfter(29_999, 6);
      await expectInstancesAfter(1, 7);

      // ...and every failure after that stays at 30s rather than growing.
      await expectInstancesAfter(29_999, 7);
      await expectInstancesAfter(1, 8);

      unmount();
    } finally {
      vi.useRealTimers();
      globalThis.WebSocket = originalWebSocket;
      MockWebSocket.instances.length = 0;
    }
  });

  it("ignores close/error events from a socket that was replaced", async () => {
    // Arrange: sockets that only emit events when the test fires them, so the
    // old socket's close can land *after* its replacement is open — the race
    // that used to overwrite the new socket's OPEN state.
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;
      static readonly instances: MockWebSocket[] = [];

      readonly url: string;
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: CloseEvent) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockWebSocket.instances.push(this);
      }

      send() {}

      close() {
        this.readyState = MockWebSocket.CLOSED;
      }

      emitOpen() {
        this.readyState = MockWebSocket.OPEN;
        this.onopen?.(new Event("open"));
      }

      emitFailure() {
        this.onerror?.(new Event("error"));
        this.emitClose();
      }

      emitClose() {
        this.onclose?.(
          new CloseEvent("close", { code: 1006, reason: "", wasClean: false }),
        );
      }
    }

    const originalWebSocket = globalThis.WebSocket;
    vi.stubGlobal("WebSocket", MockWebSocket);

    const onCloseSpy = vi.fn();
    const onErrorSpy = vi.fn();

    try {
      const { result, unmount } = renderHook(() =>
        useWebSocket("ws://acme.com/ws", {
          onClose: onCloseSpy,
          onError: onErrorSpy,
        }),
      );
      const staleSocket = MockWebSocket.instances[0];
      act(() => staleSocket.emitOpen());

      // Act: replace the socket, open the replacement, then let the stale
      // socket's error + close events land late.
      act(() => {
        result.current.reconnect();
      });
      const currentSocket = MockWebSocket.instances[1];
      act(() => currentSocket.emitOpen());
      act(() => staleSocket.emitFailure());

      // Assert: the stale socket's events reach neither handler...
      expect(onCloseSpy).not.toHaveBeenCalled();
      expect(onErrorSpy).not.toHaveBeenCalled();

      // ...while the current socket's close still notifies as before.
      act(() => currentSocket.emitClose());
      expect(onCloseSpy).toHaveBeenCalledOnce();
      expect(onErrorSpy).toHaveBeenCalledOnce();

      unmount();
    } finally {
      globalThis.WebSocket = originalWebSocket;
      MockWebSocket.instances.length = 0;
    }
  });

  it("should not send message when WebSocket is not connected", () => {
    const { result } = renderHook(() => useWebSocket("ws://acme.com/ws"));

    // Initially should not be connected
    expect(result.current.isConnected).toBe(false);
    expect(result.current.sendMessage).toBeDefined();

    // Mock the WebSocket send method (even though socket might be null)
    const sendSpy = vi.fn();
    if (result.current.socket) {
      vi.spyOn(result.current.socket, "send").mockImplementation(sendSpy);
    }

    // Try to send a message when not connected
    result.current.sendMessage("Hello WebSocket!");

    // Verify that WebSocket.send was not called
    expect(sendSpy).not.toHaveBeenCalled();
  });
});
