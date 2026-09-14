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
});
