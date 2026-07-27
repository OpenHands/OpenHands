import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";

describe("useBashCommandRunner", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("authenticates without putting the session key in the URL", async () => {
    class MockWebSocket {
      static readonly CONNECTING = 0;
      static readonly OPEN = 1;
      static readonly CLOSING = 2;
      static readonly CLOSED = 3;
      static instance: MockWebSocket | null = null;

      readonly url: string;
      readonly sent: string[] = [];
      readyState = MockWebSocket.CONNECTING;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: (() => void) | null = null;
      onerror: (() => void) | null = null;

      constructor(url: string) {
        this.url = url;
        MockWebSocket.instance = this;
        queueMicrotask(() => {
          this.readyState = MockWebSocket.OPEN;
          this.onopen?.();
        });
      }

      send(data: string) {
        this.sent.push(data);
      }

      close() {
        this.readyState = MockWebSocket.CLOSED;
      }
    }

    vi.stubGlobal("WebSocket", MockWebSocket);
    const sessionApiKey = `sk-oh-${"b".repeat(64)}`;
    const { unmount } = renderHook(() =>
      useBashCommandRunner(
        "https://runtime.example.com/api/conversations/conv-1",
        sessionApiKey,
        true,
      ),
    );

    await waitFor(() => expect(MockWebSocket.instance?.sent).toHaveLength(1));

    expect(MockWebSocket.instance?.url).not.toContain(sessionApiKey);
    expect(MockWebSocket.instance?.url).not.toContain("session_api_key");
    expect(MockWebSocket.instance?.sent[0]).toBe(
      JSON.stringify({ type: "auth", session_api_key: sessionApiKey }),
    );

    unmount();
  });
});
