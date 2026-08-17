import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";

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
  }

  send(data: string) {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new DOMException("WebSocket is not open", "InvalidStateError");
    }
    this.sent.push(data);
  }

  open() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  receive(data: unknown) {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(data) }),
    );
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
  }
}

describe("useBashCommandRunner", () => {
  afterEach(() => {
    MockWebSocket.instance = null;
    vi.unstubAllGlobals();
  });

  it("sends auth before queued commands without putting the key in the URL", async () => {
    vi.stubGlobal("WebSocket", MockWebSocket);
    const sessionApiKey = `sk-oh-${"b".repeat(64)}`;
    const { result, unmount } = renderHook(() =>
      useBashCommandRunner(
        "https://runtime.example.com/api/conversations/conv-1",
        sessionApiKey,
        true,
      ),
    );
    const socket = MockWebSocket.instance!;
    socket.readyState = MockWebSocket.OPEN;

    const command = result.current("pwd", "/workspace", 30);

    expect(socket.sent).toEqual([]);
    socket.open();

    expect(socket.url).not.toContain(sessionApiKey);
    expect(socket.url).not.toContain("session_api_key");
    expect(socket.sent).toEqual([
      JSON.stringify({ type: "auth", session_api_key: sessionApiKey }),
      JSON.stringify({ command: "pwd", cwd: "/workspace", timeout: 30 }),
    ]);

    socket.receive({ kind: "BashCommand", id: "command-1" });
    socket.receive({
      kind: "BashOutput",
      command_id: "command-1",
      stdout: "/workspace\n",
      stderr: "",
      exit_code: 0,
    });
    await expect(command).resolves.toEqual({
      exit_code: 0,
      stdout: "/workspace\n",
      stderr: "",
    });

    unmount();
  });

  it("sends queued commands without an auth frame when no key is configured", async () => {
    vi.stubGlobal("WebSocket", MockWebSocket);
    const { result, unmount } = renderHook(() =>
      useBashCommandRunner(
        "http://runtime.example.com/api/conversations/conv-1",
        null,
        true,
      ),
    );
    const socket = MockWebSocket.instance!;
    const command = result.current("git status", "/workspace", 10);

    expect(socket.sent).toEqual([]);
    socket.open();
    expect(socket.sent).toEqual([
      JSON.stringify({
        command: "git status",
        cwd: "/workspace",
        timeout: 10,
      }),
    ]);

    socket.receive({ kind: "BashCommand", id: "command-1" });
    socket.receive({
      kind: "BashOutput",
      command_id: "command-1",
      stdout: "",
      stderr: "",
      exit_code: 0,
    });
    await expect(command).resolves.toEqual({
      exit_code: 0,
      stdout: "",
      stderr: "",
    });

    unmount();
  });
});
