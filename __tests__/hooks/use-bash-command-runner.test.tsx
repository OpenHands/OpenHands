import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";

function installControlledWebSocket() {
  class ControlledWebSocket {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSING = 2;
    static readonly CLOSED = 3;

    readonly sent: string[] = [];
    readonly url: string;
    readyState = ControlledWebSocket.CONNECTING;
    onopen: ((event: Event) => void) | null = null;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;
    close = vi.fn(() => {
      this.readyState = ControlledWebSocket.CLOSED;
    });

    constructor(url: string) {
      this.url = url;
      instances.push(this);
    }

    send(data: string) {
      this.sent.push(data);
    }

    open() {
      this.readyState = ControlledWebSocket.OPEN;
      this.onopen?.(new Event("open"));
    }

    emitRaw(data: string) {
      this.onmessage?.(new MessageEvent("message", { data }));
    }

    emit(data: Record<string, unknown>) {
      this.emitRaw(JSON.stringify(data));
    }

    fail() {
      this.onerror?.(new Event("error"));
    }

    disconnect() {
      this.readyState = ControlledWebSocket.CLOSED;
      this.onclose?.(new CloseEvent("close"));
    }
  }

  const instances: ControlledWebSocket[] = [];
  vi.stubGlobal("WebSocket", ControlledWebSocket);

  return {
    instances,
    get socket() {
      const socket = instances.at(-1);
      if (!socket) throw new Error("Expected a WebSocket to be created");
      return socket;
    },
  };
}

const bashCommand = (id: string) => ({
  kind: "BashCommand",
  id,
  timestamp: "2026-07-12T00:00:00.000Z",
  command: "ignored server echo",
});

const bashOutput = (
  commandId: string,
  output: {
    stdout?: string | null;
    stderr?: string | null;
    exit_code?: number | null;
  },
) => ({
  kind: "BashOutput",
  id: `output-${commandId}`,
  timestamp: "2026-07-12T00:00:01.000Z",
  command_id: commandId,
  order: 0,
  ...output,
});

const bashError = (code: string, detail: string) => ({
  kind: "BashError",
  id: `error-${code}`,
  timestamp: "2026-07-12T00:00:01.000Z",
  code,
  detail,
});

function withSettlementDeadline<T>(promise: Promise<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error("Bash command promise did not settle")),
      250,
    );
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("bash command execution over a persistent socket", () => {
  it("stays disconnected when disabled and rejects commands immediately", async () => {
    const sockets = installControlledWebSocket();
    const { result } = renderHook(() =>
      useBashCommandRunner(
        "https://runtime.example.com/api/conversations/conv-1",
        "session-key",
        false,
      ),
    );

    await expect(
      withSettlementDeadline(result.current("pwd", "/workspace", 10)),
    ).rejects.toThrow("Bash WebSocket not available");
    expect(sockets.instances).toHaveLength(0);
  });

  it("buffers commands when the socket is OPEN but not yet the ready socket", async () => {
    const sockets = installControlledWebSocket();
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", "session-key", true),
    );

    // The socket reports OPEN before the hook's open handler has run, so it is
    // not yet the tracked "ready" socket. Commands must still be buffered
    // rather than sent on a socket the hook has not authenticated.
    act(() => {
      sockets.socket.readyState = 1; // WebSocket.OPEN
    });
    const pending = withSettlementDeadline(
      result.current("pwd", "/workspace", 10),
    );
    expect(sockets.socket.sent).toEqual([]);

    // Once the open handler runs, the hook authenticates and flushes the queue.
    act(() => sockets.socket.open());
    expect(sockets.socket.sent.map((payload) => JSON.parse(payload))).toEqual([
      { type: "auth", session_api_key: "session-key" },
      { command: "pwd", cwd: "/workspace", timeout: 10 },
    ]);

    act(() => sockets.socket.emit(bashCommand("command-1")));
    act(() =>
      sockets.socket.emit(bashOutput("command-1", { stdout: "ok", exit_code: 0 })),
    );
    await expect(pending).resolves.toEqual({
      exit_code: 0,
      stdout: "ok",
      stderr: "",
    });

    // A command issued once the socket is ready is sent immediately rather than
    // buffered.
    const followUp = withSettlementDeadline(
      result.current("ls", "/workspace", 5),
    );
    expect(
      sockets.socket.sent.map((payload) => JSON.parse(payload)).at(-1),
    ).toEqual({ command: "ls", cwd: "/workspace", timeout: 5 });
    act(() => sockets.socket.emit(bashCommand("command-2")));
    act(() =>
      sockets.socket.emit(bashOutput("command-2", { stdout: "", exit_code: 0 })),
    );
    await expect(followUp).resolves.toEqual({
      exit_code: 0,
      stdout: "",
      stderr: "",
    });
  });

  it("buffers commands while connecting, correlates them FIFO, and aggregates output chunks", async () => {
    const sockets = installControlledWebSocket();
    const { result } = renderHook(() =>
      useBashCommandRunner(
        "https://runtime.example.com/runtime/18000/api/conversations/conv-1",
        "key with spaces",
        true,
      ),
    );

    const first = withSettlementDeadline(
      result.current("printf first", "/workspace/one", 11),
    );
    const second = withSettlementDeadline(
      result.current("printf second", "/workspace/two", 22),
    );
    expect(sockets.socket.sent).toEqual([]);
    expect(sockets.socket.url).toBe(
      "wss://runtime.example.com/runtime/18000/sockets/bash-events",
    );

    act(() => sockets.socket.open());
    // On open the hook first authenticates over the socket, then flushes the
    // buffered commands in FIFO order.
    expect(sockets.socket.sent.map((payload) => JSON.parse(payload))).toEqual([
      { type: "auth", session_api_key: "key with spaces" },
      { command: "printf first", cwd: "/workspace/one", timeout: 11 },
      { command: "printf second", cwd: "/workspace/two", timeout: 22 },
    ]);

    act(() => {
      sockets.socket.emit(bashCommand("command-1"));
      sockets.socket.emit(
        bashOutput("command-1", {
          stdout: "first ",
          stderr: "warn ",
          exit_code: null,
        }),
      );
      sockets.socket.emit(
        bashOutput("command-1", {
          stdout: "result",
          stderr: "again",
          exit_code: 0,
        }),
      );
      sockets.socket.emit(bashCommand("command-2"));
      sockets.socket.emit(
        bashOutput("command-2", {
          stdout: "second result",
          stderr: "",
          exit_code: 7,
        }),
      );
    });

    await expect(first).resolves.toEqual({
      exit_code: 0,
      stdout: "first result",
      stderr: "warn again",
    });
    await expect(second).resolves.toEqual({
      exit_code: 7,
      stdout: "second result",
      stderr: "",
    });
  });
  it("ignores malformed, unknown, uncorrelated, and incomplete server frames", async () => {
    const sockets = installControlledWebSocket();
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", null, true),
    );
    act(() => sockets.socket.open());

    const command = withSettlementDeadline(
      result.current("echo safe", "/workspace", 5),
    );
    act(() => {
      sockets.socket.emitRaw("not-json");
      sockets.socket.emit({ kind: "FutureBashEvent" });
      sockets.socket.emit(bashOutput("", { exit_code: 0 }));
      sockets.socket.emit(bashOutput("not-active", { exit_code: 0 }));
      sockets.socket.emit(bashCommand("active-command"));
      sockets.socket.emit({
        kind: "FutureBashEvent",
        command_id: "active-command",
        stdout: "forged",
        stderr: "forged",
        exit_code: 99,
      });
      sockets.socket.emit(bashCommand("unrequested-command"));
      sockets.socket.emit(
        bashOutput("active-command", {
          stdout: "",
          stderr: null,
          exit_code: null,
        }),
      );
    });

    expect(
      await Promise.race([command.then(() => "resolved"), "pending"]),
    ).toBe("pending");

    act(() => {
      sockets.socket.emit(
        bashOutput("active-command", {
          stdout: null,
          stderr: "",
          exit_code: 0,
        }),
      );
    });
    await expect(command).resolves.toEqual({
      exit_code: 0,
      stdout: "",
      stderr: "",
    });
  });

  it("rejects commands waiting for connection when the server reports a bash error", async () => {
    const sockets = installControlledWebSocket();
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", null, true),
    );
    const command = withSettlementDeadline(
      result.current("pwd", "/workspace", 10),
    );

    act(() => sockets.socket.emit(bashError("NOT_READY", "runtime starting")));

    await expect(command).rejects.toThrow(
      "Bash error: NOT_READY: runtime starting",
    );
  });

  it("rejects sent commands that have not received their server id on socket error", async () => {
    const sockets = installControlledWebSocket();
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", null, true),
    );
    act(() => sockets.socket.open());
    const command = withSettlementDeadline(
      result.current("pwd", "/workspace", 10),
    );
    expect(sockets.socket.sent.map((payload) => JSON.parse(payload))).toEqual([
      { command: "pwd", cwd: "/workspace", timeout: 10 },
    ]);

    act(() => sockets.socket.fail());

    await expect(command).rejects.toThrow("Bash WebSocket error");
    await expect(
      withSettlementDeadline(result.current("pwd", "/workspace", 10)),
    ).rejects.toThrow("Bash WebSocket not available");
  });

  it("rejects active commands when the socket closes", async () => {
    const sockets = installControlledWebSocket();
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", null, true),
    );
    act(() => sockets.socket.open());
    const command = withSettlementDeadline(
      result.current("pwd", "/workspace", 10),
    );
    act(() => sockets.socket.emit(bashCommand("active-command")));

    act(() => sockets.socket.disconnect());

    await expect(command).rejects.toThrow("Bash WebSocket closed");
  });

  it("rejects pending work on unmount without running close handlers twice", async () => {
    const sockets = installControlledWebSocket();
    const { result, unmount } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", null, true),
    );
    act(() => sockets.socket.open());
    const command = withSettlementDeadline(
      result.current("pwd", "/workspace", 10),
    );

    unmount();

    await expect(command).rejects.toThrow("Bash WebSocket unmounted");
    expect(sockets.socket.close).toHaveBeenCalledOnce();
    expect(sockets.socket.onclose).toBeNull();
    expect(sockets.socket.onerror).toBeNull();
  });

  it("reconnects with the latest runtime URL, session key, and enabled state", async () => {
    const sockets = installControlledWebSocket();
    const { result, rerender } = renderHook(
      ({ url, sessionKey, enabled }) =>
        useBashCommandRunner(url, sessionKey, enabled),
      {
        initialProps: {
          url: "https://runtime-a.example.com/api/conversations/conv-1",
          sessionKey: "key-a",
          enabled: true,
        },
      },
    );
    const firstSocket = sockets.socket;
    const queuedCommand = withSettlementDeadline(
      result.current("pwd", "/workspace", 10),
    );

    rerender({
      url: "https://runtime-b.example.com/api/conversations/conv-2",
      sessionKey: "key-b",
      enabled: true,
    });

    await expect(queuedCommand).rejects.toThrow("Bash WebSocket unmounted");
    expect(firstSocket.close).toHaveBeenCalledOnce();
    expect(sockets.instances).toHaveLength(2);
    expect(sockets.socket.url).toBe(
      "wss://runtime-b.example.com/sockets/bash-events",
    );

    const secondSocket = sockets.socket;
    // The reconnected socket authenticates with the latest session key.
    act(() => secondSocket.open());
    expect(secondSocket.sent.map((payload) => JSON.parse(payload))).toEqual([
      { type: "auth", session_api_key: "key-b" },
    ]);
    rerender({
      url: "https://runtime-b.example.com/api/conversations/conv-2",
      sessionKey: "key-b",
      enabled: false,
    });
    expect(secondSocket.close).toHaveBeenCalledOnce();
    expect(sockets.instances).toHaveLength(2);
  });

  it.each([
    { state: 2, label: "closing" },
    { state: 3, label: "closed" },
  ])("rejects commands while the socket is $label", async ({ state }) => {
    const sockets = installControlledWebSocket();
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", null, true),
    );
    sockets.socket.readyState = state;

    await expect(
      withSettlementDeadline(result.current("pwd", "/workspace", 10)),
    ).rejects.toThrow("Bash WebSocket not available");
  });
});
