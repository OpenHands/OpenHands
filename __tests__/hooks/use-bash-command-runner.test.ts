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

    socket.receive({
      kind: "BashCommand",
      id: "command-1",
      command: "pwd",
      cwd: "/workspace",
      timeout: 30,
    });
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

    socket.receive({
      kind: "BashCommand",
      id: "command-1",
      command: "git status",
      cwd: "/workspace",
      timeout: 10,
    });
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

  // Regression test for #15543 ("Bad parsing for branch retrieval"), where a
  // filename showed up in the git branch badge.
  //
  // `/sockets/bash-events` is not a private channel: the agent-server
  // subscribes every socket to one shared `BashEventService` PubSub, and
  // `start_bash_command` publishes the `BashCommand` event to all subscribers
  // regardless of who started it (REST `POST /api/bash/start_bash_command`
  // included). This hook used to pair each echo with the oldest outstanding
  // request by queue position, so a command we never sent could capture our
  // pending promise and resolve it with its own stdout.
  //
  // Downstream, `useLocalGitInfo`'s `probeGitInfo` reads line 1 as the git
  // remote URL and everything after the first newline as the branch — which is
  // how foreign output whose second line was a filename rendered that filename
  // as the branch.
  it("ignores a foreign command's echo and still resolves with its own output", async () => {
    vi.stubGlobal("WebSocket", MockWebSocket);
    const { result, unmount } = renderHook(() =>
      useBashCommandRunner(
        "http://runtime.example.com/api/conversations/conv-1",
        null,
        true,
      ),
    );
    const socket = MockWebSocket.instance!;
    socket.open();

    // The git-info probe this hook actually exists to run.
    const probe = result.current(
      "git rev-parse --abbrev-ref HEAD",
      "/workspace",
      10,
    );
    expect(socket.sent).toEqual([
      JSON.stringify({
        command: "git rev-parse --abbrev-ref HEAD",
        cwd: "/workspace",
        timeout: 10,
      }),
    ]);

    // Something else on this conversation starts a command — another client,
    // an automation run, a second tab. The server broadcasts it to us too.
    // Note the id and the command text are both plainly not ours.
    socket.receive({
      kind: "BashCommand",
      id: "foreign-1",
      command: "ls",
      cwd: "/tmp",
      timeout: 10,
    });
    socket.receive({
      kind: "BashOutput",
      command_id: "foreign-1",
      stdout: "x\nNOT-A-BRANCH.txt",
      stderr: "",
      exit_code: 0,
      order: 0,
    });

    // The probe must not adopt that output. Racing against a sentinel is how
    // we assert "still pending" without hanging the test.
    const pendingSentinel = Symbol("pending");
    await expect(
      Promise.race([probe, Promise.resolve().then(() => pendingSentinel)]),
    ).resolves.toBe(pendingSentinel);

    // Our own echo and output still land correctly afterwards.
    socket.receive({
      kind: "BashCommand",
      id: "ours-1",
      command: "git rev-parse --abbrev-ref HEAD",
      cwd: "/workspace",
      timeout: 10,
    });
    socket.receive({
      kind: "BashOutput",
      command_id: "ours-1",
      stdout: "main\n",
      stderr: "",
      exit_code: 0,
      order: 0,
    });
    await expect(probe).resolves.toEqual({
      exit_code: 0,
      stdout: "main\n",
      stderr: "",
    });

    unmount();
  });

  // Two conversations probing different workspaces run a byte-identical script,
  // so `command` alone cannot tell their echoes apart — `cwd` breaks the tie.
  it("routes echoes of an identical command to the request with the matching cwd", async () => {
    vi.stubGlobal("WebSocket", MockWebSocket);
    const { result, unmount } = renderHook(() =>
      useBashCommandRunner(
        "http://runtime.example.com/api/conversations/conv-1",
        null,
        true,
      ),
    );
    const socket = MockWebSocket.instance!;
    socket.open();

    const probeA = result.current("git branch --show-current", "/repo-a", 10);
    const probeB = result.current("git branch --show-current", "/repo-b", 10);

    // B's echo arrives first, out of request order.
    socket.receive({
      kind: "BashCommand",
      id: "cmd-b",
      command: "git branch --show-current",
      cwd: "/repo-b",
      timeout: 10,
    });
    socket.receive({
      kind: "BashOutput",
      command_id: "cmd-b",
      stdout: "branch-b\n",
      stderr: "",
      exit_code: 0,
      order: 0,
    });
    socket.receive({
      kind: "BashCommand",
      id: "cmd-a",
      command: "git branch --show-current",
      cwd: "/repo-a",
      timeout: 10,
    });
    socket.receive({
      kind: "BashOutput",
      command_id: "cmd-a",
      stdout: "branch-a\n",
      stderr: "",
      exit_code: 0,
      order: 0,
    });

    await expect(probeA).resolves.toMatchObject({ stdout: "branch-a\n" });
    await expect(probeB).resolves.toMatchObject({ stdout: "branch-b\n" });

    unmount();
  });
});
