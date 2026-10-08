import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";

const { createClient, executeCommand, close } = vi.hoisted(() => ({
  createClient: vi.fn(),
  executeCommand: vi.fn(),
  close: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@openhands/typescript-client/clients")
  >()),
  BashClient: class {
    constructor(options: unknown) {
      createClient(options);
    }

    executeCommand = executeCommand;
    close = close;
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  executeCommand.mockResolvedValue({ exit_code: 0, stdout: "ok", stderr: "" });
});

describe("bash command execution through the SDK", () => {
  it("forwards command options and releases the client after success", async () => {
    const { result } = renderHook(() =>
      useBashCommandRunner(
        "http://runtime.example.com/api/conversations/conv-1",
        "session-key",
        true,
      ),
    );
    await expect(result.current("pwd", "/workspace", 10)).resolves.toEqual({
      exit_code: 0,
      stdout: "ok",
      stderr: "",
    });
    expect(createClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: "http://runtime.example.com",
        apiKey: "session-key",
        conversationId: "conv-1",
      }),
    );
    expect(executeCommand).toHaveBeenCalledWith({
      command: "pwd",
      cwd: "/workspace",
      timeout: 10,
    });
    expect(close).toHaveBeenCalledOnce();
  });

  it("releases the client and propagates execution failures", async () => {
    const failure = new Error("Runtime unavailable");
    executeCommand.mockRejectedValueOnce(failure);
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", "key", true, "conv-1"),
    );
    await expect(result.current("pwd", "/workspace", 10)).rejects.toBe(failure);
    expect(close).toHaveBeenCalledOnce();
  });

  it("rejects disabled commands without creating a client", async () => {
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", "key", false),
    );
    await expect(result.current("pwd", "/workspace", 10)).rejects.toThrow(
      "disabled",
    );
    expect(createClient).not.toHaveBeenCalled();
  });

  it("uses the latest runtime identity after rerender", async () => {
    const { result, rerender } = renderHook(
      ({ host, key, id }) => useBashCommandRunner(host, key, true, id),
      {
        initialProps: {
          host: "http://first.example.com",
          key: "first",
          id: "one",
        },
      },
    );
    await result.current("pwd", "/workspace", 10);
    rerender({ host: "http://second.example.com", key: "second", id: "two" });
    await result.current("pwd", "/workspace", 10);
    expect(createClient).toHaveBeenLastCalledWith(
      expect.objectContaining({
        host: "http://second.example.com",
        apiKey: "second",
        conversationId: "two",
      }),
    );
    expect(close).toHaveBeenCalledTimes(2);
  });

  it("normalizes absent command output without inventing success", async () => {
    executeCommand.mockResolvedValueOnce({});
    const { result } = renderHook(() =>
      useBashCommandRunner("http://runtime.example.com", "key", true),
    );
    await expect(result.current("pwd", "/workspace", 10)).resolves.toEqual({
      exit_code: -1,
      stdout: "",
      stderr: "",
    });
    expect(close).toHaveBeenCalledOnce();
  });
});
