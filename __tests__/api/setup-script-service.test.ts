import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  clearCachedAgentServerInfo,
  isSdkHttpStatusError,
} from "#/api/agent-server-compatibility";
import { isSdkHttpError } from "#/utils/sdk-http-error";
import type { Backend } from "#/api/backend-registry/types";

const {
  mockDownloadTextFile,
  mockExecuteTool,
  mockGetServerInfo,
  mockConversationClient,
  mockFileClient,
  mockServerClient,
} = vi.hoisted(() => {
  const downloadTextFile = vi.fn();
  const executeTool = vi.fn();
  const getServerInfo = vi.fn();
  return {
    mockDownloadTextFile: downloadTextFile,
    mockExecuteTool: executeTool,
    mockGetServerInfo: getServerInfo,
    mockConversationClient: vi.fn(function ConversationClientMock() {
      return { executeTool };
    }),
    mockFileClient: vi.fn(function FileClientMock() {
      return { downloadTextFile };
    }),
    mockServerClient: vi.fn(function ServerClientMock() {
      return { getServerInfo };
    }),
  };
});

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: mockConversationClient,
  FileClient: mockFileClient,
  ServerClient: mockServerClient,
}));

import SetupScriptService, {
  DIRECT_TOOL_EXECUTION_CAPABILITY,
} from "#/api/setup-script-service";

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8000",
  apiKey: "",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud",
  name: "Cloud",
  host: "https://openhands.dev",
  apiKey: "cloud-key",
  kind: "cloud",
};

/** An sdk http error shaped like the client's, for the 404/409 branches. */
function sdkStatusError(status: number) {
  return Object.assign(new Error(`HTTP ${status}`), {
    name: "HttpError",
    status,
  });
}

const supportingServerInfo = {
  capabilities: [DIRECT_TOOL_EXECUTION_CAPABILITY],
};

const unsupportedServerInfo = { capabilities: ["hooks"] };

const plan = {
  workingDir: "/workspace/project",
  scriptPath: "/workspace/project/.openhands/setup.sh",
  command:
    "chmod +x '/workspace/project/.openhands/setup.sh' && . '/workspace/project/.openhands/setup.sh'",
};

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  clearCachedAgentServerInfo();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  mockDownloadTextFile.mockReset();
  mockExecuteTool.mockReset();
  mockGetServerInfo.mockReset();
  mockConversationClient.mockClear();
  mockFileClient.mockClear();
  mockServerClient.mockClear();
});

afterEach(() => {
  __resetActiveStoreForTests();
  clearCachedAgentServerInfo();
});

describe("SetupScriptService.prepare", () => {
  it("returns a plan when the workspace has the script", async () => {
    mockDownloadTextFile.mockResolvedValue(
      "#!/bin/sh\nexport PATH=$PATH:/opt/bin",
    );

    expect(await SetupScriptService.prepare("/workspace/project")).toEqual(plan);
  });

  it("returns null when the workspace has no setup script", async () => {
    mockDownloadTextFile.mockRejectedValue(sdkStatusError(404));

    expect(await SetupScriptService.prepare("/workspace/project")).toBeNull();
  });

  it("quotes the path so a shell-metacharacter workspace cannot inject", async () => {
    mockDownloadTextFile.mockResolvedValue("#!/bin/sh");

    const result = await SetupScriptService.prepare("/tmp/$(touch x)/it's");

    expect(result?.command).toBe(
      "chmod +x '/tmp/$(touch x)/it'\\''s/.openhands/setup.sh' && . '/tmp/$(touch x)/it'\\''s/.openhands/setup.sh'",
    );
  });
});

describe("SetupScriptService.isSupported", () => {
  it("requires the server capability and a client that can call it", async () => {
    mockGetServerInfo.mockResolvedValue(supportingServerInfo);

    expect(await SetupScriptService.isSupported()).toBe(true);
  });

  it("is false when the agent-server lacks the capability", async () => {
    mockGetServerInfo.mockResolvedValue(unsupportedServerInfo);

    expect(await SetupScriptService.isSupported()).toBe(false);
  });

  it("is false when the pinned client has no executeTool", async () => {
    mockGetServerInfo.mockResolvedValue(supportingServerInfo);
    // A server that advertises the capability while this Canvas build pins a
    // client predating `executeTool` must skip, not throw mid-launch.
    mockConversationClient.mockImplementationOnce(
      (function ConversationClientMock() {
        return {};
      }) as unknown as Parameters<
        typeof mockConversationClient.mockImplementationOnce
      >[0],
    );

    expect(await SetupScriptService.isSupported()).toBe(false);
  });

  it("is false on a cloud backend without probing", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    expect(await SetupScriptService.isSupported()).toBe(false);
    expect(mockGetServerInfo).not.toHaveBeenCalled();
    expect(mockDownloadTextFile).not.toHaveBeenCalled();
  });

  it("is false when the capability probe fails", async () => {
    mockGetServerInfo.mockRejectedValue(new Error("unreachable"));

    expect(await SetupScriptService.isSupported()).toBe(false);
  });
});

describe("SetupScriptService.run", () => {
  it("runs the script through the terminal and reports success", async () => {
    const onProgress = vi.fn();
    mockExecuteTool.mockResolvedValue({
      observation: {
        kind: "TerminalObservation",
        exit_code: 0,
        content: [{ type: "text", text: "installed" }],
      },
    });

    const result = await SetupScriptService.run(plan, "conv-1", onProgress);

    expect(mockExecuteTool).toHaveBeenCalledWith("conv-1", {
      tool_name: "terminal",
      action: { kind: "TerminalAction", command: plan.command },
    });
    expect(result).toEqual({
      status: "ran",
      exitCode: 0,
      output: "installed",
      failed: false,
    });
    expect(onProgress.mock.calls.map(([update]) => update.phase)).toEqual([
      "running",
      "succeeded",
    ]);
  });

  it("reports a non-zero exit as a failed run", async () => {
    const onProgress = vi.fn();
    mockExecuteTool.mockResolvedValue({
      observation: {
        kind: "TerminalObservation",
        exit_code: 2,
        content: [{ type: "text", text: "npm ERR!" }],
      },
    });

    const result = await SetupScriptService.run(plan, "conv-1", onProgress);

    expect(result).toMatchObject({ status: "ran", exitCode: 2, failed: true });
    expect(onProgress).toHaveBeenLastCalledWith({
      phase: "failed",
      exitCode: 2,
      output: "npm ERR!",
    });
  });

  it("retries while the conversation is busy, then succeeds", async () => {
    vi.useFakeTimers();
    mockExecuteTool
      .mockRejectedValueOnce(sdkStatusError(409))
      .mockResolvedValueOnce({
        observation: { kind: "TerminalObservation", exit_code: 0 },
      });

    const pending = SetupScriptService.run(plan, "conv-1");
    await vi.runAllTimersAsync();

    await expect(pending).resolves.toMatchObject({
      status: "ran",
      failed: false,
    });
    expect(mockExecuteTool).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("returns an error result instead of throwing when the call fails", async () => {
    const onProgress = vi.fn();
    mockExecuteTool.mockRejectedValue(new Error("no such route"));

    const result = await SetupScriptService.run(plan, "conv-1", onProgress);

    expect(result).toEqual({ status: "error", message: "no such route" });
    expect(onProgress).toHaveBeenLastCalledWith({
      phase: "error",
      message: "no such route",
    });
  });
});

describe("SetupScriptService.isDirectToolExecutionSupported", () => {
  it("requires the capability to be present in the list", () => {
    expect(
      SetupScriptService.isDirectToolExecutionSupported(supportingServerInfo),
    ).toBe(true);
    expect(
      SetupScriptService.isDirectToolExecutionSupported(unsupportedServerInfo),
    ).toBe(false);
    expect(SetupScriptService.isDirectToolExecutionSupported(null)).toBe(false);
    expect(SetupScriptService.isDirectToolExecutionSupported({})).toBe(false);
  });
});

// Guard the 404 detection helper the service relies on; a change to either
// side would silently turn "no script" into "skip after a warning".
describe("sdk http status detection", () => {
  it("recognises the 404 the service uses to mean 'no script'", () => {
    expect(isSdkHttpStatusError(sdkStatusError(404), 404)).toBe(true);
    expect(isSdkHttpError(sdkStatusError(404))).toBe(true);
  });
});
