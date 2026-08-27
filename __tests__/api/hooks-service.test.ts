import { HooksClient } from "@openhands/typescript-client/clients";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

const { mockLoadHooks } = vi.hoisted(() => ({
  mockLoadHooks: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  HooksClient: vi.fn(function HooksClientMock() {
    return { loadHooks: mockLoadHooks };
  }),
}));

import HooksService from "#/api/hooks-service";

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

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  mockLoadHooks.mockReset();
  vi.mocked(HooksClient).mockClear();
});

afterEach(() => {
  vi.unstubAllEnvs();
  __resetActiveStoreForTests();
});

describe("HooksService.loadWorkspaceHooks", () => {
  it("loads workspace hooks from the agent-server when present", async () => {
    const mockHookConfig = {
      session_start: [
        {
          matcher: "*",
          hooks: [{ command: "cat AGENTS.md", type: "command" }],
        },
      ],
      pre_tool_use: [],
      post_tool_use: [],
      user_prompt_submit: [],
      session_end: [],
      stop: [],
    };

    mockLoadHooks.mockResolvedValue({ hook_config: mockHookConfig });

    const result = await HooksService.loadWorkspaceHooks(
      "/workspace/test-project",
    );

    expect(mockLoadHooks).toHaveBeenCalledTimes(1);
    expect(mockLoadHooks).toHaveBeenCalledWith({
      project_dir: "/workspace/test-project",
    });
    expect(result).toEqual(mockHookConfig);
  });

  it("defaults project_dir to getAgentServerWorkingDir() when omitted", async () => {
    mockLoadHooks.mockResolvedValue({ hook_config: null });

    const result = await HooksService.loadWorkspaceHooks();

    expect(mockLoadHooks).toHaveBeenCalledTimes(1);
    expect(mockLoadHooks).toHaveBeenCalledWith({
      project_dir: "workspace/project",
    });
    expect(result).toBeNull();
  });

  it("gracefully returns null when the agent-server throws an error", async () => {
    mockLoadHooks.mockRejectedValue(new Error("500 Internal Server Error"));

    const result = await HooksService.loadWorkspaceHooks("/workspace/broken");

    expect(result).toBeNull();
  });

  it("returns null immediately for cloud backend without calling HooksClient", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    const result = await HooksService.loadWorkspaceHooks("/workspace/cloud");

    expect(mockLoadHooks).not.toHaveBeenCalled();
    expect(result).toBeNull();
  });
});
