import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";
import { listCloudConversationFiles } from "#/api/cloud/conversation-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { DEFAULT_FILE_DISCOVERY } from "#/utils/workspace-file-discovery";

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: { getSettings: vi.fn(), saveSettings: vi.fn() },
}));

// The hook reads cloud/local from the backend-registry store (the same source
// the transport layer branches on), so drive the store snapshot in tests.
// `getSnapshot` must return a STABLE reference per state or `useSyncExternalStore`
// re-renders forever — precompute one frozen snapshot per kind.
const STORE_SNAPSHOTS = {
  local: {
    active: {
      backend: {
        id: "backend-id",
        name: "Local",
        host: "http://127.0.0.1:8000",
        apiKey: "test-key",
        kind: "local",
      },
      orgId: null,
    },
  },
  cloud: {
    active: {
      backend: {
        id: "backend-id",
        name: "Production",
        host: "https://app.all-hands.dev",
        apiKey: "test-key",
        kind: "cloud",
      },
      orgId: null,
    },
  },
} as const;
let storeBackendKind: "local" | "cloud" = "local";
vi.mock("#/api/backend-registry/active-store", () => ({
  subscribeActiveBackend: () => () => {},
  getSnapshot: () => STORE_SNAPSHOTS[storeBackendKind],
}));

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const useRuntimeIsReadyMock = vi.fn();
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: (...args: unknown[]) => useRuntimeIsReadyMock(...args),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: "error" }),
}));

const useOptionalConversationIdMock = vi.fn();
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => useOptionalConversationIdMock(),
}));

vi.mock("#/api/cloud/conversation-service.api", () => ({
  listCloudConversationFiles: vi.fn(),
}));

const executeCommandSpy = vi.spyOn(AgentServerRuntimeService, "executeCommand");
const listCloudFilesMock = vi.mocked(listCloudConversationFiles);

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function WorkspaceFilesTestWrapper({
    children,
  }: {
    children: React.ReactNode;
  }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

const conversation = {
  id: "conv-1",
  conversation_url: "https://runtime.example.com/api/conversations/conv-1",
  session_api_key: "session-key",
  workspace: { working_dir: "/workspace/project" },
};

beforeEach(() => {
  storeBackendKind = "local";
  useActiveConversationMock.mockReset();
  useRuntimeIsReadyMock.mockReset();
  useOptionalConversationIdMock.mockReset();
  executeCommandSpy.mockReset();
  listCloudFilesMock.mockReset();
  vi.mocked(SettingsService.getSettings).mockResolvedValue(DEFAULT_SETTINGS);

  useRuntimeIsReadyMock.mockReturnValue(true);
  useActiveConversationMock.mockReturnValue({ data: conversation });
  useOptionalConversationIdMock.mockReturnValue({ conversationId: "conv-1" });
  listCloudFilesMock.mockResolvedValue([]);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("useWorkspaceFiles — local backend", () => {
  beforeEach(() => {
    storeBackendKind = "local";
  });

  // @spec WFD-002 — Workspace-scoped server persistence
  it("uses persisted limits for the active workspace and changes them on navigation", async () => {
    vi.mocked(SettingsService.getSettings).mockResolvedValue({
      ...DEFAULT_SETTINGS,
      workspace_file_discovery: {
        "/workspace/project": { ...DEFAULT_FILE_DISCOVERY, maxFiles: 1 },
        "/workspace/other": { ...DEFAULT_FILE_DISCOVERY, maxFiles: 0 },
      },
    });
    executeCommandSpy.mockResolvedValue({
      exit_code: 0,
      stdout: "./a.txt\n./b.txt\n",
      stderr: "",
    });
    const { result, rerender } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.data).toEqual(["a.txt"]));
    expect(result.current.isTruncated).toBe(true);
    useActiveConversationMock.mockReturnValue({
      data: {
        ...conversation,
        id: "conv-2",
        workspace: { working_dir: "/workspace/other" },
      },
    });
    rerender();
    await waitFor(() =>
      expect(result.current.data).toEqual(["a.txt", "b.txt"]),
    );
    expect(result.current.isTruncated).toBe(false);
  });

  it("lists diagnostic files while the conversation is in Error", async () => {
    const { useRuntimeIsReady } = await vi.importActual<
      typeof import("#/hooks/use-runtime-is-ready")
    >("#/hooks/use-runtime-is-ready");
    useRuntimeIsReadyMock.mockImplementation(useRuntimeIsReady);
    useActiveConversationMock.mockReturnValue({
      data: { ...conversation, execution_status: "error" },
    });
    executeCommandSpy.mockResolvedValue({
      exit_code: 0,
      stdout: "./evidence/checkpoint.json\n",
      stderr: "",
    });
    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() =>
      expect(result.current.data).toEqual(["evidence/checkpoint.json"]),
    );
  });

  it("lists files via bash find and does not touch git changes", async () => {
    executeCommandSpy.mockResolvedValue({
      exit_code: 0,
      stdout: "./hello.txt\n./src/index.ts\n",
      stderr: "",
    });

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.data).toEqual(["hello.txt", "src/index.ts"]),
    );
    expect(executeCommandSpy).toHaveBeenCalledTimes(1);
  });
});

describe("useWorkspaceFiles — cloud backend", () => {
  beforeEach(() => {
    storeBackendKind = "cloud";
  });

  it("lists the full tree via the cloud files endpoint without running bash", async () => {
    listCloudFilesMock.mockResolvedValue([
      "hello.txt",
      "src/index.ts",
      "src/untouched.ts",
    ]);

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.data).toEqual([
        "hello.txt",
        "src/index.ts",
        "src/untouched.ts",
      ]),
    );
    // The cloud path uses the first-class listing endpoint, anchored at the
    // conversation's absolute working dir.
    expect(listCloudFilesMock).toHaveBeenCalledWith(
      "conv-1",
      "/workspace/project",
    );
    // Cloud must never drive the removed bash/cloud-proxy path.
    expect(executeCommandSpy).not.toHaveBeenCalled();
  });

  it("normalizes leading ./ and de-dupes the returned paths", async () => {
    listCloudFilesMock.mockResolvedValue([
      "./hello.txt",
      "hello.txt",
      "./src/index.ts",
    ]);

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.data).toEqual(["hello.txt", "src/index.ts"]),
    );
  });

  it("fires using the route id even when the active-conversation query has no data yet", async () => {
    // Regression: the query id must come from the route, not from
    // `useActiveConversation().data.id`. If the batch-get query is still
    // loading (data === undefined) the listing must still fire — otherwise the
    // Files tab makes no `/files` call at all on cloud.
    useActiveConversationMock.mockReturnValue({ data: undefined });
    listCloudFilesMock.mockResolvedValue(["hello.txt"]);

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.data).toEqual(["hello.txt"]));
    // Falls back to the default working dir when the conversation metadata
    // (and thus its working_dir) isn't available yet.
    expect(listCloudFilesMock).toHaveBeenCalledWith(
      "conv-1",
      "/workspace/project",
    );
  });
});
