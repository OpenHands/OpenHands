import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";
import { listCloudConversationFiles } from "#/api/cloud/conversation-service.api";

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
  useRuntimeIsReady: () => useRuntimeIsReadyMock(),
}));

const useOptionalConversationIdMock = vi.fn();
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => useOptionalConversationIdMock(),
}));

const useUnifiedGetGitChangesMock = vi.fn();
vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: () => useUnifiedGetGitChangesMock(),
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

function gitChangesResult(
  data: { status: string; path: string }[],
  isLoading = false,
) {
  return { data, isLoading };
}

beforeEach(() => {
  storeBackendKind = "local";
  useActiveConversationMock.mockReset();
  useRuntimeIsReadyMock.mockReset();
  useOptionalConversationIdMock.mockReset();
  useUnifiedGetGitChangesMock.mockReset();
  useUnifiedGetGitChangesMock.mockReturnValue(gitChangesResult([]));
  executeCommandSpy.mockReset();
  listCloudFilesMock.mockReset();

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

  it("unions the full tree with git changes and never runs bash", async () => {
    useUnifiedGetGitChangesMock.mockReturnValue(
      gitChangesResult([
        { status: "A", path: "hello.txt" },
        { status: "M", path: "src/index.ts" },
      ]),
    );
    listCloudFilesMock.mockResolvedValue([
      "hello.txt",
      "src/index.ts",
      "src/untouched.ts",
    ]);

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    // Full tree (from /files) unioned with the changed files, de-duped/sorted.
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

  it("degrades to the changed-files view when the full-tree endpoint is empty", async () => {
    // Regression guard: even if /files returns nothing (or fails), the agent's
    // created/modified files must still show — the previous behavior.
    useUnifiedGetGitChangesMock.mockReturnValue(
      gitChangesResult([
        { status: "A", path: "hello.txt" },
        { status: "D", path: "gone.txt" },
      ]),
    );
    listCloudFilesMock.mockResolvedValue([]);

    const { result } = renderHook(() => useWorkspaceFiles(), {
      wrapper: makeWrapper(),
    });

    // Deleted files are dropped (can't be opened); the added file remains.
    await waitFor(() => expect(result.current.data).toEqual(["hello.txt"]));
  });

  it("normalizes leading ./ and de-dupes across both sources", async () => {
    useUnifiedGetGitChangesMock.mockReturnValue(
      gitChangesResult([{ status: "M", path: "hello.txt" }]),
    );
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
