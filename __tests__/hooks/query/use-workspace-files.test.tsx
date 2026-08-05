import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useWorkspaceFiles } from "#/hooks/query/use-workspace-files";
import { listCloudConversationFiles } from "#/api/cloud/conversation-service.api";

const useActiveBackendMock = vi.fn();
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const useRuntimeIsReadyMock = vi.fn();
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => useRuntimeIsReadyMock(),
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
  useActiveBackendMock.mockReset();
  useActiveConversationMock.mockReset();
  useRuntimeIsReadyMock.mockReset();
  executeCommandSpy.mockReset();
  listCloudFilesMock.mockReset();

  useRuntimeIsReadyMock.mockReturnValue(true);
  useActiveConversationMock.mockReturnValue({ data: conversation });
  listCloudFilesMock.mockResolvedValue([]);
});

afterEach(() => {
  vi.clearAllMocks();
});

const makeBackend = (kind: "local" | "cloud") => ({
  backend: {
    id: "backend-id",
    name: kind === "local" ? "Local" : "Production",
    host:
      kind === "local" ? "http://127.0.0.1:8000" : "https://app.all-hands.dev",
    apiKey: "test-key",
    kind,
  },
  orgId: null,
});

describe("useWorkspaceFiles — local backend", () => {
  beforeEach(() => useActiveBackendMock.mockReturnValue(makeBackend("local")));

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
  beforeEach(() => useActiveBackendMock.mockReturnValue(makeBackend("cloud")));

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
});
