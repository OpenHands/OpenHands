import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import type { GitChange, GitChangeStatus } from "#/api/open-hands.types";

const route = vi.hoisted(() => ({ conversationId: "first" }));
vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => route,
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "local", kind: "local" },
    orgId: null,
  }),
}));

function renderChanges() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderHook(() => useUnifiedGetGitChanges(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ),
  });
}

describe("useUnifiedGetGitChanges", () => {
  beforeEach(() => {
    route.conversationId = "first";
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockImplementation(
      async ([id]) =>
        [
          {
            id,
            conversation_url: "http://localhost",
            session_api_key: "test",
            execution_status: "idle",
            workspace: { working_dir: "/workspace/project" },
          },
        ] as never,
    );
  });

  afterEach(() => vi.restoreAllMocks());

  it.each([
    ["M", "D"],
    ["U", "A"],
  ] as const)(
    "renders the current status after %s becomes %s",
    async (before: GitChangeStatus, after: GitChangeStatus) => {
      const getChanges = vi
        .spyOn(AgentServerGitService, "getGitChanges")
        .mockResolvedValue([{ path: "file.ts", status: before }]);
      const { result } = renderChanges();
      await waitFor(() =>
        expect(result.current.data).toEqual([
          { path: "file.ts", status: before },
        ]),
      );

      const latest: GitChange[] = [{ path: "file.ts", status: after }];
      getChanges.mockResolvedValue(latest);
      await act(async () => {
        await result.current.refetch();
      });

      await waitFor(() => expect(result.current.data).toEqual(latest));
    },
  );

  it("prepends new paths, preserves existing order and removes absent paths using current objects", async () => {
    const getChanges = vi
      .spyOn(AgentServerGitService, "getGitChanges")
      .mockResolvedValue([
        { path: "oldest.ts", status: "M" },
        { path: "existing.ts", status: "U" },
        { path: "gone.ts", status: "M" },
      ]);
    const { result } = renderChanges();
    await waitFor(() => expect(result.current.data).toHaveLength(3));

    getChanges.mockResolvedValue([
      { path: "existing.ts", status: "A" },
      { path: "new.ts", status: "A" },
      { path: "oldest.ts", status: "D" },
    ]);
    await act(async () => {
      await result.current.refetch();
    });

    await waitFor(() =>
      expect(result.current.data).toEqual([
        { path: "new.ts", status: "A" },
        { path: "oldest.ts", status: "D" },
        { path: "existing.ts", status: "A" },
      ]),
    );
  });

  it("starts with the new conversation's response order and data", async () => {
    const getChanges = vi
      .spyOn(AgentServerGitService, "getGitChanges")
      .mockResolvedValue([
        { path: "first.ts", status: "M" },
        { path: "shared.ts", status: "M" },
      ]);
    const { result, rerender } = renderChanges();
    await waitFor(() => expect(result.current.data).toHaveLength(2));
    const second: GitChange[] = [
      { path: "shared.ts", status: "D" },
      { path: "second.ts", status: "A" },
    ];
    getChanges.mockResolvedValue(second);

    route.conversationId = "second";
    rerender();

    await waitFor(() => expect(result.current.data).toEqual(second));
  });

  it("recovers from a failed initial fetch and clears the list when all changes disappear", async () => {
    const getChanges = vi
      .spyOn(AgentServerGitService, "getGitChanges")
      .mockRejectedValueOnce(new Error("runtime unavailable"))
      .mockResolvedValue([{ path: "recovered.ts", status: "M" }]);
    const { result } = renderChanges();
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.data).toEqual([]);

    await act(async () => {
      await result.current.refetch();
    });
    await waitFor(() =>
      expect(result.current.data).toEqual([
        { path: "recovered.ts", status: "M" },
      ]),
    );
    getChanges.mockResolvedValue([]);
    await act(async () => {
      await result.current.refetch();
    });

    await waitFor(() => expect(result.current.data).toEqual([]));
  });
});
