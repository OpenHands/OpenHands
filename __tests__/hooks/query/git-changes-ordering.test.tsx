import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import type { GitChange } from "#/api/open-hands.types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";

const mocks = vi.hoisted(() => ({
  conversationId: "conversation-1" as string | undefined,
  conversation: undefined as AppConversation | undefined,
  runtimeIsReady: true,
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: mocks.conversationId }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: mocks.conversation }),
}));

vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => mocks.runtimeIsReady,
}));

const getGitChanges = vi.spyOn(AgentServerGitService, "getGitChanges");

const makeConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => ({
  id: "conversation-1",
  created_by_user_id: "user-1",
  selected_repository: "OpenHands/agent-canvas",
  selected_branch: "main",
  git_provider: "github",
  title: "Mutation-tested changes",
  trigger: "gui",
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2026-07-13T00:00:00.000Z",
  updated_at: "2026-07-13T00:00:00.000Z",
  execution_status: null,
  conversation_url: "https://runtime.example.test/conversations/conversation-1",
  session_api_key: "session-key",
  sandbox_id: "sandbox-1",
  workspace: { working_dir: "/workspace/agent-canvas" },
  sub_conversation_ids: [],
  ...overrides,
});

const makeChange = (
  path: string,
  status: GitChange["status"] = "M",
): GitChange => ({ path, status });

function makeDeferred<T>() {
  const resolver = { current: (_value: T) => {} };
  const promise = new Promise<T>((resolve) => {
    resolver.current = resolve;
  });

  return {
    promise,
    resolve: (value: T) => resolver.current(value),
  };
}

function prepareHook(
  overrides: {
    conversationId?: string | undefined;
    conversation?: AppConversation | undefined;
    runtimeIsReady?: boolean;
  } = {},
) {
  getGitChanges.mockReset();
  mocks.conversationId = Object.hasOwn(overrides, "conversationId")
    ? overrides.conversationId
    : "conversation-1";
  mocks.conversation = Object.hasOwn(overrides, "conversation")
    ? overrides.conversation
    : makeConversation();
  mocks.runtimeIsReady = overrides.runtimeIsReady ?? true;

  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: Number.POSITIVE_INFINITY,
      },
    },
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  return { queryClient, wrapper };
}

describe("git change loading and ordering", () => {
  it("loads the current workspace changes with runtime credentials", async () => {
    const changes = [
      makeChange("src/new-file.ts", "A"),
      makeChange("src/existing.ts"),
    ];
    const { wrapper } = prepareHook();
    getGitChanges.mockResolvedValue(changes);

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current).toMatchObject({
      data: changes,
      isLoading: false,
      isFetching: false,
      isSuccess: true,
      isError: false,
      error: null,
    });
    expect(result.current.refetch).toEqual(expect.any(Function));
    expect(getGitChanges).toHaveBeenCalledOnce();
    expect(getGitChanges).toHaveBeenCalledWith(
      "conversation-1",
      "https://runtime.example.test/conversations/conversation-1",
      "session-key",
      "/workspace/agent-canvas",
    );
  });

  it("derives the git path from the selected repository when the workspace is blank", async () => {
    const { wrapper } = prepareHook({
      conversation: makeConversation({
        selected_repository: "OpenHands/software-agent-sdk",
        workspace: { working_dir: "  " },
      }),
    });
    getGitChanges.mockResolvedValue([]);

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(getGitChanges).toHaveBeenCalledWith(
      "conversation-1",
      expect.any(String),
      "session-key",
      "workspace/project/software-agent-sdk",
    );
    expect(result.current.data).toEqual([]);
  });

  it("uses the default git path while conversation metadata is still unavailable", async () => {
    const { wrapper } = prepareHook({ conversation: undefined });
    getGitChanges.mockResolvedValue([makeChange("README.md")]);

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(getGitChanges).toHaveBeenCalledWith(
      "conversation-1",
      undefined,
      undefined,
      "workspace/project",
    );
  });

  it("uses the selected repository when its workspace has no working directory", async () => {
    const { wrapper } = prepareHook({
      conversation: makeConversation({
        selected_repository: "OpenHands/openhands",
        workspace: { working_dir: null },
      }),
    });
    getGitChanges.mockResolvedValue([]);

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(getGitChanges).toHaveBeenCalledWith(
      "conversation-1",
      expect.any(String),
      "session-key",
      "workspace/project/openhands",
    );
  });

  it("does not contact the backend until the runtime is ready", async () => {
    const { wrapper } = prepareHook({ runtimeIsReady: false });

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(getGitChanges).not.toHaveBeenCalled();
    expect(result.current).toMatchObject({
      data: [],
      isLoading: false,
      isFetching: false,
      isSuccess: false,
      isError: false,
    });
  });

  it("reports a clear error if a disabled query is manually run without a conversation id", async () => {
    const { wrapper } = prepareHook({ conversationId: undefined });

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await act(async () => {
      await result.current.refetch();
    });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toEqual(new Error("No conversation ID"));
    expect(getGitChanges).not.toHaveBeenCalled();
  });

  it("puts newly discovered files first and removes files absent from a refresh", async () => {
    const originalChanges = [
      makeChange("src/removed.ts", "D"),
      makeChange("src/kept.ts"),
    ];
    const refreshedChanges = [
      makeChange("src/kept.ts"),
      makeChange("src/new.ts", "A"),
    ];
    const { wrapper } = prepareHook();
    getGitChanges
      .mockResolvedValueOnce(originalChanges)
      .mockResolvedValueOnce(refreshedChanges);

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(result.current.data).toEqual(originalChanges));

    await act(async () => {
      await result.current.refetch();
    });

    await waitFor(() =>
      expect(result.current.data).toEqual([
        makeChange("src/new.ts", "A"),
        makeChange("src/kept.ts"),
      ]),
    );
    expect(getGitChanges).toHaveBeenCalledTimes(2);
  });

  it("ignores a refreshed payload when the backend returns the same array reference", async () => {
    const sharedChanges = [makeChange("src/original.ts")];
    const refresh = makeDeferred<GitChange[]>();
    const { wrapper } = prepareHook();
    getGitChanges
      .mockResolvedValueOnce(sharedChanges)
      .mockReturnValueOnce(refresh.promise);

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(result.current.data).toEqual(sharedChanges));

    sharedChanges.push(makeChange("src/mutated-in-place.ts", "A"));
    act(() => {
      void result.current.refetch();
    });
    await waitFor(() => expect(result.current.isFetching).toBe(true));
    await act(async () => {
      refresh.resolve(sharedChanges);
    });
    await waitFor(() => expect(result.current.isFetching).toBe(false));

    expect(getGitChanges).toHaveBeenCalledTimes(2);
    expect(result.current.data).toEqual([makeChange("src/original.ts")]);
  });

  it("normalizes a single change returned by a legacy backend", async () => {
    const singleChange = makeChange("legacy.txt", "U");
    const { wrapper } = prepareHook();
    getGitChanges.mockResolvedValue(
      singleChange as unknown as Awaited<
        ReturnType<typeof AgentServerGitService.getGitChanges>
      >,
    );

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual([singleChange]);
  });

  it("surfaces backend failures without retrying", async () => {
    const backendError = new Error("runtime unavailable");
    const { queryClient, wrapper } = prepareHook();
    getGitChanges.mockRejectedValue(backendError);

    const { result } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toBe(backendError);
    expect(result.current.data).toEqual([]);
    expect(getGitChanges).toHaveBeenCalledOnce();

    const query = queryClient.getQueryCache().find({
      queryKey: [
        "file_changes",
        "conversation-1",
        "https://runtime.example.test/conversations/conversation-1",
        "session-key",
        "/workspace/agent-canvas",
      ],
    });
    expect(query?.options).toMatchObject({
      retry: false,
      staleTime: 300_000,
      gcTime: 900_000,
      refetchOnMount: "always",
      meta: { disableToast: true },
    });
  });

  it("keeps query caches isolated across every runtime identity field", async () => {
    const { wrapper } = prepareHook();
    getGitChanges.mockImplementation(async () => [
      makeChange(`response-${getGitChanges.mock.calls.length}.txt`),
    ]);

    const { rerender } = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(getGitChanges).toHaveBeenCalledTimes(1));

    mocks.conversationId = "conversation-2";
    rerender();
    await waitFor(() => expect(getGitChanges).toHaveBeenCalledTimes(2));

    mocks.conversation = makeConversation({
      conversation_url: "https://runtime-2.example.test/conversations/2",
    });
    rerender();
    await waitFor(() => expect(getGitChanges).toHaveBeenCalledTimes(3));

    mocks.conversation = makeConversation({ session_api_key: "session-key-2" });
    rerender();
    await waitFor(() => expect(getGitChanges).toHaveBeenCalledTimes(4));

    mocks.conversation = makeConversation({
      workspace: { working_dir: "/workspace/other" },
    });
    rerender();
    await waitFor(() => expect(getGitChanges).toHaveBeenCalledTimes(5));

    mocks.conversation = makeConversation({
      selected_repository: "OpenHands/another-repository",
      workspace: null,
    });
    rerender();
    await waitFor(() => expect(getGitChanges).toHaveBeenCalledTimes(6));
  });

  it("refetches on remount even while the cached changes are fresh", async () => {
    const { queryClient, wrapper } = prepareHook();
    const refresh = makeDeferred<GitChange[]>();
    getGitChanges
      .mockResolvedValueOnce([makeChange("cached.ts")])
      .mockReturnValueOnce(refresh.promise);

    const firstRender = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(getGitChanges).toHaveBeenCalledOnce());
    await waitFor(() =>
      expect(firstRender.result.current.data).toEqual([
        makeChange("cached.ts"),
      ]),
    );
    firstRender.unmount();

    const secondRender = renderHook(() => useUnifiedGetGitChanges(), {
      wrapper,
    });
    await waitFor(() => expect(getGitChanges).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(secondRender.result.current.isFetching).toBe(true),
    );
    expect(secondRender.result.current.data).toEqual([]);

    await act(async () => {
      refresh.resolve([makeChange("refreshed.ts")]);
    });
    await waitFor(() =>
      expect(secondRender.result.current.data).toEqual([
        makeChange("refreshed.ts"),
      ]),
    );
    expect(
      queryClient.getQueryState([
        "file_changes",
        "conversation-1",
        "https://runtime.example.test/conversations/conversation-1",
        "session-key",
        "/workspace/agent-canvas",
      ])?.status,
    ).toBe("success");
  });
});
