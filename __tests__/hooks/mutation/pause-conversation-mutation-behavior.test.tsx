import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError } from "axios";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { usePauseConversation } from "#/hooks/mutation/use-pause-conversation";

const { pauseConversationMock } = vi.hoisted(() => ({
  pauseConversationMock: vi.fn(),
}));

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  pauseConversation: (...args: unknown[]) => pauseConversationMock(...args),
}));

const CONVERSATIONS_QUERY_KEY = ["user", "conversations"] as const;

interface SetupOptions {
  previousConversations?: unknown;
}

const setup = ({ previousConversations }: SetupOptions = {}) => {
  vi.clearAllMocks();
  pauseConversationMock.mockResolvedValue({ success: true });

  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { retry: false },
    },
  });

  if (previousConversations !== undefined) {
    queryClient.setQueryData(CONVERSATIONS_QUERY_KEY, previousConversations);
  }

  const cancelQueries = vi.spyOn(queryClient, "cancelQueries");
  const getQueryData = vi.spyOn(queryClient, "getQueryData");
  const setQueryData = vi.spyOn(queryClient, "setQueryData");
  const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  const hook = renderHook(() => usePauseConversation(), { wrapper });

  return {
    ...hook,
    cancelQueries,
    getQueryData,
    invalidateQueries,
    queryClient,
    setQueryData,
  };
};

const createDeferred = () => {
  const resolvers: { resolve?: () => void } = {};
  const promise = new Promise<void>((resolve) => {
    resolvers.resolve = resolve;
  });

  return {
    promise,
    resolve: () => resolvers.resolve?.(),
  };
};

const expectedInvalidations = (conversationId: string) => [
  [{ queryKey: ["user", "conversation", conversationId] }],
  [{ queryKey: ["user", "conversations"] }],
  [{ queryKey: ["v1-batch-get-app-conversations"] }],
];

describe("pause conversation mutation behavior", () => {
  it("waits for cancellation, snapshots the list, forwards the id, and invalidates exact caches", async () => {
    const previousConversations = {
      pages: [{ items: [{ id: "conversation-42", title: "Before" }] }],
    };
    const cancellation = createDeferred();
    const { cancelQueries, getQueryData, invalidateQueries, result } = setup({
      previousConversations,
    });
    cancelQueries.mockReturnValueOnce(cancellation.promise);
    const mutation = {
      promise: Promise.resolve<unknown>(undefined),
    };

    act(() => {
      mutation.promise = result.current.mutateAsync({
        conversationId: "conversation-42",
      });
    });

    await waitFor(() => expect(cancelQueries).toHaveBeenCalledOnce());
    expect(cancelQueries).toHaveBeenCalledWith({
      queryKey: ["user", "conversations"],
    });
    expect(getQueryData).not.toHaveBeenCalled();
    expect(pauseConversationMock).not.toHaveBeenCalled();

    cancellation.resolve();
    await act(async () => {
      await expect(mutation.promise).resolves.toEqual({ success: true });
    });

    expect(getQueryData).toHaveBeenCalledOnce();
    expect(getQueryData).toHaveBeenCalledWith(["user", "conversations"]);
    expect(pauseConversationMock).toHaveBeenCalledOnce();
    expect(pauseConversationMock).toHaveBeenCalledWith("conversation-42");
    expect(cancelQueries.mock.invocationCallOrder[0]).toBeLessThan(
      getQueryData.mock.invocationCallOrder[0]!,
    );
    expect(getQueryData.mock.invocationCallOrder[0]).toBeLessThan(
      pauseConversationMock.mock.invocationCallOrder[0]!,
    );
    expect(invalidateQueries.mock.calls).toEqual(
      expectedInvalidations("conversation-42"),
    );
  });

  it("restores the exact conversation list snapshot when pausing fails", async () => {
    const previousConversations = {
      pages: [{ items: [{ id: "conversation-1", title: "Original" }] }],
    };
    const failure = new Error("pause failed");
    const { queryClient, result, setQueryData } = setup({
      previousConversations,
    });
    pauseConversationMock.mockRejectedValueOnce(failure);

    await act(async () => {
      await expect(
        result.current.mutateAsync({ conversationId: "conversation-1" }),
      ).rejects.toBe(failure);
    });

    expect(setQueryData).toHaveBeenCalledOnce();
    expect(setQueryData).toHaveBeenCalledWith(
      ["user", "conversations"],
      previousConversations,
    );
    expect(queryClient.getQueryData(CONVERSATIONS_QUERY_KEY)).toBe(
      previousConversations,
    );
  });

  it("does not invent rollback data when no conversation list was cached", async () => {
    const failure = new Error("pause failed without cache");
    const { queryClient, result, setQueryData } = setup();
    pauseConversationMock.mockRejectedValueOnce(failure);

    await act(async () => {
      await expect(
        result.current.mutateAsync({ conversationId: "conversation-2" }),
      ).rejects.toBe(failure);
    });

    expect(pauseConversationMock).toHaveBeenCalledWith("conversation-2");
    expect(setQueryData).not.toHaveBeenCalled();
    expect(queryClient.getQueryData(CONVERSATIONS_QUERY_KEY)).toBeUndefined();
  });

  it("handles a missing optimistic context without attempting rollback", async () => {
    const { queryClient, result, setQueryData } = setup();

    await act(async () => {
      await result.current.mutateAsync({ conversationId: "conversation-3" });
    });
    vi.clearAllMocks();

    const onError = queryClient.getMutationCache().getAll().at(-1)
      ?.options.onError;

    expect(onError).toBeTypeOf("function");
    expect(() =>
      onError?.(
        new AxiosError("optimistic setup failed"),
        { conversationId: "conversation-3" },
        undefined,
        {
          client: queryClient,
          meta: undefined,
          mutationKey: undefined,
        },
      ),
    ).not.toThrow();
    expect(setQueryData).not.toHaveBeenCalled();
  });
});
