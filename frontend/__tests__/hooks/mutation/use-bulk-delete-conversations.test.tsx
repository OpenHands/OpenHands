import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useBulkDeleteConversations } from "#/hooks/mutation/use-bulk-delete-conversations";

vi.mock("#/api/conversation-service/conversation-service.api", () => ({
  default: {
    bulkDeleteConversations: vi.fn(),
  },
}));

vi.mock("#/utils/conversation-local-storage", () => ({
  clearConversationLocalStorage: vi.fn(),
}));

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  removeConversationsFromCache: vi.fn(() => undefined),
  restoreConversationsCache: vi.fn(),
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("useBulkDeleteConversations", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should send a single batch when ≤50 IDs", async () => {
    const ids = Array.from({ length: 10 }, (_, i) => `id-${i}`);
    vi.mocked(ConversationService.bulkDeleteConversations).mockResolvedValue({
      succeeded: ids,
      failed: [],
    });

    const { result } = renderHook(() => useBulkDeleteConversations(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ conversationIds: ids });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(ConversationService.bulkDeleteConversations).toHaveBeenCalledTimes(
      1,
    );
    expect(ConversationService.bulkDeleteConversations).toHaveBeenCalledWith(
      ids,
    );
  });

  it("should chunk into multiple batches when >50 IDs", async () => {
    const ids = Array.from({ length: 120 }, (_, i) => `id-${i}`);
    vi.mocked(ConversationService.bulkDeleteConversations).mockImplementation(
      async (batch) => ({
        succeeded: batch,
        failed: [],
      }),
    );

    const { result } = renderHook(() => useBulkDeleteConversations(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ conversationIds: ids });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // 120 IDs → 3 batches: 50 + 50 + 20
    expect(ConversationService.bulkDeleteConversations).toHaveBeenCalledTimes(
      3,
    );
    expect(
      vi.mocked(ConversationService.bulkDeleteConversations).mock.calls[0][0],
    ).toHaveLength(50);
    expect(
      vi.mocked(ConversationService.bulkDeleteConversations).mock.calls[1][0],
    ).toHaveLength(50);
    expect(
      vi.mocked(ConversationService.bulkDeleteConversations).mock.calls[2][0],
    ).toHaveLength(20);

    expect(result.current.data?.succeeded).toHaveLength(120);
    expect(result.current.data?.failed).toHaveLength(0);
  });

  it("should aggregate failures across batches", async () => {
    const ids = Array.from({ length: 60 }, (_, i) => `id-${i}`);
    vi.mocked(ConversationService.bulkDeleteConversations)
      .mockResolvedValueOnce({
        succeeded: ids.slice(0, 48),
        failed: ids.slice(48, 50),
      })
      .mockResolvedValueOnce({
        succeeded: ids.slice(50, 59),
        failed: [ids[59]],
      });

    const { result } = renderHook(() => useBulkDeleteConversations(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ conversationIds: ids });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data?.succeeded).toHaveLength(57);
    expect(result.current.data?.failed).toHaveLength(3);
  });

  it("should handle errors gracefully", async () => {
    vi.mocked(ConversationService.bulkDeleteConversations).mockRejectedValue(
      new Error("Network error"),
    );

    const { result } = renderHook(() => useBulkDeleteConversations(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ conversationIds: ["id-1"] });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
  });
});
