import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import type {
  V1AppConversation,
  V1AppConversationPage,
} from "#/api/conversation-service/v1-conversation-service.types";

// --------------------
// Helpers
// --------------------
function makeConversation(id: string): V1AppConversation {
  return {
    id,
    created_by_user_id: null,
    sandbox_id: "sandbox-1",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: `Conversation ${id}`,
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    sandbox_status: "RUNNING",
    execution_status: null,
    conversation_url: null,
    session_api_key: null,
    sub_conversation_ids: [],
  };
}

function makePage(
  ids: string[],
  nextPageId: string | null,
): V1AppConversationPage {
  return {
    items: ids.map(makeConversation),
    next_page_id: nextPageId,
  };
}

// --------------------
// Mocks
// --------------------
vi.mock("#/api/open-hands-axios", () => ({
  openHands: {
    get: vi.fn(),
  },
}));

vi.mock("#/api/conversation-service/v1-conversation-service.api");
vi.mock("#/hooks/query/use-is-authed");

// --------------------
// Tests
// --------------------
describe("usePaginatedConversations", () => {
  let queryClient: QueryClient;
  let wrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        children,
      );
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it("fetches the first page of conversations when authenticated", async () => {
    vi.mocked(useIsAuthed).mockReturnValue({
      data: true,
    } as ReturnType<typeof useIsAuthed>);

    const page = makePage(["conv-1", "conv-2"], "page-2");
    vi.mocked(V1ConversationService.searchConversations).mockResolvedValue(
      page,
    );

    const { result } = renderHook(() => usePaginatedConversations(), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    expect(V1ConversationService.searchConversations).toHaveBeenCalledWith(
      20,
      undefined,
    );
    expect(result.current.data?.pages).toHaveLength(1);
    expect(result.current.data?.pages[0].items).toHaveLength(2);
  });

  it("does not fetch when user is not authenticated", async () => {
    vi.mocked(useIsAuthed).mockReturnValue({
      data: false,
    } as ReturnType<typeof useIsAuthed>);

    const { result } = renderHook(() => usePaginatedConversations(), {
      wrapper,
    });

    // Allow any potential async fetch to trigger
    await new Promise((r) => {
      setTimeout(r, 50);
    });

    expect(V1ConversationService.searchConversations).not.toHaveBeenCalled();
    expect(result.current.data).toBeUndefined();
  });

  it("respects the custom limit parameter", async () => {
    vi.mocked(useIsAuthed).mockReturnValue({
      data: true,
    } as ReturnType<typeof useIsAuthed>);

    const page = makePage(["conv-1"], null);
    vi.mocked(V1ConversationService.searchConversations).mockResolvedValue(
      page,
    );

    const { result } = renderHook(() => usePaginatedConversations(10), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    expect(V1ConversationService.searchConversations).toHaveBeenCalledWith(
      10,
      undefined,
    );
  });

  it("sets hasNextPage to true when next_page_id is a string", async () => {
    vi.mocked(useIsAuthed).mockReturnValue({
      data: true,
    } as ReturnType<typeof useIsAuthed>);

    const page = makePage(["conv-1"], "page-2");
    vi.mocked(V1ConversationService.searchConversations).mockResolvedValue(
      page,
    );

    const { result } = renderHook(() => usePaginatedConversations(), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    expect(result.current.hasNextPage).toBe(true);
  });

  it("sets hasNextPage to false when next_page_id is null", async () => {
    vi.mocked(useIsAuthed).mockReturnValue({
      data: true,
    } as ReturnType<typeof useIsAuthed>);

    const page = makePage(["conv-1", "conv-2"], null);
    vi.mocked(V1ConversationService.searchConversations).mockResolvedValue(
      page,
    );

    const { result } = renderHook(() => usePaginatedConversations(), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    // This is the core fix: null must be converted to undefined so
    // TanStack Query correctly sets hasNextPage to false
    expect(result.current.hasNextPage).toBe(false);
  });

  it("does not trigger additional fetches when next_page_id is null", async () => {
    vi.mocked(useIsAuthed).mockReturnValue({
      data: true,
    } as ReturnType<typeof useIsAuthed>);

    const lastPage = makePage(["conv-1"], null);
    vi.mocked(V1ConversationService.searchConversations).mockResolvedValue(
      lastPage,
    );

    const { result } = renderHook(() => usePaginatedConversations(), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    // Wait to ensure no additional fetches are triggered
    await new Promise((r) => {
      setTimeout(r, 100);
    });

    // Should only have been called once (for the initial page)
    expect(V1ConversationService.searchConversations).toHaveBeenCalledTimes(1);
    expect(result.current.isFetchingNextPage).toBe(false);
  });

  it("can fetch the next page when next_page_id is provided", async () => {
    vi.mocked(useIsAuthed).mockReturnValue({
      data: true,
    } as ReturnType<typeof useIsAuthed>);

    const page1 = makePage(["conv-1", "conv-2"], "cursor-abc");
    const page2 = makePage(["conv-3"], null);

    vi.mocked(V1ConversationService.searchConversations)
      .mockResolvedValueOnce(page1)
      .mockResolvedValueOnce(page2);

    const { result } = renderHook(() => usePaginatedConversations(), {
      wrapper,
    });

    // Wait for first page
    await waitFor(() => {
      expect(result.current.data?.pages).toHaveLength(1);
    });

    expect(result.current.hasNextPage).toBe(true);

    // Fetch next page
    result.current.fetchNextPage();

    await waitFor(() => {
      expect(result.current.data?.pages).toHaveLength(2);
    });

    // Second call should use the cursor from the first page
    expect(V1ConversationService.searchConversations).toHaveBeenCalledWith(
      20,
      "cursor-abc",
    );

    // After fetching the last page, hasNextPage should be false
    expect(result.current.hasNextPage).toBe(false);
  });
});
