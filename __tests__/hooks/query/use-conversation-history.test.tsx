import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import {
  INITIAL_HISTORY_PAGE_SIZE,
  useConversationHistory,
} from "#/hooks/query/use-conversation-history";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import type { Conversation } from "#/api/open-hands.types";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import type { EventSearchPage } from "#/api/event-service/event-service.types";

function makeConversation(version: "V0" | "V1"): Conversation {
  return {
    conversation_id: "conv-test",
    title: "Test Conversation",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    last_updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    status: "RUNNING",
    runtime_status: null,
    url: null,
    session_api_key: null,
    conversation_version: version,
  };
}

function makeEvent(id = "evt-1", timestamp = "2024-01-01T00:00:00Z") {
  return { id, timestamp } as unknown as OpenHandsEvent;
}

function makePage(
  items: OpenHandsEvent[] = [],
  nextPageId: string | null = null,
): EventSearchPage<OpenHandsEvent> {
  return { items, next_page_id: nextPageId };
}

// --------------------
// Mocks
// --------------------
vi.mock("#/api/event-service/event-service.api");
vi.mock("#/hooks/query/use-user-conversation");

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// --------------------
// Tests
// --------------------
describe("useConversationHistory", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("requests the most recent INITIAL_HISTORY_PAGE_SIZE events sorted desc", async () => {
    const v1SearchEventsSpy = vi.spyOn(EventService, "searchEvents");

    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V1"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    v1SearchEventsSpy.mockResolvedValue(makePage([makeEvent()]));

    const { result } = renderHook(() => useConversationHistory("conv-123"), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    // Signature: (conversationId, conversationUrl, sessionApiKey, options).
    // Initial load only fetches the tail (TIMESTAMP_DESC + page-size limit)
    // so the user sees the most recent 50 events first.
    expect(EventService.searchEvents).toHaveBeenCalledWith(
      "conv-123",
      null,
      null,
      {
        limit: INITIAL_HISTORY_PAGE_SIZE,
        sortOrder: "TIMESTAMP_DESC",
      },
    );
  });

  it("returns events in chronological order even though the server returns desc", async () => {
    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V1"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    const newest = makeEvent("evt-newest", "2024-02-01T00:00:00Z");
    const middle = makeEvent("evt-middle", "2024-01-15T00:00:00Z");
    const oldest = makeEvent("evt-oldest", "2024-01-01T00:00:00Z");

    vi.spyOn(EventService, "searchEvents").mockResolvedValue(
      makePage([newest, middle, oldest], "page-2"),
    );

    const { result } = renderHook(() => useConversationHistory("conv-order"), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    expect(result.current.data?.events.map((e: any) => e.id)).toEqual([
      "evt-oldest",
      "evt-middle",
      "evt-newest",
    ]);
    expect(result.current.data?.hasMore).toBe(true);
    expect(result.current.data?.nextPageId).toBe("page-2");
  });

  it("treats a full initial page without next_page_id as having more history", async () => {
    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V1"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    vi.spyOn(EventService, "searchEvents").mockResolvedValue(
      makePage(
        Array.from({ length: INITIAL_HISTORY_PAGE_SIZE }, (_, index) =>
          makeEvent(`evt-${index}`, new Date(2024, 0, index + 1).toISOString()),
        ),
        null,
      ),
    );

    const { result } = renderHook(
      () => useConversationHistory("conv-full-page"),
      {
        wrapper,
      },
    );

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    expect(result.current.data?.hasMore).toBe(true);
  });

  it("throws a descriptive error when searchEvents returns malformed items", async () => {
    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V1"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    vi.spyOn(EventService, "searchEvents").mockResolvedValue({
      items: { bad: true },
      next_page_id: null,
    } as unknown as EventSearchPage<OpenHandsEvent>);

    const { result } = renderHook(
      () => useConversationHistory("conv-malformed"),
      { wrapper },
    );

    await waitFor(() => {
      expect(result.current.error).toBeInstanceOf(Error);
    });

    expect((result.current.error as Error).message).toBe(
      "Invalid conversation history response: expected page.items to be an array.",
    );
  });

});

describe("useConversationHistory cache key stability", () => {
  let localQueryClient: QueryClient;
  let localWrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    localQueryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    localWrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(
        QueryClientProvider,
        { client: localQueryClient },
        children,
      );
  });

  afterEach(() => {
    localQueryClient.clear();
    vi.clearAllMocks();
  });

  it("does not refetch when conversation object changes but version stays the same", async () => {
    const v1Spy = vi.spyOn(EventService, "searchEvents");
    v1Spy.mockResolvedValue(makePage([makeEvent()]));

    const conv1 = makeConversation("V1");
    vi.mocked(useUserConversation).mockReturnValue({
      data: conv1,
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    const { result, rerender } = renderHook(
      () => useConversationHistory("conv-stable"),
      { wrapper: localWrapper },
    );

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    expect(v1Spy).toHaveBeenCalledTimes(1);

    // Simulate background polling: new object reference with different mutable fields
    // but the SAME conversation_version
    const conv2: Conversation = {
      ...conv1,
      last_updated_at: "2099-01-01T00:00:00Z",
      status: "STOPPED",
      runtime_status: "STATUS$STOPPED",
    };
    vi.mocked(useUserConversation).mockReturnValue({
      data: conv2,
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    rerender();

    // Allow any potential async refetch to trigger
    await new Promise((r) => {
      setTimeout(r, 50);
    });

    // Must NOT refetch — version hasn't changed, only mutable fields did
    expect(v1Spy).toHaveBeenCalledTimes(1);

    // Note: The behavior of always using V1 API regardless of conversation_version
    // means the "version change triggers refetch" test is no longer applicable.
    // The hook now consistently uses searchEvents for all conversations.
  });

  it("refetches the tail on remount (returning to a conversation) so events produced while away load in one batched page", async () => {
    const v1Spy = vi.spyOn(EventService, "searchEvents");
    v1Spy.mockResolvedValue(makePage([makeEvent()]));

    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V1"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    const { unmount } = renderHook(
      () => useConversationHistory("conv-remount"),
      { wrapper: localWrapper },
    );
    await waitFor(() => expect(v1Spy).toHaveBeenCalledTimes(1));

    // Leaving keeps the page cached (gcTime), so returning renders instantly —
    // but it MUST refetch the tail so events the agent produced while we were
    // away (e.g. an active /goal loop emitting turns on another conversation)
    // arrive in this one batched REST page instead of being back-filled one at
    // a time over the WebSocket `since` replay.
    unmount();
    renderHook(() => useConversationHistory("conv-remount"), {
      wrapper: localWrapper,
    });
    await waitFor(() => expect(v1Spy).toHaveBeenCalledTimes(2));
  });

  it("keeps the page cached (no window-focus refetch) so the gated WebSocket doesn't churn", async () => {
    const v1Spy = vi.spyOn(EventService, "searchEvents");
    v1Spy.mockResolvedValue(makePage([makeEvent()]));

    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V1"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    const { result } = renderHook(
      () => useConversationHistory("conv-focus-check"),
      { wrapper: localWrapper },
    );

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    const queries = localQueryClient.getQueryCache().findAll({
      queryKey: ["conversation-history", "conv-focus-check"],
    });
    expect(queries).toHaveLength(1);
    const options = queries[0].options as Record<string, unknown>;
    // Cached data renders instantly on return (gcTime), the tail refetches on
    // mount, and window focus is a no-op — the socket connection is gated on
    // this query settling, so a focus refetch would drop and reconnect it.
    expect(options.gcTime).toBeGreaterThanOrEqual(30 * 60 * 1000);
    expect(options.refetchOnMount).toBe("always");
    expect(options.refetchOnWindowFocus).toBe(false);
  });

  it("has gcTime of at least 30 minutes for navigation resilience", async () => {
    const v1Spy = vi.spyOn(EventService, "searchEvents");
    v1Spy.mockResolvedValue(makePage([makeEvent()]));

    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V1"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    const { result } = renderHook(
      () => useConversationHistory("conv-gc-check"),
      { wrapper: localWrapper },
    );

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    const queries = localQueryClient.getQueryCache().findAll({
      queryKey: ["conversation-history", "conv-gc-check"],
    });
    expect(queries).toHaveLength(1);
    expect(queries[0].options.gcTime).toBeGreaterThanOrEqual(30 * 60 * 1000);
  });
});
