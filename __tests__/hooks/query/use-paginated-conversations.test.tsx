import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { NavigationProvider } from "#/context/navigation-context";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import type { AppConversationPage } from "#/api/conversation-service/agent-server-conversation-service.types";

const { useActiveBackendMock } = vi.hoisted(() => ({
  useActiveBackendMock: vi.fn(),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      searchConversations: vi.fn(),
    },
  }),
);

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: useActiveBackendMock,
}));

const emptyPage: AppConversationPage = { items: [], next_page_id: null };

function createWrapper(currentPath: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const navigation = {
    currentPath,
    conversationId: null,
    isNavigating: false,
    navigate: () => {},
  };
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <NavigationProvider value={navigation}>{children}</NavigationProvider>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

/** Flush timers and the promise work they unblock inside act. */
async function flushAsync(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("usePaginatedConversations — interval polling", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(AgentServerConversationService.searchConversations).mockReset();
    vi.mocked(
      AgentServerConversationService.searchConversations,
    ).mockResolvedValue(emptyPage);
    useActiveBackendMock.mockReset();
    useActiveBackendMock.mockReturnValue({
      backend: { id: "test-backend", kind: "local" },
      orgId: null,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("fetches once but does not interval-poll while an automations route is open", async () => {
    // Arrange & Act — mount the sidebar list on the automations route.
    renderHook(() => usePaginatedConversations(), {
      wrapper: createWrapper("/automations"),
    });
    await flushAsync();
    await flushAsync();

    // Assert — the initial fetch happened, and a full poll window later no
    // further request has fired.
    expect(
      AgentServerConversationService.searchConversations,
    ).toHaveBeenCalledTimes(1);
    await flushAsync(31_000);
    expect(
      AgentServerConversationService.searchConversations,
    ).toHaveBeenCalledTimes(1);
  });

  it("keeps the 30s poll on routes outside the automation surface", async () => {
    // Arrange & Act
    renderHook(() => usePaginatedConversations(), {
      wrapper: createWrapper("/"),
    });
    await flushAsync();
    await flushAsync();
    expect(
      AgentServerConversationService.searchConversations,
    ).toHaveBeenCalledTimes(1);

    // Assert — one poll window later the list refreshed.
    await flushAsync(31_000);
    expect(
      AgentServerConversationService.searchConversations,
    ).toHaveBeenCalledTimes(2);
  });

  it("does not search Cloud conversations before an org is selected", async () => {
    useActiveBackendMock.mockReturnValue({
      backend: { id: "cloud-backend", kind: "cloud" },
      orgId: null,
    });

    renderHook(() => usePaginatedConversations(), {
      wrapper: createWrapper("/"),
    });
    await flushAsync();

    expect(
      AgentServerConversationService.searchConversations,
    ).not.toHaveBeenCalled();
  });

  it("fetches a fresh list after the active connection changes", async () => {
    useActiveBackendMock.mockReturnValue({
      backend: {
        id: "test-backend",
        kind: "local",
        connectionRevision: 0,
      },
      orgId: null,
    });
    const { rerender } = renderHook(() => usePaginatedConversations(), {
      wrapper: createWrapper("/"),
    });
    await flushAsync();
    await flushAsync();

    useActiveBackendMock.mockReturnValue({
      backend: {
        id: "test-backend",
        kind: "local",
        connectionRevision: 1,
      },
      orgId: null,
    });
    rerender();
    await flushAsync();
    await flushAsync();

    expect(
      AgentServerConversationService.searchConversations,
    ).toHaveBeenCalledTimes(2);
  });
});
