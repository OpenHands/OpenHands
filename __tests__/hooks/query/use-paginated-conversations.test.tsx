import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { NavigationProvider } from "#/context/navigation-context";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import type { AppConversationPage } from "#/api/conversation-service/agent-server-conversation-service.types";

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      searchConversations: vi.fn(),
    },
  }),
);

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "test-backend", kind: "local" },
    orgId: null,
  }),
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
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <NavigationProvider value={navigation}>{children}</NavigationProvider>
    </QueryClientProvider>
  );
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
});
