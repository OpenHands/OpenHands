import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { NavigationProvider } from "#/context/navigation-context";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "local", kind: "local", host: "http://127.0.0.1:18000" },
    orgId: null,
  }),
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  isNoBackend: () => false,
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      searchConversations: vi.fn(async () => ({
        items: [],
        next_page_id: null,
      })),
    },
  }),
);

describe("usePaginatedConversations", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    vi.mocked(AgentServerConversationService.searchConversations).mockClear();
  });

  afterEach(() => {
    queryClient.clear();
  });

  const renderAtPath = (currentPath: string) => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <NavigationProvider
          value={{
            currentPath,
            conversationId: null,
            isNavigating: false,
            navigate: vi.fn(),
          }}
        >
          {children}
        </NavigationProvider>
      </QueryClientProvider>
    );

    return renderHook(() => usePaginatedConversations(), { wrapper });
  };

  it("polls the conversation list on non-automation routes", async () => {
    renderAtPath("/conversations");

    await waitFor(() => {
      expect(AgentServerConversationService.searchConversations).toHaveBeenCalled();
    });

    const query = queryClient
      .getQueryCache()
      .getAll()
      .find(
        (entry) =>
          entry.queryKey[0] === "user" && entry.queryKey[1] === "conversations",
      );

    const options = query?.options as {
      refetchInterval?: number | false;
      refetchIntervalInBackground?: boolean;
    };

    expect(options.refetchInterval).toBe(30_000);
    expect(options.refetchIntervalInBackground).toBe(false);
  });

  it("does not poll while the automations route is active", async () => {
    renderAtPath("/automations");

    await waitFor(() => {
      expect(AgentServerConversationService.searchConversations).toHaveBeenCalled();
    });

    const query = queryClient
      .getQueryCache()
      .getAll()
      .find(
        (entry) =>
          entry.queryKey[0] === "user" && entry.queryKey[1] === "conversations",
      );

    const options = query?.options as {
      refetchInterval?: number | false;
      refetchIntervalInBackground?: boolean;
    };

    expect(options.refetchInterval).toBe(false);
    expect(options.refetchIntervalInBackground).toBe(false);
  });
});
