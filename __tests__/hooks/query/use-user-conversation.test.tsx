import { useConversationDisplayTitle } from "#/hooks/use-conversation-display-title";
import { NavigationProvider } from "#/context/navigation-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  useUserConversation,
  usePrefetchUserConversation,
} from "#/hooks/query/use-user-conversation";
import type { Backend } from "#/api/backend-registry/types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

// Mock the underlying service the hook depends on (not the hook itself).
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { batchGetAppConversations: vi.fn() },
  }),
);

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const CLOUD_CONVERSATION_ID = "conv-cloud";

function makeConversation(id: string): AppConversation {
  return {
    id,
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "Test",
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    execution_status: null,
    sandbox_status: null,
    conversation_url: "https://sandbox.example.com/api",
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
  };
}

function makeWrapper(conversationId?: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>
          <NavigationProvider
            value={{
              conversationId: conversationId ?? null,
              currentPath: `/conversations/${conversationId ?? ""}`,
              navigate: vi.fn(),
              isNavigating: false,
            }}
          >
            {children}
          </NavigationProvider>
        </ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockReset();
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockResolvedValue([makeConversation(CLOUD_CONVERSATION_ID)]);
  setRegisteredBackends([localBackend, cloudBackend]);
});

afterEach(() => {
  vi.useRealTimers();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useUserConversation — backend switch", () => {
  it("never looks up a client-only launch ID, even on an explicit refetch", async () => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    const { result } = renderHook(
      () => useUserConversation("launch-client-id"),
      { wrapper: makeWrapper() },
    );
    await act(async () => {
      await result.current.refetch();
    });
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).not.toHaveBeenCalled();
  });

  it("does not load the conversation id against a different backend after switching", async () => {
    // Arrange — the conversation is opened while the cloud backend is active.
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    const { result } = renderHook(
      () => useUserConversation(CLOUD_CONVERSATION_ID),
      { wrapper: makeWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(1);

    // Act — switch to the local backend without leaving the conversation.
    setActiveSelection({ backendId: localBackend.id });

    // Assert — the cloud id is foreign to the local backend, so the query is
    // disabled and never fetched against it (no second service call).
    await waitFor(() => expect(result.current.data).toBeUndefined());
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(1);
  });

  it("still loads the conversation when only the org changes on the same backend", async () => {
    // Arrange — opened under a cloud backend scoped to one org.
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    const { result } = renderHook(
      () => useUserConversation(CLOUD_CONVERSATION_ID),
      { wrapper: makeWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(1);

    // Act — switch org within the same backend (e.g. the cloud personal-
    // workspace self-heal), which keeps the same agent-server schema.
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-b" });

    // Assert — same backend, so the conversation is fetched again for the
    // new org rather than being disabled.
    await waitFor(() =>
      expect(
        AgentServerConversationService.batchGetAppConversations,
      ).toHaveBeenCalledTimes(2),
    );
  });

  it("waits for a Cloud org before loading a conversation", async () => {
    // Arrange — locked-cloud bootstraps with the backend selected before its
    // organization boundary has resolved the concrete workspace.
    setActiveSelection({ backendId: cloudBackend.id });
    const { result } = renderHook(
      () => useUserConversation(CLOUD_CONVERSATION_ID),
      { wrapper: makeWrapper() },
    );

    // Assert — the detail lookup does not fall back to the server's mutable
    // current_org_id while Canvas still has no explicit org scope.
    expect(result.current.fetchStatus).toBe("idle");
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).not.toHaveBeenCalled();

    // Act — once the Cloud organization boundary resolves the workspace, the
    // query can run under the org-scoped key.
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });

    await waitFor(() =>
      expect(
        AgentServerConversationService.batchGetAppConversations,
      ).toHaveBeenCalledTimes(1),
    );
  });
});

describe("selected conversation prefetch", () => {
  it("shares the request started before navigation with the conversation page", async () => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    let resolve!: (value: (AppConversation | null)[]) => void;
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockReturnValue(
      new Promise((done) => {
        resolve = done;
      }),
    );
    const wrapper = makeWrapper();
    const { result: prefetch } = renderHook(
      () => usePrefetchUserConversation(),
      { wrapper },
    );
    act(() => {
      prefetch.current(CLOUD_CONVERSATION_ID);
    });
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledOnce();
    const { result: page } = renderHook(
      () => useUserConversation(CLOUD_CONVERSATION_ID),
      { wrapper },
    );
    await act(async () => {
      resolve([makeConversation(CLOUD_CONVERSATION_ID)]);
    });
    await waitFor(() =>
      expect(page.current.data?.id).toBe(CLOUD_CONVERSATION_ID),
    );
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledOnce();
  });

  it("lets the page retry a failed early request instead of caching an empty conversation", async () => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    vi.mocked(AgentServerConversationService.batchGetAppConversations)
      .mockRejectedValueOnce(new Error("Request failed"))
      .mockResolvedValue([makeConversation(CLOUD_CONVERSATION_ID)]);
    const wrapper = makeWrapper();
    const { result: prefetch } = renderHook(
      () => usePrefetchUserConversation(),
      { wrapper },
    );
    await act(async () => {
      await prefetch.current(CLOUD_CONVERSATION_ID);
    });
    const { result: page } = renderHook(
      () => useUserConversation(CLOUD_CONVERSATION_ID),
      { wrapper },
    );
    await waitFor(() =>
      expect(page.current.data?.id).toBe(CLOUD_CONVERSATION_ID),
    );
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(2);
  });

  it("cancels a retry when the selected organization changes", async () => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    vi.mocked(AgentServerConversationService.batchGetAppConversations)
      .mockRejectedValueOnce(
        Object.assign(new Error("Unavailable"), { status: 503 }),
      )
      .mockResolvedValue([makeConversation(CLOUD_CONVERSATION_ID)]);
    const { result } = renderHook(() => usePrefetchUserConversation(), {
      wrapper: makeWrapper(),
    });
    let pending!: Promise<void>;
    act(() => {
      pending = result.current(CLOUD_CONVERSATION_ID);
    });
    await act(async () => {
      await Promise.resolve();
    });
    act(() =>
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-b" }),
    );
    await act(async () => {
      await pending;
    });
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledOnce();
    act(() =>
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" }),
    );
    await act(async () => {
      await result.current(CLOUD_CONVERSATION_ID);
    });
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(2);
  });

  it("keeps early requests scoped to the selected Cloud organization and real IDs", async () => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    const { result } = renderHook(() => usePrefetchUserConversation(), {
      wrapper: makeWrapper(),
    });
    await act(async () => {
      await result.current("launch-client-id");
      await result.current("task-pending");
    });
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).not.toHaveBeenCalled();
    await act(async () => {
      await result.current(CLOUD_CONVERSATION_ID);
    });
    act(() =>
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-b" }),
    );
    await act(async () => {
      await result.current(CLOUD_CONVERSATION_ID);
    });
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(2);
    act(() => setActiveSelection({ backendId: localBackend.id, orgId: null }));
    await act(async () => {
      await result.current("local-conversation");
    });
    expect(
      AgentServerConversationService.batchGetAppConversations,
    ).toHaveBeenCalledTimes(2);
  });
});

describe("active conversation title freshness", () => {
  it.each([false, true])(
    "shares generated titles after completion with prefetch=%s",
    async (prefetchFirst) => {
      vi.useFakeTimers();
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
      const conversation = {
        ...makeConversation(CLOUD_CONVERSATION_ID),
        created_at: new Date().toISOString(),
        title: "Conversation conv-",
        execution_status: ExecutionStatus.FINISHED,
      };
      const fetchConversation = vi.mocked(
        AgentServerConversationService.batchGetAppConversations,
      );
      fetchConversation
        .mockResolvedValueOnce([conversation])
        .mockResolvedValue([{ ...conversation, title: "Startup explained" }]);
      const wrapper = makeWrapper(CLOUD_CONVERSATION_ID);
      if (prefetchFirst) {
        const { result: prefetch } = renderHook(
          () => usePrefetchUserConversation(),
          { wrapper },
        );
        await act(async () => {
          await prefetch.current(CLOUD_CONVERSATION_ID);
        });
      }
      const { result } = renderHook(
        () => ({
          conversation: useActiveConversation(),
          getTitle: useConversationDisplayTitle(),
        }),
        { wrapper },
      );
      await act(async () => vi.advanceTimersByTimeAsync(10));
      expect(result.current.conversation.data?.title).toBe(
        "Conversation conv-",
      );
      expect(fetchConversation).toHaveBeenCalledTimes(1);
      await act(async () => vi.advanceTimersByTimeAsync(3100));
      expect(result.current.conversation.data?.title).toBe("Startup explained");
      expect(result.current.getTitle(CLOUD_CONVERSATION_ID)).toBe(
        "Startup explained",
      );
      expect(fetchConversation).toHaveBeenCalledTimes(2);
      await act(async () => vi.advanceTimersByTimeAsync(10000));
      expect(fetchConversation).toHaveBeenCalledTimes(2);
    },
  );

  it("returns to slow polling when title generation stays unavailable", async () => {
    vi.useFakeTimers();
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    const conversation = {
      ...makeConversation(CLOUD_CONVERSATION_ID),
      created_at: new Date(Date.now() - 55000).toISOString(),
      title: "Conversation conv-",
      execution_status: ExecutionStatus.FINISHED,
    };
    const fetchConversation = vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    );
    fetchConversation.mockResolvedValue([conversation]);
    renderHook(() => useActiveConversation(), {
      wrapper: makeWrapper(CLOUD_CONVERSATION_ID),
    });
    await act(async () => vi.advanceTimersByTimeAsync(10000));
    expect(fetchConversation).toHaveBeenCalledTimes(3);
    await act(async () => vi.advanceTimersByTimeAsync(10000));
    expect(fetchConversation).toHaveBeenCalledTimes(3);
  });
});
