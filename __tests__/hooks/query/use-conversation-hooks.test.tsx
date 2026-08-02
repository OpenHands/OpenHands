import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useConversationHooks } from "#/hooks/query/use-conversation-hooks";
import { CONVERSATION_HOOKS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { AgentState } from "#/types/agent-state";

const state = vi.hoisted(() => ({
  backendId: "backend-a",
  orgId: "org-a" as string | null,
  conversationId: "conversation-x",
  agentState: "running" as "loading" | "init" | "running",
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: state.backendId, kind: "cloud" },
    orgId: state.orgId,
  }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: state.conversationId }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: state.agentState }),
}));

const hookFor = (name: string) => ({
  event_type: name,
  matchers: [],
});

const makeHarness = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return { queryClient, wrapper };
};

describe("useConversationHooks", () => {
  beforeEach(() => {
    state.backendId = "backend-a";
    state.orgId = "org-a";
    state.conversationId = "conversation-x";
    state.agentState = AgentState.RUNNING;
    vi.restoreAllMocks();
  });

  it("isolates hooks when the organization changes on one backend", async () => {
    const getHooks = vi
      .spyOn(AgentServerConversationService, "getHooks")
      .mockImplementation(async () => ({
        hooks: [hookFor(state.orgId ?? "no-org")],
      }));
    const { queryClient, wrapper } = makeHarness();
    const { result, rerender } = renderHook(() => useConversationHooks(), {
      wrapper,
    });

    await waitFor(() =>
      expect(result.current.data?.[0]?.event_type).toBe("org-a"),
    );

    state.orgId = "org-b";
    rerender();

    await waitFor(() =>
      expect(result.current.data?.[0]?.event_type).toBe("org-b"),
    );
    expect(getHooks).toHaveBeenCalledTimes(2);
    expect(
      queryClient.getQueryData(
        CONVERSATION_HOOKS_QUERY_KEYS.detail(
          "backend-a",
          "org-a",
          "conversation-x",
        ),
      ),
    ).toEqual([hookFor("org-a")]);
    expect(
      queryClient.getQueryData(
        CONVERSATION_HOOKS_QUERY_KEYS.detail(
          "backend-a",
          "org-b",
          "conversation-x",
        ),
      ),
    ).toEqual([hookFor("org-b")]);
  });

  it("isolates hooks when the backend changes for one conversation ID", async () => {
    const getHooks = vi
      .spyOn(AgentServerConversationService, "getHooks")
      .mockImplementation(async () => ({
        hooks: [hookFor(state.backendId)],
      }));
    const { queryClient, wrapper } = makeHarness();
    const { result, rerender } = renderHook(() => useConversationHooks(), {
      wrapper,
    });

    await waitFor(() =>
      expect(result.current.data?.[0]?.event_type).toBe("backend-a"),
    );

    state.backendId = "backend-b";
    rerender();

    await waitFor(() =>
      expect(result.current.data?.[0]?.event_type).toBe("backend-b"),
    );
    expect(getHooks).toHaveBeenCalledTimes(2);
    expect(
      queryClient.getQueryData(
        CONVERSATION_HOOKS_QUERY_KEYS.detail(
          "backend-a",
          "org-a",
          "conversation-x",
        ),
      ),
    ).toEqual([hookFor("backend-a")]);
    expect(
      queryClient.getQueryData(
        CONVERSATION_HOOKS_QUERY_KEYS.detail(
          "backend-b",
          "org-a",
          "conversation-x",
        ),
      ),
    ).toEqual([hookFor("backend-b")]);
  });

  it("reuses the cache when backend, org, and conversation are unchanged", async () => {
    const getHooks = vi
      .spyOn(AgentServerConversationService, "getHooks")
      .mockResolvedValue({ hooks: [hookFor("cached")] });
    const { wrapper } = makeHarness();
    const { result, rerender } = renderHook(() => useConversationHooks(), {
      wrapper,
    });

    await waitFor(() => expect(result.current.data).toEqual([hookFor("cached")]));
    rerender();

    expect(result.current.data).toEqual([hookFor("cached")]);
    expect(getHooks).toHaveBeenCalledTimes(1);
  });

  it("waits for the agent to leave loading and initial states", async () => {
    const getHooks = vi
      .spyOn(AgentServerConversationService, "getHooks")
      .mockResolvedValue({ hooks: [] });
    state.agentState = AgentState.LOADING;
    const { wrapper } = makeHarness();
    const { rerender } = renderHook(() => useConversationHooks(), { wrapper });

    expect(getHooks).not.toHaveBeenCalled();
    state.agentState = AgentState.INIT;
    rerender();
    expect(getHooks).not.toHaveBeenCalled();

    state.agentState = AgentState.RUNNING;
    rerender();
    await waitFor(() => expect(getHooks).toHaveBeenCalledTimes(1));
  });
});
