import type { ReactNode } from "react";
import {
  QueryClient,
  QueryClientProvider,
  type Query,
} from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { HookEvent } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useConversationHooks } from "#/hooks/query/use-conversation-hooks";
import { AgentState } from "#/types/agent-state";

const { mockUseAgentState, mockUseConversationId } = vi.hoisted(() => ({
  mockUseAgentState: vi.fn(),
  mockUseConversationId: vi.fn(),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: mockUseAgentState,
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: mockUseConversationId,
}));

interface RenderOptions {
  conversationId?: string | null;
  agentState?: AgentState;
}

function renderConversationHooks(options: RenderOptions = {}) {
  const conversationId = Object.hasOwn(options, "conversationId")
    ? options.conversationId
    : "conversation-123";
  const agentState = options.agentState ?? AgentState.RUNNING;
  mockUseConversationId.mockReturnValue({ conversationId });
  mockUseAgentState.mockReturnValue({ curAgentState: agentState });

  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  return {
    ...renderHook(() => useConversationHooks(), { wrapper }),
    queryClient,
  };
}

function getHooksQuery(
  queryClient: QueryClient,
  conversationId: string | null | undefined,
): Query {
  const query = queryClient.getQueryCache().find({
    queryKey: ["conversation", conversationId, "hooks"],
    exact: true,
  });
  expect(query).toBeDefined();
  return query!;
}

function getHooksQueryOptions(
  queryClient: QueryClient,
  conversationId: string | null | undefined,
): Record<string, unknown> {
  return getHooksQuery(queryClient, conversationId).options as Record<
    string,
    unknown
  >;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("useConversationHooks", () => {
  it.each([undefined, null, ""])(
    "does not request hooks without a conversation ID (%j)",
    (conversationId) => {
      const getHooksSpy = vi.spyOn(AgentServerConversationService, "getHooks");

      const { result, queryClient } = renderConversationHooks({
        conversationId,
      });

      expect(result.current.fetchStatus).toBe("idle");
      expect(result.current.data).toBeUndefined();
      expect(getHooksSpy).not.toHaveBeenCalled();
      expect(getHooksQueryOptions(queryClient, conversationId).enabled).toBe(
        false,
      );
    },
  );

  it("returns a descriptive error when a missing-ID query is explicitly refreshed", async () => {
    const getHooksSpy = vi.spyOn(AgentServerConversationService, "getHooks");
    const { result } = renderConversationHooks({ conversationId: undefined });

    const refetchResult = await act(() => result.current.refetch());

    expect(refetchResult.error).toBeInstanceOf(Error);
    expect((refetchResult.error as Error).message).toBe(
      "No conversation ID provided",
    );
    expect(getHooksSpy).not.toHaveBeenCalled();
  });

  it.each([AgentState.LOADING, AgentState.INIT])(
    "keeps the query disabled while the agent is %s",
    (agentState) => {
      const getHooksSpy = vi.spyOn(AgentServerConversationService, "getHooks");
      const { result, queryClient } = renderConversationHooks({ agentState });

      expect(result.current.fetchStatus).toBe("idle");
      expect(getHooksSpy).not.toHaveBeenCalled();
      expect(
        getHooksQueryOptions(queryClient, "conversation-123").enabled,
      ).toBe(false);
    },
  );

  it.each([
    AgentState.RUNNING,
    AgentState.AWAITING_USER_INPUT,
    AgentState.PAUSED,
    AgentState.STOPPED,
    AgentState.FINISHED,
    AgentState.REJECTED,
    AgentState.ERROR,
    AgentState.RATE_LIMITED,
    AgentState.AWAITING_USER_CONFIRMATION,
    AgentState.USER_CONFIRMED,
    AgentState.USER_REJECTED,
  ])("requests hooks while the agent is %s", async (agentState) => {
    const getHooksSpy = vi
      .spyOn(AgentServerConversationService, "getHooks")
      .mockResolvedValue({ hooks: [] });

    const { result, queryClient } = renderConversationHooks({ agentState });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(getHooksSpy).toHaveBeenCalledOnce();
    expect(getHooksSpy).toHaveBeenCalledWith("conversation-123");
    expect(getHooksQueryOptions(queryClient, "conversation-123").enabled).toBe(
      true,
    );
  });

  it("forwards the service's hooks array as query data", async () => {
    const hooks: HookEvent[] = [
      {
        event_type: "pre_tool_use",
        matchers: [
          {
            matcher: "Bash",
            hooks: [
              {
                type: "command",
                command: "npm test",
                timeout: 30,
                async: false,
              },
            ],
          },
        ],
      },
    ];
    vi.spyOn(AgentServerConversationService, "getHooks").mockResolvedValue({
      hooks,
    });

    const { result } = renderConversationHooks({
      conversationId: "conversation-with-hooks",
    });

    await waitFor(() => expect(result.current.data).toBe(hooks));
  });

  it("forwards service errors through the query result", async () => {
    const serviceError = new Error("hooks endpoint unavailable");
    vi.spyOn(AgentServerConversationService, "getHooks").mockRejectedValue(
      serviceError,
    );

    const { result } = renderConversationHooks({
      conversationId: "conversation-error",
    });

    await waitFor(() => expect(result.current.error).toBe(serviceError));
    expect(result.current.data).toBeUndefined();
  });

  it("uses conversation-scoped cache coordinates and documented cache lifetimes", async () => {
    vi.spyOn(AgentServerConversationService, "getHooks").mockResolvedValue({
      hooks: [],
    });
    const { result, queryClient } = renderConversationHooks({
      conversationId: "conversation-options",
      agentState: AgentState.AWAITING_USER_INPUT,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const query = getHooksQuery(queryClient, "conversation-options");
    expect(query.queryKey).toEqual([
      "conversation",
      "conversation-options",
      "hooks",
    ]);
    const options = getHooksQueryOptions(queryClient, "conversation-options");
    expect(options.staleTime).toBe(5 * 60 * 1000);
    expect(options.gcTime).toBe(15 * 60 * 1000);
  });
});
