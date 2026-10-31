import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";

const { useActiveBackendMock } = vi.hoisted(() => ({
  useActiveBackendMock: vi.fn(),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { batchGetAppConversations: vi.fn() },
  }),
);

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: useActiveBackendMock,
}));

const conversation: AppConversation = {
  id: "sub-1",
  created_by_user_id: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: "Planner",
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  execution_status: null,
  sandbox_status: null,
  conversation_url: "https://sandbox.example.com/api",
  session_api_key: "old-session-key",
  sandbox_id: null,
  sub_conversation_ids: [],
};

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }
  return Wrapper;
}

describe("useSubConversations", () => {
  beforeEach(() => {
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockReset();
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockResolvedValue([conversation]);
    useActiveBackendMock.mockReset();
    useActiveBackendMock.mockReturnValue({
      backend: { id: "backend-1", kind: "local", connectionRevision: 0 },
      orgId: null,
    });
  });

  it("fetches fresh child conversations after the connection changes", async () => {
    const { result, rerender } = renderHook(
      () => useSubConversations([conversation.id]),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    useActiveBackendMock.mockReturnValue({
      backend: { id: "backend-1", kind: "local", connectionRevision: 1 },
      orgId: null,
    });
    rerender();

    await waitFor(() =>
      expect(
        AgentServerConversationService.batchGetAppConversations,
      ).toHaveBeenCalledTimes(2),
    );
  });
});
