import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConversationClient } from "@openhands/typescript-client/clients";
import { describe, expect, it, vi, beforeEach } from "vitest";
import React from "react";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { useResumeConversation } from "#/hooks/mutation/use-resume-conversation";

const { runConversationMock } = vi.hoisted(() => ({
  runConversationMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: vi.fn(function ConversationClientMock() {
    return { runConversation: runConversationMock };
  }),
}));

describe("useResumeConversation", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.restoreAllMocks();
    runConversationMock.mockReset().mockResolvedValue({ success: true });
    vi.mocked(ConversationClient).mockClear();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it("invalidates conversation queries on settled", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([
      {
        id: "test-conv-id",
        created_by_user_id: null,
        conversation_url: "http://localhost:3000",
        session_api_key: "test-key",
        sandbox_id: null,
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        title: "Test",
        public: false,
        execution_status: null,
        trigger: null,
        pr_number: [],
        llm_model: null,
        metrics: null,
        sub_conversation_ids: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useResumeConversation(), { wrapper });

    result.current.mutate({ conversationId: "test-conv-id" });

    await waitFor(() => {
      expect(result.current.isSuccess || result.current.isError).toBe(true);
    });

    const invalidateCalls = invalidateSpy.mock.calls.map((call) => call[0]);
    const conversationInvalidation = invalidateCalls.find(
      (call) =>
        call?.queryKey?.[0] === "user" &&
        call?.queryKey?.[1] === "conversation",
    );

    expect(conversationInvalidation).toBeDefined();
  });
});
