import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import React from "react";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { useV1ResumeConversation } from "#/hooks/mutation/use-v1-resume-conversation";

describe("useV1ResumeConversation", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.restoreAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );

  it("invalidates sandbox and vscode_url queries on settled", async () => {
    // Mock the API calls in the mutation chain
    vi.spyOn(V1ConversationService, "batchGetAppConversations").mockResolvedValue([
      {
        id: "test-conv-id",
        created_by_user_id: null,
        sandbox_id: "test-sandbox-id",
        conversation_url: "http://localhost:3000",
        session_api_key: "test-key",
        selected_repository: null,
        selected_branch: null,
        git_provider: null,
        title: "Test",
        public: false,
        sandbox_status: "RUNNING",
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
    vi.spyOn(V1ConversationService, "resumeConversation").mockResolvedValue({
      success: true,
    });

    // Pre-populate query cache with stale sandbox data
    queryClient.setQueryData(["sandboxes", "batch", ["test-sandbox-id"]], [
      {
        sandbox_id: "test-sandbox-id",
        exposed_urls: [{ name: "VSCODE", url: "https://old-runtime.example.com" }],
      },
    ]);
    queryClient.setQueryData(["unified", "vscode_url", "test-conv-id"], {
      url: "https://old-runtime.example.com",
      error: null,
    });

    // Spy on invalidateQueries
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useV1ResumeConversation(), { wrapper });

    // Trigger the mutation
    result.current.mutate({ conversationId: "test-conv-id" });

    await waitFor(() => {
      expect(result.current.isSuccess || result.current.isError).toBe(true);
    });

    // Verify sandbox queries were invalidated
    const invalidateCalls = invalidateSpy.mock.calls.map((call) => call[0]);
    const sandboxInvalidation = invalidateCalls.find(
      (call) => call?.queryKey?.[0] === "sandboxes",
    );
    const vscodeInvalidation = invalidateCalls.find(
      (call) =>
        call?.queryKey?.[0] === "unified" &&
        call?.queryKey?.[1] === "vscode_url",
    );

    expect(sandboxInvalidation).toBeDefined();
    expect(vscodeInvalidation).toBeDefined();
  });
});
