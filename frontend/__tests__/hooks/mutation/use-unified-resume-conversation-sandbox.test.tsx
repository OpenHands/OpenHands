import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { SandboxService } from "#/api/sandbox-service/sandbox-service.api";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";
import { useUnifiedResumeConversationSandbox } from "#/hooks/mutation/use-unified-start-conversation";

// Mock the error message store
vi.mock("#/stores/error-message-store", () => ({
  useErrorMessageStore: (
    selector: (state: { removeErrorMessage: () => void }) => unknown,
  ) => selector({ removeErrorMessage: vi.fn() }),
}));

describe("useUnifiedResumeConversationSandbox", () => {
  let queryClient: QueryClient;

  const createConversation = (): V1AppConversation => ({
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
    sandbox_status: "PAUSED",
    execution_status: null,
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    sub_conversation_ids: [],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it("marks the active conversation as STARTING while resume is pending", async () => {
    const conversation = createConversation();
    queryClient.setQueryData(
      ["user", "conversation", conversation.id],
      conversation,
    );

    vi.spyOn(
      V1ConversationService,
      "batchGetAppConversations",
    ).mockImplementation(() => new Promise(() => {}));

    const { result } = renderHook(() => useUnifiedResumeConversationSandbox(), {
      wrapper,
    });

    result.current.mutate({ conversationId: conversation.id });

    await waitFor(() => {
      expect(
        queryClient.getQueryData<V1AppConversation>([
          "user",
          "conversation",
          conversation.id,
        ]),
      ).toMatchObject({
        sandbox_status: "STARTING",
        execution_status: null,
      });
    });
  });

  it("backs off exponentially when the resume preflight is rate limited", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0);

    const rateLimitError = {
      response: {
        status: 429,
        headers: { "retry-after": "0" },
      },
    } as unknown as AxiosError;
    const batchGetAppConversations = vi
      .spyOn(V1ConversationService, "batchGetAppConversations")
      .mockRejectedValueOnce(rateLimitError)
      .mockRejectedValueOnce(rateLimitError)
      .mockResolvedValueOnce([createConversation()]);
    vi.spyOn(SandboxService, "resumeSandbox").mockResolvedValue({
      success: true,
    });

    const { result } = renderHook(() => useUnifiedResumeConversationSandbox(), {
      wrapper,
    });

    const resume = result.current.mutateAsync({
      conversationId: "test-conv-id",
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(batchGetAppConversations).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(batchGetAppConversations).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1999);
    });
    expect(batchGetAppConversations).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    await expect(resume).resolves.toEqual({ success: true });
    expect(batchGetAppConversations).toHaveBeenCalledTimes(3);
  });

  it("invalidates sandbox and vscode_url queries on settled", async () => {
    // Mock the API calls in the mutation chain
    vi.spyOn(
      V1ConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([createConversation()]);
    vi.spyOn(SandboxService, "resumeSandbox").mockResolvedValue({
      success: true,
    });

    // Pre-populate query cache with stale sandbox data
    queryClient.setQueryData(
      ["sandboxes", "batch", ["test-sandbox-id"]],
      [
        {
          sandbox_id: "test-sandbox-id",
          exposed_urls: [
            { name: "VSCODE", url: "https://old-runtime.example.com" },
          ],
        },
      ],
    );
    queryClient.setQueryData(["unified", "vscode_url", "test-conv-id"], {
      url: "https://old-runtime.example.com",
      error: null,
    });

    // Spy on invalidateQueries
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useUnifiedResumeConversationSandbox(), {
      wrapper,
    });

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
