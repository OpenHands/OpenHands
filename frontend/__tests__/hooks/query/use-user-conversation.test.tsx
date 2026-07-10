import { renderHook, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError } from "axios";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";
import { useUserConversation } from "#/hooks/query/use-user-conversation";

const createConversation = (): V1AppConversation => ({
  id: "conversation-1",
  created_by_user_id: null,
  sandbox_id: "sandbox-1",
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: "Test conversation",
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2026-04-16T00:00:00Z",
  updated_at: "2026-04-16T00:00:00Z",
  sandbox_status: "PAUSED",
  execution_status: null,
  conversation_url: "http://localhost:3000/api/conversations/conversation-1",
  session_api_key: "session-key",
  sub_conversation_ids: [],
});

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("useUserConversation", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("backs off exponentially between rate-limited conversation fetches", async () => {
    const rateLimitError = {
      response: {
        status: 429,
        headers: {
          "retry-after": "1",
        },
      },
    } as unknown as AxiosError;

    const batchGetAppConversations = vi
      .spyOn(V1ConversationService, "batchGetAppConversations")
      .mockRejectedValueOnce(rateLimitError)
      .mockRejectedValueOnce(rateLimitError)
      .mockResolvedValueOnce([createConversation()]);
    vi.spyOn(Math, "random").mockReturnValue(0);

    const { result } = renderHook(() => useUserConversation("conversation-1"), {
      wrapper: createWrapper(),
    });

    expect(batchGetAppConversations).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(999);
    });
    expect(batchGetAppConversations).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });

    expect(batchGetAppConversations).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1999);
    });
    expect(batchGetAppConversations).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });

    expect(batchGetAppConversations).toHaveBeenCalledTimes(3);
    await vi.waitFor(() => {
      expect(result.current.data?.id).toBe("conversation-1");
    });
  });
});
