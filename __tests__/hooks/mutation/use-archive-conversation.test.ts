import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import React from "react";
import { useArchiveConversation } from "#/hooks/mutation/use-archive-conversation";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ARCHIVED_CONVERSATION_TAG_KEY } from "#/api/agent-server-adapter";
import { ExecutionStatus } from "#/types/agent-server/core";

const CONV_ID = "conv-archive-test";

const makeConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => ({
  id: CONV_ID,
  created_by_user_id: null,
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: "Test conversation",
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  execution_status: ExecutionStatus.FINISHED,
  conversation_url: null,
  session_api_key: null,
  sandbox_id: null,
  sandbox_status: null,
  sub_conversation_ids: [],
  tags: null,
  ...overrides,
});

let queryClient: QueryClient;

const wrapper = ({ children }: React.PropsWithChildren) =>
  React.createElement(
    QueryClientProvider,
    { client: queryClient },
    children,
  );

beforeEach(() => {
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  vi.spyOn(
    AgentServerConversationService,
    "updateConversationTags",
  ).mockResolvedValue();
});

describe("useArchiveConversation", () => {
  it("patches the paginated list cache with archived tag on success", async () => {
    queryClient.setQueryData(["user", "conversations"], {
      pages: [{ items: [makeConversation()] }],
      pageParams: [],
    });

    const { result } = renderHook(() => useArchiveConversation(), { wrapper });

    await act(async () => {
      result.current.mutate({
        conversationId: CONV_ID,
        archived: true,
        currentTags: null,
      });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const list = queryClient.getQueryData<{
      pages: Array<{ items: AppConversation[] }>;
    }>(["user", "conversations"]);
    expect(list?.pages[0]?.items[0]?.tags?.[ARCHIVED_CONVERSATION_TAG_KEY]).toBe(
      "true",
    );
  });

  it("removes archived tag from the cache on unarchive", async () => {
    queryClient.setQueryData(["user", "conversations"], {
      pages: [
        {
          items: [
            makeConversation({ tags: { [ARCHIVED_CONVERSATION_TAG_KEY]: "true", origin: "user" } }),
          ],
        },
      ],
      pageParams: [],
    });

    const { result } = renderHook(() => useArchiveConversation(), { wrapper });

    await act(async () => {
      result.current.mutate({
        conversationId: CONV_ID,
        archived: false,
        currentTags: { [ARCHIVED_CONVERSATION_TAG_KEY]: "true", origin: "user" },
      });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const list = queryClient.getQueryData<{
      pages: Array<{ items: AppConversation[] }>;
    }>(["user", "conversations"]);
    const conv = list?.pages[0]?.items[0];
    expect(conv?.tags?.[ARCHIVED_CONVERSATION_TAG_KEY]).toBeUndefined();
    expect(conv?.tags?.origin).toBe("user");
  });

  it("passes merged tags (archived + existing) to updateConversationTags", async () => {
    const updateSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTags",
    );
    const { result } = renderHook(() => useArchiveConversation(), { wrapper });

    await act(async () => {
      result.current.mutate({
        conversationId: CONV_ID,
        archived: true,
        currentTags: { origin: "automation" },
      });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(updateSpy).toHaveBeenCalledWith(CONV_ID, {
      origin: "automation",
      [ARCHIVED_CONVERSATION_TAG_KEY]: "true",
    });
  });

  it("invalidates conversations query on error", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "updateConversationTags",
    ).mockRejectedValue(new Error("network error"));
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useArchiveConversation(), { wrapper });

    await act(async () => {
      result.current.mutate({
        conversationId: CONV_ID,
        archived: true,
        currentTags: null,
      });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(invalidateSpy).toHaveBeenCalledWith(
      expect.objectContaining({ queryKey: ["user", "conversations"] }),
    );
  });
});
