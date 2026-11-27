import { describe, expect, it } from "vitest";
import { QueryClient } from "@tanstack/react-query";
import {
  updateConversationExecutionStatusInCache,
  updateConversationLlmModelInCache,
} from "./conversation-mutation-utils";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

const createConversation = (): AppConversation => ({
  id: "conversation-1",
  created_by_user_id: null,
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
  execution_status: ExecutionStatus.RUNNING,
  conversation_url: "http://localhost:3000/api/conversations/conversation-1",
  session_api_key: "session-key",
  sandbox_id: null,
  sub_conversation_ids: [],
});

describe("updateConversationExecutionStatusInCache", () => {
  it("updates the active conversation execution_status field", () => {
    const queryClient = new QueryClient();
    const conversation = createConversation();

    queryClient.setQueryData(
      ["user", "conversation", conversation.id],
      conversation,
    );

    updateConversationExecutionStatusInCache(
      queryClient,
      conversation.id,
      ExecutionStatus.PAUSED,
    );

    expect(
      queryClient.getQueryData<AppConversation | null>([
        "user",
        "conversation",
        conversation.id,
      ]),
    ).toMatchObject({
      execution_status: ExecutionStatus.PAUSED,
    });
  });
});

describe("updateConversationLlmModelInCache", () => {
  it("updates active conversation and list cache entries", () => {
    const queryClient = new QueryClient();
    const conversation = createConversation();
    const otherConversation = { ...createConversation(), id: "conversation-2" };

    queryClient.setQueryData(
      ["user", "conversation", conversation.id, "backend-1", null],
      conversation,
    );
    queryClient.setQueryData(["user", "conversations"], {
      pages: [
        {
          items: [conversation, otherConversation],
        },
      ],
    });

    updateConversationLlmModelInCache(
      queryClient,
      conversation.id,
      "anthropic/claude-haiku-4-5",
    );

    expect(
      queryClient.getQueryData<AppConversation | null>([
        "user",
        "conversation",
        conversation.id,
        "backend-1",
        null,
      ]),
    ).toMatchObject({
      llm_model: "anthropic/claude-haiku-4-5",
    });

    expect(
      queryClient.getQueryData<{
        pages: Array<{ items: AppConversation[] }>;
      }>(["user", "conversations"])?.pages[0].items,
    ).toEqual([
      expect.objectContaining({
        id: conversation.id,
        llm_model: "anthropic/claude-haiku-4-5",
      }),
      expect.objectContaining({
        id: otherConversation.id,
        llm_model: null,
      }),
    ]);
  });
});
