import { describe, expect, it } from "vitest";
import { QueryClient } from "@tanstack/react-query";
import { updateConversationSandboxStatusInCache } from "./conversation-mutation-utils";
import { V1ExecutionStatus } from "#/types/v1/core/base/common";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";

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
  sandbox_status: "RUNNING",
  execution_status: V1ExecutionStatus.FINISHED,
  conversation_url: "http://localhost:3000/api/conversations/conversation-1",
  session_api_key: "session-key",
  sub_conversation_ids: [],
});

describe("updateConversationSandboxStatusInCache", () => {
  it("updates the active conversation sandbox_status field", () => {
    const queryClient = new QueryClient();
    const conversation = createConversation();

    queryClient.setQueryData(
      ["user", "conversation", conversation.id],
      conversation,
    );

    updateConversationSandboxStatusInCache(
      queryClient,
      conversation.id,
      "PAUSED",
    );

    expect(
      queryClient.getQueryData<V1AppConversation | null>([
        "user",
        "conversation",
        conversation.id,
      ]),
    ).toMatchObject({
      sandbox_status: "PAUSED",
      execution_status: null,
    });
  });

  it("does not create a legacy status field on the active conversation cache", () => {
    const queryClient = new QueryClient();
    const conversation = createConversation();

    queryClient.setQueryData(
      ["user", "conversation", conversation.id],
      conversation,
    );

    updateConversationSandboxStatusInCache(
      queryClient,
      conversation.id,
      "STARTING",
    );

    const cachedConversation = queryClient.getQueryData<
      V1AppConversation & { status?: string }
    >(["user", "conversation", conversation.id]);

    expect(cachedConversation?.status).toBeUndefined();
    expect(cachedConversation?.sandbox_status).toBe("STARTING");
  });
});
