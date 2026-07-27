import { describe, expect, it } from "vitest";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core";
import {
  conversationMatchesQuery,
  filterConversationsByQuery,
  getConversationSearchText,
} from "#/components/features/conversation-panel/filter-conversations-by-query";

const baseConversation: AppConversation = {
  id: "1",
  title: "Review the project",
  selected_repository: "org/repo",
  git_provider: null,
  selected_branch: "main",
  updated_at: "2026-01-01T00:00:00.000Z",
  created_at: "2026-01-01T00:00:00.000Z",
  execution_status: ExecutionStatus.FINISHED,
  conversation_url: null,
  created_by_user_id: "user",
  metrics: null,
  llm_model: "claude-haiku",
  trigger: null,
  pr_number: [],
  session_api_key: null,
  sandbox_id: null,
  sub_conversation_ids: [],
  selected_workspace: "/workspace/project",
  workspace: { working_dir: "/workspace/project" },
};

describe("filter-conversations-by-query", () => {
  it("builds searchable text from conversation fields", () => {
    expect(getConversationSearchText(baseConversation)).toContain(
      "Review the project",
    );
    expect(getConversationSearchText(baseConversation)).toContain("org/repo");
    expect(getConversationSearchText(baseConversation)).toContain("main");
    expect(getConversationSearchText(baseConversation)).toContain(
      "/workspace/project",
    );
    expect(getConversationSearchText(baseConversation)).toContain(
      "claude-haiku",
    );
  });

  it("matches any substring of the combined conversation text", () => {
    expect(conversationMatchesQuery(baseConversation, "review")).toBe(true);
    expect(conversationMatchesQuery(baseConversation, "org/repo")).toBe(true);
    expect(conversationMatchesQuery(baseConversation, "haiku")).toBe(true);
    expect(conversationMatchesQuery(baseConversation, "missing")).toBe(false);
  });

  it("returns all conversations when the query is empty", () => {
    const conversations = [baseConversation, { ...baseConversation, id: "2" }];
    expect(filterConversationsByQuery(conversations, "")).toHaveLength(2);
    expect(filterConversationsByQuery(conversations, "   ")).toHaveLength(2);
  });

  it("filters conversations by query", () => {
    const other = {
      ...baseConversation,
      id: "2",
      title: "Enable Figma export",
    };
    expect(
      filterConversationsByQuery([baseConversation, other], "figma"),
    ).toEqual([other]);
  });
});
