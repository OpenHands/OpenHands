import { describe, expect, it, vi, beforeEach } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { searchMatchingConversations } from "#/api/conversation-service/search-matching-conversations";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core";

const createConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => ({
  id: "1",
  title: "Alpha",
  selected_repository: null,
  git_provider: null,
  selected_branch: null,
  updated_at: "2026-01-02T00:00:00.000Z",
  created_at: "2026-01-01T00:00:00.000Z",
  execution_status: ExecutionStatus.FINISHED,
  conversation_url: null,
  created_by_user_id: "user",
  metrics: null,
  llm_model: null,
  trigger: null,
  pr_number: [],
  session_api_key: null,
  sandbox_id: null,
  sub_conversation_ids: [],
  ...overrides,
});

describe("searchMatchingConversations", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("returns a recent page when the query is empty", async () => {
    const recent = [createConversation({ id: "recent" })];
    const searchSpy = vi
      .spyOn(AgentServerConversationService, "searchConversations")
      .mockResolvedValue({ items: recent, next_page_id: null });

    await expect(searchMatchingConversations("   ")).resolves.toEqual(recent);
    expect(searchSpy).toHaveBeenCalledTimes(1);
    expect(searchSpy).toHaveBeenCalledWith({ limit: 50 });
  });

  it("paginates through the server index before filtering", async () => {
    const pageOne = [
      createConversation({ id: "1", title: "Visible in sidebar" }),
    ];
    const pageTwo = [
      createConversation({ id: "2", title: "Hidden figma export" }),
    ];

    const searchSpy = vi
      .spyOn(AgentServerConversationService, "searchConversations")
      .mockResolvedValueOnce({ items: pageOne, next_page_id: "1" })
      .mockResolvedValueOnce({ items: pageTwo, next_page_id: null });

    const results = await searchMatchingConversations("figma");

    expect(searchSpy).toHaveBeenCalledTimes(2);
    expect(results).toEqual([pageTwo[0]]);
  });
});
