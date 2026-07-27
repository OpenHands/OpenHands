import { describe, expect, it } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import EventService from "#/api/event-service/event-service.api";
import { TABLE_DEMO_CONVERSATION_ID } from "#/fixtures/table-demo-conversation";

describe("mock conversation handlers", () => {
  it("returns adapted conversations for batch lookups", async () => {
    const [conversation] = await AgentServerConversationService.batchGetAppConversations([
      "1",
    ]);

    expect(conversation?.id).toBe("1");
    expect(conversation?.title).toBe("My New Project");
    expect(conversation?.conversation_url).toContain("/api/conversations/1");
    expect(conversation?.workspace?.working_dir).toBe("workspace/project");
  });

  it("returns adapted conversation pages for search", async () => {
    const page = await AgentServerConversationService.searchConversations({
      limit: 10,
    });

    expect(page.items.length).toBeGreaterThan(0);
    expect(page.next_page_id).toBeNull();
    expect(page.items[0]?.title).toBeTruthy();
  });

  it("returns pre-seeded git changes for mock conversations", async () => {
    // MOCK_GIT_CHANGES is pre-seeded in git-repository-handlers.ts with three
    // representative entries (UPDATED→M, ADDED→A, DELETED→D) so mock mode
    // exercises the full diff-viewer UI without per-test manipulation.
    const changes = await AgentServerGitService.getGitChanges(
      "1",
      "http://localhost:3000/api/conversations/1",
      null,
      "workspace/project",
    );

    expect(changes).toHaveLength(3);
    expect(changes.map((c) => c.status)).toEqual(["M", "A", "D"]);
    expect(changes.map((c) => c.path)).toEqual([
      "src/components/hello.tsx",
      "src/utils/new-helper.ts",
      "src/old-module.py",
    ]);
  });

  it("returns the table demo conversation via MSW batch lookup", async () => {
    const [conversation] =
      await AgentServerConversationService.batchGetAppConversations([
        TABLE_DEMO_CONVERSATION_ID,
      ]);

    expect(conversation?.id).toBe(TABLE_DEMO_CONVERSATION_ID);
    expect(conversation?.title).toBe("Wide table demo");
  });

  it("returns table demo events sorted for conversation history", async () => {
    const page = await EventService.searchEvents(
      TABLE_DEMO_CONVERSATION_ID,
      null,
      null,
      { limit: 50, sortOrder: "TIMESTAMP_DESC" },
    );

    expect(page.items).toHaveLength(2);
    expect(page.items[0]?.source).toBe("agent");
    expect(page.items[1]?.source).toBe("user");
  });
});
