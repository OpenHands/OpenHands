import { beforeEach, describe, expect, it, vi } from "vitest";

import * as adapter from "#/api/agent-server-adapter";
import * as serializer from "#/api/conversation-service/conversation-serializer";
import * as tagKeys from "#/api/conversation-service/conversation-tag-keys";
import type { DirectConversationInfo } from "#/api/conversation-service/conversation-serializer";

const { mockGetEffectiveLocalBackend } = vi.hoisted(() => ({
  mockGetEffectiveLocalBackend: vi.fn(),
}));

// Spread the originals: both modules have a wide export surface that the rest
// of the graph still imports, so only the two readers the serializer actually
// calls are overridden.
vi.mock("#/api/backend-registry/active-store", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getEffectiveLocalBackend: mockGetEffectiveLocalBackend,
}));

vi.mock("#/api/agent-server-config", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getAgentServerWorkingDir: vi.fn(() => "/workspace/project"),
}));

beforeEach(() => {
  mockGetEffectiveLocalBackend.mockReturnValue({
    id: "default-local",
    name: "Local backend",
    host: "http://127.0.0.1:8000",
    apiKey: "session-key",
    kind: "local" as const,
  });
});

/**
 * The adapter is now a facade over the extracted modules. These tests pin the
 * seam itself: that the split did not fork a second copy of anything, and that
 * existing importers still resolve. Behaviour of the moved code stays covered
 * by the pre-existing adapter and toAppConversation suites.
 */
describe("conversation-serializer seam", () => {
  it("exposes the serializer surface directly", () => {
    expect(typeof serializer.toAppConversation).toBe("function");
    expect(typeof serializer.toConversationPage).toBe("function");
    expect(typeof serializer.toConversationUrl).toBe("function");
    expect(typeof serializer.getDefaultConversationTitle).toBe("function");
  });

  it("re-exports the identical bindings through the adapter facade", () => {
    // Identity, not just equality: a copied implementation would still be a
    // function, but would drift the moment either side changed.
    expect(adapter.toAppConversation).toBe(serializer.toAppConversation);
    expect(adapter.toConversationPage).toBe(serializer.toConversationPage);
    expect(adapter.toConversationUrl).toBe(serializer.toConversationUrl);
    expect(adapter.getDefaultConversationTitle).toBe(
      serializer.getDefaultConversationTitle,
    );
  });

  it("re-exports the tag keys through the adapter facade", () => {
    expect(adapter.ACP_SERVER_TAG_KEY).toBe(tagKeys.ACP_SERVER_TAG_KEY);
    expect(adapter.AUTOMATION_TAG_KEYS).toBe(tagKeys.AUTOMATION_TAG_KEYS);
    expect(adapter.AUTOMATION_NAME_TAG_KEY).toBe(
      tagKeys.AUTOMATION_NAME_TAG_KEY,
    );
  });

  it("keeps the wire tag key values unchanged", () => {
    // These are wire values the agent-server validates; a rename here is a
    // breaking protocol change, not a refactor.
    expect(tagKeys.ACP_SERVER_TAG_KEY).toBe("acpserver");
    expect(tagKeys.AUTOMATION_TRIGGER_TAG_KEY).toBe("automationtrigger");
    expect(tagKeys.AUTOMATION_ID_TAG_KEY).toBe("automationid");
    expect(tagKeys.AUTOMATION_NAME_TAG_KEY).toBe("automationname");
    expect(tagKeys.AUTOMATION_RUN_ID_TAG_KEY).toBe("automationrunid");
    expect([...tagKeys.AUTOMATION_TAG_KEYS]).toEqual([
      "automationtrigger",
      "automationid",
      "automationname",
      "automationrunid",
    ]);
  });
});

describe("getDefaultConversationTitle", () => {
  it("uses the first five characters of the id", () => {
    expect(serializer.getDefaultConversationTitle("abcdefghij")).toBe(
      "Conversation abcde",
    );
  });

  it("does not pad a shorter id", () => {
    expect(serializer.getDefaultConversationTitle("ab")).toBe(
      "Conversation ab",
    );
  });
});

/**
 * Direct unit coverage for the two helpers that previously had only indirect
 * coverage through the service layer — `mock-conversation-handlers.test.ts`
 * asserts the tail of `conversation_url`, and `searchConversations` drives
 * `toConversationPage`. Neither pins the host `toConversationUrl` composes, nor
 * the `?? null` fallback, so those are covered here.
 */
describe("toConversationUrl", () => {
  it("builds the url against the effective local backend host", () => {
    expect(serializer.toConversationUrl("372eb-1234")).toBe(
      "http://127.0.0.1:8000/api/conversations/372eb-1234",
    );
  });

  it("re-reads the active backend on every call rather than caching a host", () => {
    // Both calls live in this test so the assertion holds when it runs in
    // isolation (`vitest -t`), not just when a sibling test happens to run
    // first and warm the module.
    expect(serializer.toConversationUrl("abc")).toBe(
      "http://127.0.0.1:8000/api/conversations/abc",
    );

    mockGetEffectiveLocalBackend.mockReturnValue({
      id: "other",
      name: "Other backend",
      host: "http://192.168.1.5:3000",
      apiKey: null,
      kind: "local" as const,
    });

    expect(serializer.toConversationUrl("abc")).toBe(
      "http://192.168.1.5:3000/api/conversations/abc",
    );
  });
});

describe("toConversationPage", () => {
  const info = (id: string): DirectConversationInfo => ({
    id,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  });

  it("maps every item through toAppConversation", () => {
    const page = serializer.toConversationPage({
      items: [info("aaaaa-1"), info("bbbbb-2")],
      next_page_id: "cursor-2",
    });

    expect(page.items.map((item) => item.id)).toEqual(["aaaaa-1", "bbbbb-2"]);
    // Proves the mapping ran rather than passing the wire shape through.
    expect(page.items[0].title).toBe("Conversation aaaaa");
    expect(page.next_page_id).toBe("cursor-2");
  });

  it("normalises a missing or null next_page_id to null", () => {
    expect(
      serializer.toConversationPage({ items: [] }).next_page_id,
    ).toBeNull();
    expect(
      serializer.toConversationPage({ items: [], next_page_id: null })
        .next_page_id,
    ).toBeNull();
  });
});
