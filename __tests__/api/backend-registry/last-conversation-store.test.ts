import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  clearLastConversationId,
  getLastConversationId,
  LAST_CONVERSATION_STORAGE_KEY,
  setLastConversationId,
} from "#/api/backend-registry/last-conversation-store";

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  window.localStorage.clear();
});

describe("last-conversation-store", () => {
  it("returns null when nothing has been remembered for a backend", () => {
    expect(getLastConversationId("backend-a", null)).toBeNull();
  });

  it("remembers and reads back the most recently selected conversation per backend", () => {
    setLastConversationId("backend-a", null, "convo-a-1");
    setLastConversationId("backend-b", null, "convo-b-1");

    expect(getLastConversationId("backend-a", null)).toBe("convo-a-1");
    expect(getLastConversationId("backend-b", null)).toBe("convo-b-1");

    // The most recent selection wins.
    setLastConversationId("backend-a", null, "convo-a-2");
    expect(getLastConversationId("backend-a", null)).toBe("convo-a-2");
    // …without affecting other backends.
    expect(getLastConversationId("backend-b", null)).toBe("convo-b-1");
  });

  it("keys cloud backends by (backendId, orgId) so each org gets its own slot", () => {
    setLastConversationId("cloud-x", "org-1", "convo-org-1");
    setLastConversationId("cloud-x", "org-2", "convo-org-2");

    expect(getLastConversationId("cloud-x", "org-1")).toBe("convo-org-1");
    expect(getLastConversationId("cloud-x", "org-2")).toBe("convo-org-2");
    expect(getLastConversationId("cloud-x", null)).toBeNull();
  });

  it("clears a backend's slot without touching others", () => {
    setLastConversationId("backend-a", null, "convo-a");
    setLastConversationId("backend-b", null, "convo-b");

    clearLastConversationId("backend-a", null);

    expect(getLastConversationId("backend-a", null)).toBeNull();
    expect(getLastConversationId("backend-b", null)).toBe("convo-b");
  });

  it("ignores empty conversation ids on write", () => {
    setLastConversationId("backend-a", null, "");
    expect(getLastConversationId("backend-a", null)).toBeNull();
  });

  it("survives malformed JSON in storage", () => {
    window.localStorage.setItem(LAST_CONVERSATION_STORAGE_KEY, "not-json");
    expect(getLastConversationId("backend-a", null)).toBeNull();
    // A subsequent write recovers the storage shape.
    setLastConversationId("backend-a", null, "convo-a");
    expect(getLastConversationId("backend-a", null)).toBe("convo-a");
  });
});
