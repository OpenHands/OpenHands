import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  clearLastConversationId,
  LAST_CONVERSATION_STORAGE_KEY,
  setLastConversationId,
} from "#/api/backend-registry/last-conversation-store";

const readStoredMap = (): Record<string, string> =>
  JSON.parse(
    window.localStorage.getItem(LAST_CONVERSATION_STORAGE_KEY) ?? "{}",
  ) as Record<string, string>;

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  window.localStorage.clear();
});

describe("last-conversation-store", () => {
  it("remembers the most recently selected conversation per backend", () => {
    setLastConversationId("backend-a", null, "convo-a-1");
    setLastConversationId("backend-b", null, "convo-b-1");

    expect(readStoredMap()).toEqual({
      "backend-a": "convo-a-1",
      "backend-b": "convo-b-1",
    });

    // The most recent selection wins…
    setLastConversationId("backend-a", null, "convo-a-2");
    // …without affecting other backends.
    expect(readStoredMap()).toEqual({
      "backend-a": "convo-a-2",
      "backend-b": "convo-b-1",
    });
  });

  it("keys cloud backends by (backendId, orgId) so each org gets its own slot", () => {
    setLastConversationId("cloud-x", "org-1", "convo-org-1");
    setLastConversationId("cloud-x", "org-2", "convo-org-2");

    expect(readStoredMap()).toEqual({
      "cloud-x::org-1": "convo-org-1",
      "cloud-x::org-2": "convo-org-2",
    });
  });

  it("clears a backend's slot without touching others", () => {
    setLastConversationId("backend-a", null, "convo-a");
    setLastConversationId("backend-b", null, "convo-b");

    clearLastConversationId("backend-a", null);

    expect(readStoredMap()).toEqual({ "backend-b": "convo-b" });
  });

  it("ignores empty conversation ids on write", () => {
    setLastConversationId("backend-a", null, "");
    expect(
      window.localStorage.getItem(LAST_CONVERSATION_STORAGE_KEY),
    ).toBeNull();
  });

  it("survives malformed JSON in storage", () => {
    window.localStorage.setItem(LAST_CONVERSATION_STORAGE_KEY, "not-json");
    // A subsequent write recovers the storage shape.
    setLastConversationId("backend-a", null, "convo-a");
    expect(readStoredMap()).toEqual({ "backend-a": "convo-a" });
  });
});
