import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getPinnedConversationsScopeKey,
  usePinnedConversationsStore,
} from "#/stores/pinned-conversations-store";

const STORAGE_KEY = "pinned-conversations";
const BACKEND_ID = "default-local";
const CLOUD_BACKEND_ID = "locked-cloud";

describe("pinned-conversations store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    usePinnedConversationsStore.setState({ pinsByBackendId: {} });
  });

  describe("getPinnedConversationsScopeKey", () => {
    it("returns backendId directly when orgId is null or undefined", () => {
      expect(getPinnedConversationsScopeKey("default-local")).toBe(
        "default-local",
      );
      expect(getPinnedConversationsScopeKey("default-local", null)).toBe(
        "default-local",
      );
      expect(getPinnedConversationsScopeKey("default-local", undefined)).toBe(
        "default-local",
      );
    });

    it("returns backendId::orgId when orgId is provided", () => {
      expect(getPinnedConversationsScopeKey("locked-cloud", "org-1")).toBe(
        "locked-cloud::org-1",
      );
      expect(getPinnedConversationsScopeKey("locked-cloud", "org-2")).toBe(
        "locked-cloud::org-2",
      );
    });
  });

  it("pins a conversation at the front of the backend list", () => {
    usePinnedConversationsStore
      .getState()
      .pinConversation(BACKEND_ID, "conversation-a");
    usePinnedConversationsStore
      .getState()
      .pinConversation(BACKEND_ID, "conversation-b");

    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[BACKEND_ID],
    ).toEqual(["conversation-b", "conversation-a"]);
  });

  it("does not duplicate pins for the same conversation", () => {
    usePinnedConversationsStore
      .getState()
      .pinConversation(BACKEND_ID, "conversation-a");
    usePinnedConversationsStore
      .getState()
      .pinConversation(BACKEND_ID, "conversation-a");

    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[BACKEND_ID],
    ).toEqual(["conversation-a"]);
  });

  it("toggles pin state", () => {
    usePinnedConversationsStore
      .getState()
      .togglePin(BACKEND_ID, "conversation-a");
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[BACKEND_ID],
    ).toEqual(["conversation-a"]);

    usePinnedConversationsStore
      .getState()
      .togglePin(BACKEND_ID, "conversation-b");

    usePinnedConversationsStore
      .getState()
      .togglePin(BACKEND_ID, "conversation-a");
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[BACKEND_ID],
    ).toEqual(["conversation-b"]);
  });

  it("prunes missing conversations and persists pin order", () => {
    usePinnedConversationsStore
      .getState()
      .pinConversation(BACKEND_ID, "conversation-a");
    usePinnedConversationsStore
      .getState()
      .pinConversation(BACKEND_ID, "conversation-b");
    usePinnedConversationsStore
      .getState()
      .pruneMissingConversations(BACKEND_ID, ["conversation-b"]);

    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[BACKEND_ID],
    ).toEqual(["conversation-b"]);

    const persisted = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY) ?? "{}",
    );
    expect(persisted.state.pinsByBackendId[BACKEND_ID]).toEqual([
      "conversation-b",
    ]);
  });

  it("does not change state when unpinning or pruning already-current pins", () => {
    const initial = usePinnedConversationsStore.getState().pinsByBackendId;
    usePinnedConversationsStore
      .getState()
      .unpinConversation(BACKEND_ID, "missing");
    expect(usePinnedConversationsStore.getState().pinsByBackendId).toBe(
      initial,
    );

    usePinnedConversationsStore
      .getState()
      .pinConversation(BACKEND_ID, "conversation-a");
    const pinned = usePinnedConversationsStore.getState().pinsByBackendId;
    usePinnedConversationsStore
      .getState()
      .pruneMissingConversations(BACKEND_ID, ["conversation-a", "other"]);
    expect(usePinnedConversationsStore.getState().pinsByBackendId).toBe(pinned);
  });

  it("isolates pinned conversations between different organizations on the same cloud backend", () => {
    const scopeOrg1 = getPinnedConversationsScopeKey(CLOUD_BACKEND_ID, "org-1");
    const scopeOrg2 = getPinnedConversationsScopeKey(CLOUD_BACKEND_ID, "org-2");

    usePinnedConversationsStore
      .getState()
      .pinConversation(scopeOrg1, "conv-org-1");
    usePinnedConversationsStore
      .getState()
      .pinConversation(scopeOrg2, "conv-org-2");

    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[scopeOrg1],
    ).toEqual(["conv-org-1"]);
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[scopeOrg2],
    ).toEqual(["conv-org-2"]);

    // Pruning in org-2 does not prune pins in org-1
    usePinnedConversationsStore
      .getState()
      .pruneMissingConversations(scopeOrg2, ["conv-org-2"]);

    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[scopeOrg1],
    ).toEqual(["conv-org-1"]);
  });

  it("creates a complete fresh store with scoped persistence", async () => {
    window.localStorage.clear();
    vi.resetModules();

    try {
      const { usePinnedConversationsStore: freshStore } =
        await import("#/stores/pinned-conversations-store");

      expect(freshStore.getState().pinsByBackendId).toEqual({});
      freshStore.getState().pinConversation(BACKEND_ID, "conversation-a");

      expect(freshStore.getState().pinsByBackendId).toEqual({
        [BACKEND_ID]: ["conversation-a"],
      });
      expect(
        JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}"),
      ).toEqual({
        state: {
          pinsByBackendId: {
            [BACKEND_ID]: ["conversation-a"],
          },
        },
        version: 0,
      });
    } finally {
      window.localStorage.clear();
      vi.resetModules();
    }
  });
});
