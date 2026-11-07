import { beforeEach, describe, expect, it, vi } from "vitest";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";

const STORAGE_KEY = "pinned-conversations";
const BACKEND_ID = "default-local";

describe("pinned-conversations store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    usePinnedConversationsStore.setState({ pinsByBackendId: {} });
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
