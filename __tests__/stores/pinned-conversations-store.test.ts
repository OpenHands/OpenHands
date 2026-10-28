import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getPinnedConversationsScopeKey,
  getPinnedConversationIds,
  usePinnedConversationsStore,
} from "#/stores/pinned-conversations-store";

const STORAGE_KEY = "pinned-conversations";
const BACKEND_ID = "default-local";

describe("getPinnedConversationsScopeKey", () => {
  it("keeps the bare backend id for backends without an org", () => {
    // Local backends never carry an org, and the bare id keeps pins written
    // before pins became org-aware readable.
    expect(getPinnedConversationsScopeKey("default-local")).toBe(
      "default-local",
    );
    expect(getPinnedConversationsScopeKey("default-local", null)).toBe(
      "default-local",
    );
  });

  it("separates the orgs of a cloud backend", () => {
    expect(getPinnedConversationsScopeKey("cloud-prod", "org-1")).toBe(
      "cloud-prod:org-1",
    );
    expect(getPinnedConversationsScopeKey("cloud-prod", "org-2")).not.toBe(
      getPinnedConversationsScopeKey("cloud-prod", "org-1"),
    );
  });
});

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

describe("legacy Cloud pins", () => {
  const legacy = "cloud-prod";
  const scope = "cloud-prod:org-1";

  beforeEach(() => {
    window.localStorage.clear();
    usePinnedConversationsStore.setState({
      pinsByBackendId: { [legacy]: ["old-a", "old-b"] },
    });
  });

  it("preserves the old list when pinning the first scoped conversation", () => {
    usePinnedConversationsStore
      .getState()
      .pinConversation(scope, "new", legacy);
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[scope],
    ).toEqual(["new", "old-a", "old-b"]);
  });

  it("persists an unpin against the inherited list without deleting other orgs' fallback", () => {
    usePinnedConversationsStore
      .getState()
      .unpinConversation(scope, "old-a", legacy);
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[scope],
    ).toEqual(["old-b"]);
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[legacy],
    ).toEqual(["old-a", "old-b"]);
  });

  it("toggles inherited pins off rather than creating a fresh pin", () => {
    usePinnedConversationsStore.getState().togglePin(scope, "old-a", legacy);
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[scope],
    ).toEqual(["old-b"]);
  });

  it("prunes missing inherited pins into the current scope", () => {
    usePinnedConversationsStore
      .getState()
      .pruneMissingConversations(scope, ["old-b"], legacy);
    expect(
      usePinnedConversationsStore.getState().pinsByBackendId[scope],
    ).toEqual(["old-b"]);
  });
});

it("hydrates legacy Cloud pins and remembers an empty scoped override after reload", async () => {
  const legacy = "cloud-prod";
  const scope = "cloud-prod:org-1";
  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      state: { pinsByBackendId: { [legacy]: ["old-pin"] } },
      version: 0,
    }),
  );
  await usePinnedConversationsStore.persist.rehydrate();
  expect(
    getPinnedConversationIds(
      usePinnedConversationsStore.getState().pinsByBackendId,
      scope,
      legacy,
    ),
  ).toEqual(["old-pin"]);
  usePinnedConversationsStore.getState().togglePin(scope, "old-pin", legacy);
  await usePinnedConversationsStore.persist.rehydrate();
  const pins = usePinnedConversationsStore.getState().pinsByBackendId;
  expect(getPinnedConversationIds(pins, scope, legacy)).toEqual([]);
  expect(getPinnedConversationIds(pins, "cloud-prod:org-2", legacy)).toEqual([
    "old-pin",
  ]);
});
