import { beforeEach, describe, expect, it } from "vitest";
import {
  ARCHIVED_CONVERSATIONS_STORAGE_KEY,
  useArchivedConversationsStore,
} from "#/stores/archived-conversations-store";

const BACKEND_ID = "default-local";

describe("archived-conversations store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useArchivedConversationsStore.setState({ archivesByBackendId: {} });
  });

  it("archives a conversation for a backend", () => {
    // Arrange / Act
    useArchivedConversationsStore
      .getState()
      .archiveConversation(BACKEND_ID, "conversation-a");

    // Assert
    expect(
      useArchivedConversationsStore.getState().archivesByBackendId[BACKEND_ID],
    ).toEqual(["conversation-a"]);
    expect(
      useArchivedConversationsStore
        .getState()
        .isArchived(BACKEND_ID, "conversation-a"),
    ).toBe(true);
  });

  it("does not duplicate archives for the same conversation", () => {
    useArchivedConversationsStore
      .getState()
      .archiveConversation(BACKEND_ID, "conversation-a");
    useArchivedConversationsStore
      .getState()
      .archiveConversation(BACKEND_ID, "conversation-a");

    expect(
      useArchivedConversationsStore.getState().archivesByBackendId[BACKEND_ID],
    ).toEqual(["conversation-a"]);
  });

  it("removes an archived conversation id", () => {
    useArchivedConversationsStore
      .getState()
      .archiveConversation(BACKEND_ID, "conversation-a");
    useArchivedConversationsStore
      .getState()
      .removeArchivedConversation(BACKEND_ID, "conversation-a");

    expect(
      useArchivedConversationsStore.getState().archivesByBackendId[BACKEND_ID],
    ).toEqual([]);
  });

  it("persists archives across reloads so they survive pagination", () => {
    // Archives are never pruned against the loaded pages: an archived
    // conversation that is not on the currently fetched page must stay
    // archived instead of reappearing in the list.
    useArchivedConversationsStore
      .getState()
      .archiveConversation(BACKEND_ID, "conversation-a");
    useArchivedConversationsStore
      .getState()
      .archiveConversation(BACKEND_ID, "conversation-b");

    const persisted = JSON.parse(
      window.localStorage.getItem(ARCHIVED_CONVERSATIONS_STORAGE_KEY) ?? "{}",
    );
    expect(persisted.state.archivesByBackendId[BACKEND_ID]).toEqual([
      "conversation-b",
      "conversation-a",
    ]);
  });
});
