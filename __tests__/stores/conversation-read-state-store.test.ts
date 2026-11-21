import { beforeEach, describe, expect, it } from "vitest";
import {
  useConversationReadStateStore,
  INITIAL_CONVERSATION_READ_STATE,
} from "#/stores/conversation-read-state-store";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

describe("conversation-read-state store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useConversationReadStateStore.setState(INITIAL_CONVERSATION_READ_STATE);
  });

  it("marks a conversation as read and records timestamp", () => {
    useConversationReadStateStore.getState().markAsRead("conv-1");

    const state = useConversationReadStateStore.getState();
    expect(state.lastReadAt["conv-1"]).toBeDefined();
    expect(state.manualUnread["conv-1"]).toBeUndefined();
  });

  it("marks a conversation as unread and overrides previous read state", () => {
    useConversationReadStateStore.getState().markAsRead("conv-1");
    useConversationReadStateStore.getState().markAsUnread("conv-1");

    const state = useConversationReadStateStore.getState();
    expect(state.manualUnread["conv-1"]).toBe(true);

    const isUnread = state.isConversationUnread({
      conversationId: "conv-1",
      lastUpdatedAt: new Date(Date.now() - 10000).toISOString(),
    });
    expect(isUnread).toBe(true);
  });

  it("clears manualUnread when conversation is marked as read", () => {
    useConversationReadStateStore.getState().markAsUnread("conv-1");
    expect(
      useConversationReadStateStore.getState().manualUnread["conv-1"],
    ).toBe(true);

    useConversationReadStateStore.getState().markAsRead("conv-1");
    expect(
      useConversationReadStateStore.getState().manualUnread["conv-1"],
    ).toBeUndefined();
  });

  it("marks all conversations as read", () => {
    useConversationReadStateStore.getState().markAsUnread("conv-1");
    useConversationReadStateStore.getState().markAsUnread("conv-2");

    useConversationReadStateStore
      .getState()
      .markAllAsRead(["conv-1", "conv-2", "conv-3"]);

    const state = useConversationReadStateStore.getState();
    expect(state.lastReadAt["conv-1"]).toBeDefined();
    expect(state.lastReadAt["conv-2"]).toBeDefined();
    expect(state.lastReadAt["conv-3"]).toBeDefined();
    expect(state.manualUnread["conv-1"]).toBeUndefined();
    expect(state.manualUnread["conv-2"]).toBeUndefined();
  });

  describe("isConversationUnread", () => {
    it("returns false if the conversation is currently active / open", () => {
      const isUnread = useConversationReadStateStore
        .getState()
        .isConversationUnread({
          conversationId: "conv-1",
          lastUpdatedAt: new Date().toISOString(),
          isCurrent: true,
        });
      expect(isUnread).toBe(false);
    });

    it("returns false if lastUpdatedAt is null or undefined", () => {
      const isUnread = useConversationReadStateStore
        .getState()
        .isConversationUnread({
          conversationId: "conv-1",
          lastUpdatedAt: null,
        });
      expect(isUnread).toBe(false);
    });

    it("returns false while execution is actively RUNNING (smart stream tracking)", () => {
      const isUnread = useConversationReadStateStore
        .getState()
        .isConversationUnread({
          conversationId: "conv-1",
          lastUpdatedAt: new Date().toISOString(),
          executionStatus: ExecutionStatus.RUNNING,
        });
      expect(isUnread).toBe(false);
    });

    it("returns true for a conversation that has never been read", () => {
      const isUnread = useConversationReadStateStore
        .getState()
        .isConversationUnread({
          conversationId: "conv-new",
          lastUpdatedAt: new Date().toISOString(),
        });
      expect(isUnread).toBe(true);
    });

    it("returns false when lastUpdatedAt is earlier than lastReadAt", () => {
      const pastTime = new Date(Date.now() - 50000).toISOString();
      const readTime = new Date(Date.now() - 10000).toISOString();

      useConversationReadStateStore.setState({
        lastReadAt: { "conv-1": readTime },
        manualUnread: {},
      });

      const isUnread = useConversationReadStateStore
        .getState()
        .isConversationUnread({
          conversationId: "conv-1",
          lastUpdatedAt: pastTime,
        });
      expect(isUnread).toBe(false);
    });

    it("returns true when new messages arrive after lastReadAt", () => {
      const readTime = new Date(Date.now() - 50000).toISOString();
      const updatedTime = new Date(Date.now() - 10000).toISOString();

      useConversationReadStateStore.setState({
        lastReadAt: { "conv-1": readTime },
        manualUnread: {},
      });

      const isUnread = useConversationReadStateStore
        .getState()
        .isConversationUnread({
          conversationId: "conv-1",
          lastUpdatedAt: updatedTime,
        });
      expect(isUnread).toBe(true);
    });
  });
});
