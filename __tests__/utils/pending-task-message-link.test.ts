import { describe, expect, it, beforeEach } from "vitest";
import {
  linkPendingTaskMessages,
  matchesPendingConversationId,
  resetPendingTaskMessageLinkState,
  schedulePendingTaskMessageReassign,
  consumeScheduledPendingTaskMessageReassign,
  clearPendingTaskMessageLink,
} from "#/utils/pending-task-message-link";

describe("pending-task-message-link", () => {
  beforeEach(() => {
    resetPendingTaskMessageLinkState();
  });

  it("matches pending messages linked from a task placeholder route", () => {
    linkPendingTaskMessages("conv-real", "task-abc");

    expect(
      matchesPendingConversationId("conv-real", "task-abc"),
    ).toBe(true);
    expect(
      matchesPendingConversationId("conv-real", "conv-real"),
    ).toBe(true);
    expect(
      matchesPendingConversationId("conv-real", "task-other"),
    ).toBe(false);
  });

  it("schedules and consumes a pending reassign once for the target conversation", () => {
    schedulePendingTaskMessageReassign("task-abc", "conv-real");

    expect(consumeScheduledPendingTaskMessageReassign("conv-other")).toBeNull();
    expect(consumeScheduledPendingTaskMessageReassign("conv-real")).toEqual({
      fromConversationId: "task-abc",
      toConversationId: "conv-real",
    });
    expect(consumeScheduledPendingTaskMessageReassign("conv-real")).toBeNull();
  });

  it("clears linked task ids for a real conversation", () => {
    linkPendingTaskMessages("conv-real", "task-abc");
    clearPendingTaskMessageLink("conv-real");

    expect(
      matchesPendingConversationId("conv-real", "task-abc"),
    ).toBe(false);
  });
});
