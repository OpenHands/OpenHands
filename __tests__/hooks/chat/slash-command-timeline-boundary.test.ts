import { beforeEach, describe, expect, it } from "vitest";
import { getLastConversationTimelineEventId } from "#/hooks/chat/slash-command-timeline-boundary";
import { useEventStore } from "#/stores/use-event-store";
import {
  createPlanningFileEditorActionEvent,
  createUserMessageEvent,
} from "test-utils";

describe("getLastConversationTimelineEventId", () => {
  beforeEach(() => useEventStore.getState().clearEvents());

  it("returns null for an empty raw timeline", () => {
    expect(getLastConversationTimelineEventId()).toBeNull();
  });

  it("captures the latest raw event even when it is not renderable", () => {
    useEventStore.getState().addEvent(createUserMessageEvent("message-1"));
    useEventStore
      .getState()
      .addEvent(createPlanningFileEditorActionEvent("plan-action"));

    expect(getLastConversationTimelineEventId()).toBe("plan-action");
  });
});
