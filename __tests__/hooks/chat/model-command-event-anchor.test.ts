import { beforeEach, describe, expect, it } from "vitest";
import { getLastRenderableEventId } from "#/hooks/chat/model-command-event-anchor";
import { useEventStore } from "#/stores/use-event-store";
import {
  createOtherActionEvent,
  createPlanningFileEditorActionEvent,
  createUserMessageEvent,
} from "test-utils";

describe("getLastRenderableEventId", () => {
  beforeEach(() => {
    useEventStore.getState().clearEvents();
  });

  it("returns null when no renderable events exist", () => {
    useEventStore
      .getState()
      .addEvent("test-conversation-id", createPlanningFileEditorActionEvent("plan"));

    expect(getLastRenderableEventId("test-conversation-id")).toBeNull();
  });

  it("returns the latest renderable UI event id as a string", () => {
    useEventStore.getState().addEvent("test-conversation-id", createUserMessageEvent("message-1"));
    useEventStore
      .getState()
      .addEvent("test-conversation-id", createPlanningFileEditorActionEvent("plan"));
    useEventStore.getState().addEvent("test-conversation-id", createOtherActionEvent("action-1"));

    expect(getLastRenderableEventId("test-conversation-id")).toBe("action-1");
  });
});
