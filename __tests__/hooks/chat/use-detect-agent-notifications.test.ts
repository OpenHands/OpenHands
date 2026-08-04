import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useDetectAgentNotifications } from "#/hooks/chat/use-detect-agent-notifications";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";

const conversationId = "conv-detect-test";

describe("useDetectAgentNotifications", () => {
  beforeEach(() => {
    useAgentNotificationsStore.setState({
      historyByConversation: {},
      seenByConversation: {},
    });
    useEventStore.setState({
      events: [],
      eventIds: new Set(),
      uiEvents: [],
      loadedConversationId: null,
    });
  });

  it("adds recommendations from the latest event store snapshot", () => {
    const fileEditEvent = {
      id: "file-edit-1",
      source: "agent",
      timestamp: "2026-01-01T00:00:00.000Z",
      tool_name: "file_editor",
      tool_call_id: "call-1",
      action: {
        kind: "FileEditorAction",
        command: "str_replace",
        path: "/workspace/project/src/app.ts",
        file_text: null,
        old_str: "a",
        new_str: "b",
        insert_line: null,
        view_range: null,
      },
    } as OHEvent;

    useEventStore.setState({
      events: [fileEditEvent],
      eventIds: new Set(["file-edit-1"]),
      uiEvents: [fileEditEvent],
      loadedConversationId: conversationId,
    });

    const { result } = renderHook(() =>
      useDetectAgentNotifications(conversationId),
    );

    let detectResult = { found: 0, added: 0 };
    act(() => {
      detectResult = result.current.detectNow();
    });

    expect(detectResult).toEqual({ found: 1, added: 1 });
    expect(
      useAgentNotificationsStore.getState().historyByConversation[conversationId],
    ).toHaveLength(1);
  });
});
