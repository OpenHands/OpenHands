import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useIngestAgentNotificationsFromEvents } from "#/hooks/chat/use-ingest-agent-notifications-from-events";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";
import { AgentState } from "#/types/agent-state";

const conversationId = "conv-ingest-gate";

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({
    curAgentState: AgentState.AWAITING_USER_INPUT,
  }),
}));

describe("useIngestAgentNotificationsFromEvents", () => {
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

  it("skips heuristic ingest when enabled is false", () => {
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

    renderHook(() =>
      useIngestAgentNotificationsFromEvents(conversationId, {
        enabled: false,
      }),
    );

    expect(
      useAgentNotificationsStore.getState().historyByConversation[
        conversationId
      ],
    ).toBeUndefined();
  });
});
