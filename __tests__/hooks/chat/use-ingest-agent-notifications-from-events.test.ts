import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useIngestAgentNotificationsFromEvents } from "#/hooks/chat/use-ingest-agent-notifications-from-events";
import { useEventStore, type OHEvent } from "#/stores/use-event-store";
import { useAgentNotificationsStore } from "#/stores/use-agent-notifications-store";
import { AgentState } from "#/types/agent-state";

const conversationId = "conv-ingest-gate";

const agentStateMock = vi.hoisted(() => ({
  curAgentState: "awaiting_user_input" as string,
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({
    curAgentState: agentStateMock.curAgentState,
  }),
}));

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

function seedFileEditEvents() {
  useEventStore.setState({
    events: [fileEditEvent],
    eventIds: new Set(["file-edit-1"]),
    uiEvents: [fileEditEvent],
    loadedConversationId: conversationId,
  });
}

describe("useIngestAgentNotificationsFromEvents", () => {
  beforeEach(() => {
    agentStateMock.curAgentState = AgentState.AWAITING_USER_INPUT;
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
    seedFileEditEvents();

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

  it("skips heuristic ingest while the agent is still running", () => {
    agentStateMock.curAgentState = AgentState.RUNNING;
    seedFileEditEvents();

    renderHook(() =>
      useIngestAgentNotificationsFromEvents(conversationId, {
        enabled: true,
      }),
    );

    const history =
      useAgentNotificationsStore.getState().historyByConversation[
        conversationId
      ] ?? [];
    expect(history).toHaveLength(0);
  });

  it.each([
    AgentState.AWAITING_USER_INPUT,
    AgentState.FINISHED,
  ])("runs heuristic ingest when agent state is %s", (state) => {
    agentStateMock.curAgentState = state;
    seedFileEditEvents();

    renderHook(() =>
      useIngestAgentNotificationsFromEvents(conversationId, {
        enabled: true,
      }),
    );

    const history =
      useAgentNotificationsStore.getState().historyByConversation[
        conversationId
      ];
    expect(history?.some((n) => n.kind === "skill")).toBe(true);
  });
});
