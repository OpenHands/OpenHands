import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useEventStore } from "#/stores/use-event-store";
import {
  ActionEvent,
  MessageEvent,
  ObservationEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";

const mockUserMessageEvent: MessageEvent = {
  id: "test-event-1",
  timestamp: Date.now().toString(),
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "Hello, world!" }],
  },
  activated_skills: [],
  extended_content: [],
};

const mockActionEvent: ActionEvent = {
  id: "test-action-1",
  timestamp: Date.now().toString(),
  source: "agent",
  thought: [{ type: "text", text: "I need to execute a bash command" }],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: "echo hello",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: "call_123",
  tool_call: {
    id: "call_123",
    type: "function",
    function: {
      name: "execute_bash",
      arguments: '{"command": "echo hello"}',
    },
  },
  llm_response_id: "response_123",
  security_risk: SecurityRisk.UNKNOWN,
};

const mockObservationEvent: ObservationEvent = {
  id: "test-observation-1",
  timestamp: Date.now().toString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: "call_123",
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "hello\n" }],
    command: "echo hello",
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {
      exit_code: 0,
      pid: 12345,
      username: "user",
      hostname: "localhost",
      working_dir: "/home/user",
      py_interpreter_path: null,
      prefix: "",
      suffix: "",
    },
  },
  action_id: "test-action-1",
};

const mockAgentMessageEvent: MessageEvent = {
  ...mockUserMessageEvent,
  id: "test-agent-message-1",
  source: "agent",
  llm_message: {
    role: "assistant",
    content: [{ type: "text", text: "partial text, finished" }],
  },
};

const makeUserMessageEvent = (id: string, timestamp: string): MessageEvent => ({
  ...mockUserMessageEvent,
  id,
  timestamp,
});

describe("useEventStore", () => {
  it("should render initial state correctly", () => {
    const { result } = renderHook(() => useEventStore());
    expect(result.current.events).toEqual([]);
  });

  it("should add an event to the store", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(mockUserMessageEvent);
    });

    expect(result.current.events).toEqual([mockUserMessageEvent]);
  });

  it("should retrieve events whose actions are replaced by their observations", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(mockUserMessageEvent);
      result.current.addEvent(mockActionEvent);
      result.current.addEvent(mockObservationEvent);
    });

    expect(result.current.uiEvents).toEqual([
      mockUserMessageEvent,
      mockObservationEvent,
    ]);
  });

  it("should bulk-add events and sort them chronologically", () => {
    const { result } = renderHook(() => useEventStore());

    const newest = makeUserMessageEvent("evt-newest", "2024-03-01T00:00:00Z");
    const middle = makeUserMessageEvent("evt-middle", "2024-02-01T00:00:00Z");
    const oldest = makeUserMessageEvent("evt-oldest", "2024-01-01T00:00:00Z");

    // Seed with the newest event, then bulk-prepend older ones (the
    // pagination-on-scroll case). The store should re-sort chronologically.
    act(() => {
      result.current.addEvent(newest);
      result.current.addEvents([oldest, middle]);
    });

    expect(result.current.events.map((event) => event.id)).toEqual([
      "evt-oldest",
      "evt-middle",
      "evt-newest",
    ]);
  });

  it("prepends tied history in server order without moving existing slots or duplicating rows", () => {
    const timestamp = "2024-06-01T00:00:00Z";
    const recent = { ...makeUserMessageEvent("recent", timestamp), seq: 10 };
    const older = makeUserMessageEvent("older", timestamp);
    const oldest = makeUserMessageEvent("oldest", timestamp);
    const store = useEventStore.getState();
    store.addEvents([recent], { nextPageId: "older-page", hasMore: true });
    store.openStreamingSlot({
      type: "item_started",
      item_id: "stream",
      anchor_seq: 10,
    });
    const slot = useEventStore.getState().uiEvents[1];
    store.addEvents([oldest, older, { ...recent }], {
      nextPageId: null,
      hasMore: false,
      prepend: true,
    });
    expect(useEventStore.getState().events).toEqual([oldest, older, recent]);
    expect(useEventStore.getState().uiEvents).toEqual([
      oldest,
      older,
      recent,
      slot,
    ]);
    expect(useEventStore.getState().uiEvents[3]).toBe(slot);
    store.addEvent(makeUserMessageEvent("live", timestamp));
    expect(useEventStore.getState().uiEvents.map((event) => event.id)).toEqual([
      "oldest",
      "older",
      "recent",
      "stream",
      "live",
    ]);
  });

  it("refreshes the initial history frontier until backfill starts, independently of live or planner events", () => {
    const store = useEventStore.getState();
    const recent = makeUserMessageEvent("recent", "2024-06-01T00:00:00Z");
    store.addEvents([recent], { nextPageId: null, hasMore: false });
    store.addEvents([recent], { nextPageId: "refreshed-page", hasMore: true });
    const pagination = useEventStore.getState().historyPagination;
    expect(pagination).toMatchObject({
      nextPageId: "refreshed-page",
      hasMore: true,
    });
    store.addEvent({
      ...makeUserMessageEvent("planner", recent.timestamp),
      isFromPlanningAgent: true,
    });
    expect(useEventStore.getState().historyPagination).toBe(pagination);
    store.clearEventsForConversation("another-conversation", "another-scope");
    expect(useEventStore.getState().historyPagination).toBeNull();
  });

  it("should de-duplicate events on bulk add", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(mockUserMessageEvent);
      result.current.addEvents([mockUserMessageEvent, mockActionEvent]);
    });

    expect(result.current.events).toHaveLength(2);
  });

  it("keeps streaming slots out of the durable event log and eventIds", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(mockUserMessageEvent);
      result.current.openStreamingSlot({
        type: "item_started",
        item_id: "item-1",
      });
      for (let i = 0; i < 1000; i += 1) {
        result.current.appendStreamingDeltas([
          {
            type: "delta",
            item_id: "item-1",
            attempt: 1,
            order: i,
            content: "x",
          },
        ]);
      }
    });

    // The slot renders, but `events` and `eventIds` only ever hold durable
    // records — the slot's id IS the coming message's id, so tracking it would
    // dedupe the real message away.
    expect(result.current.events).toEqual([mockUserMessageEvent]);
    expect(result.current.eventIds.size).toBe(1);
    expect(result.current.uiEvents).toHaveLength(2);
    expect(
      (result.current.uiEvents[1] as StreamingDeltaEvent).content,
    ).toHaveLength(1000);
  });

  it("retires a slot when the durable event with its id arrives", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.openStreamingSlot({
        type: "item_started",
        item_id: mockAgentMessageEvent.id,
      });
      result.current.appendStreamingDeltas([
        {
          type: "delta",
          item_id: mockAgentMessageEvent.id,
          attempt: 1,
          order: 0,
          content: "partial",
        },
      ]);
      result.current.addEvent(mockAgentMessageEvent);
    });

    expect(result.current.uiEvents).toEqual([mockAgentMessageEvent]);
    expect(result.current.events).toEqual([mockAgentMessageEvent]);
  });

  it("clears only the addressed socket's slots", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.openStreamingSlot({
        type: "item_started",
        item_id: "main-item",
      });
      result.current.openStreamingSlot(
        { type: "item_started", item_id: "plan-item" },
        { isFromPlanningAgent: true },
      );
      result.current.clearStreamingSlots();
    });

    expect(result.current.uiEvents.map((event) => event.id)).toEqual([
      "plan-item",
    ]);

    act(() => {
      result.current.clearStreamingSlots(true);
    });
    expect(result.current.uiEvents).toEqual([]);
  });

  it("should apply action-to-observation UI replacement during bulk add", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvents([
        mockUserMessageEvent,
        mockActionEvent,
        mockObservationEvent,
      ]);
    });

    expect(result.current.uiEvents).toEqual([
      mockUserMessageEvent,
      mockObservationEvent,
    ]);
  });

  it.each([
    ["earlier", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
    ["equal", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
    ["missing", undefined, undefined],
  ])(
    "reconciles an older action across a %s timestamp page boundary",
    (_, actionTimestamp, observationTimestamp) => {
      // Also exercise the store's supported missing-timestamp fallback.
      const action = {
        ...mockActionEvent,
        timestamp: actionTimestamp,
      } as ActionEvent;
      const observation = {
        ...mockObservationEvent,
        timestamp: observationTimestamp,
      } as ObservationEvent;
      const store = useEventStore.getState();
      store.addEvents([observation], { nextPageId: "older", hasMore: true });

      store.addEvents([action, { ...observation }], {
        nextPageId: null,
        hasMore: false,
        prepend: true,
      });

      expect(useEventStore.getState().events).toEqual([action, observation]);
      expect(useEventStore.getState().uiEvents).toEqual([observation]);
      expect(useEventStore.getState().historyPagination).toEqual({
        nextPageId: null,
        hasMore: false,
        hasLoadedOlder: true,
      });
      const settled = useEventStore.getState();
      store.addEvents([action, observation]);
      expect(useEventStore.getState()).toBe(settled);
    },
  );

  it.each(["completed", "failed"] as const)(
    "keeps an ACP %s result when its older start is loaded",
    (status) => {
      const started: ACPToolCallEvent = {
        kind: "ACPToolCallEvent",
        id: "acp-started",
        timestamp: "2024-01-01T00:00:00Z",
        source: "agent",
        tool_call_id: "tool-page-boundary",
        title: "Read a file",
        tool_kind: "read",
        status: "in_progress",
        raw_input: null,
        raw_output: null,
        content: null,
        is_error: false,
      };
      const terminal = {
        ...started,
        id: "acp-terminal",
        timestamp: "2024-01-02T00:00:00Z",
        status,
        isFromPlanningAgent: true,
      };
      const store = useEventStore.getState();
      store.addEvent(terminal);

      store.addEvents([{ ...started, isFromPlanningAgent: true }]);

      expect(useEventStore.getState().uiEvents).toEqual([terminal]);
    },
  );

  it("reconciles an out-of-order single durable action", () => {
    const action = { ...mockActionEvent, timestamp: "2024-01-01T00:00:00Z" };
    const observation = {
      ...mockObservationEvent,
      timestamp: "2024-01-02T00:00:00Z",
    };
    const store = useEventStore.getState();
    const middle = makeUserMessageEvent("middle", "2024-01-01T12:00:00Z");
    store.addEvent(middle);
    store.addEvent(observation);

    store.addEvent(action);

    expect(useEventStore.getState().uiEvents).toEqual([middle, observation]);
  });

  it("preserves main and planning slots between tied rows while replaying older history", () => {
    const timestamp = "2024-01-02T00:00:00Z";
    const observation = { ...mockObservationEvent, timestamp, seq: 20 };
    const following = {
      ...makeUserMessageEvent("following", timestamp),
      seq: 21,
    };
    const planner = {
      ...makeUserMessageEvent("planner", timestamp),
      seq: 20,
      isFromPlanningAgent: true,
    };
    const store = useEventStore.getState();
    store.addEvents([observation, following, planner]);
    store.openStreamingSlot({
      type: "item_started",
      item_id: "main-slot",
      anchor_seq: 20,
    });
    store.appendStreamingDeltas([
      {
        type: "delta",
        item_id: "main-slot",
        content: "main partial",
        order: 0,
      },
    ]);
    store.openStreamingSlot(
      { type: "item_started", item_id: "planner-slot", anchor_seq: 20 },
      { isFromPlanningAgent: true },
    );
    store.appendStreamingDeltas(
      [
        {
          type: "delta",
          item_id: "planner-slot",
          content: "plan partial",
          kind: "reasoning",
          order: 0,
        },
      ],
      { isFromPlanningAgent: true },
    );
    const before = useEventStore.getState().uiEvents;

    store.addEvents([{ ...mockActionEvent, timestamp }], {
      nextPageId: null,
      hasMore: false,
      prepend: true,
    });

    expect(useEventStore.getState().uiEvents).toEqual(before);
    expect(useEventStore.getState().uiEvents[1]).toBe(before[1]);
    expect(useEventStore.getState().uiEvents[4]).toBe(before[4]);
    store.addEvent({
      ...mockAgentMessageEvent,
      id: "main-slot",
      timestamp: "2024-01-01T00:00:00Z",
    });
    expect(
      useEventStore
        .getState()
        .uiEvents.filter((event) => event.id === "main-slot"),
    ).toEqual([
      {
        ...mockAgentMessageEvent,
        id: "main-slot",
        timestamp: "2024-01-01T00:00:00Z",
      },
    ]);
    expect(
      useEventStore
        .getState()
        .uiEvents.find((event) => event.id === "planner-slot"),
    ).toBe(before[4]);
  });

  it("should clear all events when clearEvents is called", () => {
    const { result } = renderHook(() => useEventStore());

    // Add some events first
    act(() => {
      result.current.addEvent(mockUserMessageEvent);
      result.current.addEvent(mockActionEvent);
    });

    // Verify events were added
    expect(result.current.events).toHaveLength(2);
    expect(result.current.uiEvents).toHaveLength(2);

    // Clear events
    act(() => {
      result.current.clearEvents();
    });

    // Verify events were cleared
    expect(result.current.events).toEqual([]);
    expect(result.current.uiEvents).toEqual([]);
  });
});
