import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useEventStore } from "#/stores/use-event-store";
import {
  ACPToolCallEvent,
  ActionEvent,
  MessageEvent,
  ObservationEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";

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

  it("should project one observation when an older action arrives after it (#16947)", () => {
    const { result } = renderHook(() => useEventStore());
    const action: ActionEvent = {
      ...mockActionEvent,
      id: "page-action",
      timestamp: "2024-01-01T00:00:00.000Z",
      tool_call_id: "call_page",
      tool_call: {
        ...mockActionEvent.tool_call,
        id: "call_page",
      },
    };
    const observation: ObservationEvent = {
      ...mockObservationEvent,
      id: "page-obs",
      timestamp: "2024-01-01T00:00:01.000Z",
      action_id: "page-action",
      tool_call_id: "call_page",
    };

    act(() => {
      result.current.clearEvents();
      // Newest REST page lands first and only has the observation.
      result.current.addEvent(observation);
      // Older-history pagination bulk-inserts the matching action.
      result.current.addEvents([action]);
    });

    expect(result.current.events.map((event) => event.id)).toEqual([
      "page-action",
      "page-obs",
    ]);
    expect(result.current.uiEvents.map((event) => event.id)).toEqual([
      "page-obs",
    ]);
  });

  it("should keep one ACP card when started/terminal events split across pages (#16947)", () => {
    const { result } = renderHook(() => useEventStore());
    const started: ACPToolCallEvent = {
      kind: "ACPToolCallEvent",
      id: "acp-started",
      timestamp: "2024-01-01T00:00:00.000Z",
      source: "agent",
      tool_call_id: "toolu_page",
      title: "ls",
      tool_kind: "execute",
      status: "in_progress",
      raw_input: { command: "ls" },
      raw_output: null,
      content: null,
      is_error: false,
    };
    const completed: ACPToolCallEvent = {
      ...started,
      id: "acp-completed",
      timestamp: "2024-01-01T00:00:01.000Z",
      status: "completed",
      raw_output: "ok",
    };

    act(() => {
      result.current.clearEvents();
      result.current.addEvent(completed);
      result.current.addEvents([started]);
    });

    expect(result.current.events.map((event) => event.id)).toEqual([
      "acp-started",
      "acp-completed",
    ]);
    expect(result.current.uiEvents.map((event) => event.id)).toEqual([
      "acp-completed",
    ]);
  });

  it("should keep equal-timestamp events in insertion order after a bulk prepend", () => {
    const { result } = renderHook(() => useEventStore());
    const first = makeUserMessageEvent("eq-first", "2024-01-01T00:00:00.000Z");
    const second = makeUserMessageEvent(
      "eq-second",
      "2024-01-01T00:00:00.000Z",
    );
    const newer = makeUserMessageEvent("eq-newer", "2024-01-02T00:00:00.000Z");

    act(() => {
      result.current.clearEvents();
      result.current.addEvent(newer);
      result.current.addEvents([first, second]);
    });

    expect(result.current.events.map((event) => event.id)).toEqual([
      "eq-first",
      "eq-second",
      "eq-newer",
    ]);
    expect(result.current.uiEvents.map((event) => event.id)).toEqual([
      "eq-first",
      "eq-second",
      "eq-newer",
    ]);
  });
});
