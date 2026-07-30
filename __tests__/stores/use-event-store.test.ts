import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import {
  EMPTY_EVENTS,
  useEventStore,
} from "#/stores/use-event-store";
import {
  ActionEvent,
  MessageEvent,
  ObservationEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";

const CONV_A = "conv-a";
const CONV_B = "conv-b";

const mockUserMessageEvent: MessageEvent = {
  id: "test-event-1",
  timestamp: Date.now().toString(),
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "Hello, world!" }],
  },
  activated_microagents: [],
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

const makeStreamingDeltaEvent = (
  id: string,
  content: string,
): StreamingDeltaEvent => ({
  id,
  timestamp: `2024-03-01T00:00:0${id.at(-1) ?? "0"}Z`,
  source: "agent",
  kind: "StreamingDeltaEvent",
  content,
  reasoning_content: null,
});

const makeUserMessageEvent = (id: string, timestamp: string): MessageEvent => ({
  ...mockUserMessageEvent,
  id,
  timestamp,
});

const bucket = (conversationId: string) =>
  useEventStore.getState().byConversation[conversationId];

describe("useEventStore", () => {
  beforeEach(() => {
    useEventStore.getState().clearEvents();
  });

  it("should render initial state correctly", () => {
    const { result } = renderHook(() => useEventStore());
    expect(result.current.byConversation).toEqual({});
  });

  it("should add an event to a conversation bucket", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(CONV_A, mockUserMessageEvent);
    });

    expect(bucket(CONV_A)?.events).toEqual([mockUserMessageEvent]);
  });

  it("should keep conversations' event streams independent", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(CONV_A, mockUserMessageEvent);
      result.current.addEvent(CONV_B, mockActionEvent);
    });

    expect(bucket(CONV_A)?.events).toEqual([mockUserMessageEvent]);
    expect(bucket(CONV_B)?.events).toEqual([mockActionEvent]);
  });

  it("should project uiEvents within a conversation bucket", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(CONV_A, mockUserMessageEvent);
      result.current.addEvent(CONV_A, mockActionEvent);
      result.current.addEvent(CONV_A, mockObservationEvent);
    });

    expect(bucket(CONV_A)?.uiEvents).toEqual([
      mockUserMessageEvent,
      mockObservationEvent,
    ]);
  });

  it("should bulk-add and sort older pages into a conversation", () => {
    const { result } = renderHook(() => useEventStore());
    const newest = makeUserMessageEvent("newest", "2024-03-03T00:00:00Z");
    const middle = makeUserMessageEvent("middle", "2024-03-02T00:00:00Z");
    const oldest = makeUserMessageEvent("oldest", "2024-03-01T00:00:00Z");

    act(() => {
      result.current.addEvent(CONV_A, newest);
      result.current.addEvents(CONV_A, [oldest, middle]);
    });

    expect(bucket(CONV_A)?.events.map((event) => event.id)).toEqual([
      "oldest",
      "middle",
      "newest",
    ]);
  });

  it("should de-duplicate events by id within a conversation", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(CONV_A, mockUserMessageEvent);
      result.current.addEvents(CONV_A, [mockUserMessageEvent, mockActionEvent]);
    });

    expect(bucket(CONV_A)?.events).toHaveLength(2);
  });

  it("should merge consecutive streaming deltas from the same sender", () => {
    const { result } = renderHook(() => useEventStore());
    const first = makeStreamingDeltaEvent("delta-1", "Hel");
    const second = makeStreamingDeltaEvent("delta-2", "lo");

    act(() => {
      result.current.addEvent(CONV_A, first);
      result.current.addEvent(CONV_A, second);
    });

    expect(bucket(CONV_A)?.events).toHaveLength(1);
    expect(bucket(CONV_A)?.events[0]).toMatchObject({
      id: "delta-1",
      content: "Hello",
    });
    expect(bucket(CONV_A)?.eventIds.has("delta-1")).toBe(true);
    expect(bucket(CONV_A)?.eventIds.has("delta-2")).toBe(true);
  });

  it("should merge streaming deltas inside a bulk add", () => {
    const { result } = renderHook(() => useEventStore());
    const first = makeStreamingDeltaEvent("delta-1", "Hel");
    const second = makeStreamingDeltaEvent("delta-2", "lo");

    act(() => {
      result.current.addEvents(CONV_A, [first, second]);
    });

    expect(bucket(CONV_A)?.events).toHaveLength(1);
    expect(bucket(CONV_A)?.events[0]).toMatchObject({
      id: "delta-1",
      content: "Hello",
    });
  });

  it("should not merge streaming deltas across conversations", () => {
    const { result } = renderHook(() => useEventStore());
    const mainDelta = makeStreamingDeltaEvent("delta-1", "Hel");
    const planningDelta = {
      ...makeStreamingDeltaEvent("delta-2", "lo"),
      isFromPlanningAgent: true,
    };

    act(() => {
      result.current.addEvent(CONV_A, mainDelta);
      result.current.addEvent(CONV_B, planningDelta);
    });

    expect(bucket(CONV_A)?.events).toEqual([mainDelta]);
    expect(bucket(CONV_B)?.events).toEqual([planningDelta]);
  });

  it("loadConversation creates an empty bucket and is idempotent on remount", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.loadConversation(CONV_A);
      result.current.addEvent(CONV_A, mockUserMessageEvent);
    });
    expect(result.current.isConversationLoaded(CONV_A)).toBe(true);
    expect(bucket(CONV_A)?.events).toHaveLength(1);

    // Remounting the same conversation must NOT wipe its events — the WS
    // provider treats an already-loaded id as a no-op.
    act(() => {
      // Simulate the provider's guard: only load when not already loaded.
      if (!result.current.isConversationLoaded(CONV_A)) {
        result.current.loadConversation(CONV_A);
      }
    });
    expect(bucket(CONV_A)?.events).toEqual([mockUserMessageEvent]);
  });

  it("clearConversation drops only the named bucket", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(CONV_A, mockUserMessageEvent);
      result.current.addEvent(CONV_B, mockActionEvent);
      result.current.clearConversation(CONV_A);
    });

    expect(bucket(CONV_A)).toBeUndefined();
    expect(bucket(CONV_B)?.events).toEqual([mockActionEvent]);
  });

  it("clearEvents drops every conversation", () => {
    const { result } = renderHook(() => useEventStore());

    act(() => {
      result.current.addEvent(CONV_A, mockUserMessageEvent);
      result.current.addEvent(CONV_B, mockActionEvent);
      result.current.clearEvents();
    });

    expect(result.current.byConversation).toEqual({});
    expect(EMPTY_EVENTS).toEqual([]);
  });
});
