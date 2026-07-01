import { describe, expect, it } from "vitest";
import {
  ActionEvent,
  ObservationEvent,
  MessageEvent,
  SecurityRisk,
  OpenHandsEvent,
} from "#/types/v1/core";
import { ACPToolCallEvent } from "#/types/v1/core/events/acp-tool-call-event";
import { StreamingDeltaEvent } from "#/types/v1/core/events/streaming-delta-event";
import { handleEventForUI } from "#/utils/handle-event-for-ui";
import { isStreamingDeltaEvent } from "#/types/v1/type-guards";

describe("handleEventForUI", () => {
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

  const mockMessageEvent: MessageEvent = {
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

  it("should add non-observation events to the end of uiEvents", () => {
    const initialUiEvents = [mockMessageEvent];
    const result = handleEventForUI(mockActionEvent, initialUiEvents);

    expect(result).toEqual([mockMessageEvent, mockActionEvent]);
    expect(result).not.toBe(initialUiEvents); // Should return a new array
  });

  it("should replace corresponding action with observation when action exists", () => {
    const initialUiEvents = [mockMessageEvent, mockActionEvent];
    const result = handleEventForUI(mockObservationEvent, initialUiEvents);

    expect(result).toEqual([mockMessageEvent, mockObservationEvent]);
    expect(result).not.toBe(initialUiEvents); // Should return a new array
  });

  it("should add observation to end when corresponding action is not found", () => {
    const initialUiEvents = [mockMessageEvent];
    const result = handleEventForUI(mockObservationEvent, initialUiEvents);

    expect(result).toEqual([mockMessageEvent, mockObservationEvent]);
    expect(result).not.toBe(initialUiEvents); // Should return a new array
  });

  it("should handle empty uiEvents array", () => {
    const initialUiEvents: OpenHandsEvent[] = [];
    const result = handleEventForUI(mockObservationEvent, initialUiEvents);

    expect(result).toEqual([mockObservationEvent]);
    expect(result).not.toBe(initialUiEvents); // Should return a new array
  });

  it("should not mutate the original uiEvents array", () => {
    const initialUiEvents = [mockMessageEvent, mockActionEvent];
    const originalLength = initialUiEvents.length;
    const originalFirstEvent = initialUiEvents[0];

    handleEventForUI(mockObservationEvent, initialUiEvents);

    expect(initialUiEvents).toHaveLength(originalLength);
    expect(initialUiEvents[0]).toBe(originalFirstEvent);
    expect(initialUiEvents[1]).toBe(mockActionEvent); // Should not be replaced
  });

  it("should replace the correct action when multiple actions exist", () => {
    const anotherActionEvent: ActionEvent = {
      ...mockActionEvent,
      id: "test-action-2",
    };

    const initialUiEvents = [
      mockMessageEvent,
      mockActionEvent,
      anotherActionEvent,
    ];
    const result = handleEventForUI(mockObservationEvent, initialUiEvents);

    expect(result).toEqual([
      mockMessageEvent,
      mockObservationEvent,
      anotherActionEvent,
    ]);
  });

  it("should NOT replace ThinkAction with ThinkObservation", () => {
    const mockThinkAction: ActionEvent = {
      id: "test-think-action-1",
      timestamp: Date.now().toString(),
      source: "agent",
      thought: [{ type: "text", text: "I am thinking..." }],
      thinking_blocks: [],
      action: {
        kind: "ThinkAction",
        thought: "I am thinking...",
      },
      tool_name: "think",
      tool_call_id: "call_think_1",
      tool_call: {
        id: "call_think_1",
        type: "function",
        function: {
          name: "think",
          arguments: "",
        },
      },
      llm_response_id: "response_think",
      security_risk: SecurityRisk.UNKNOWN,
    };

    const mockThinkObservation: ObservationEvent = {
      id: "test-think-observation-1",
      timestamp: Date.now().toString(),
      source: "environment",
      tool_name: "think",
      tool_call_id: "call_think_1",
      observation: {
        kind: "ThinkObservation",
        content: [{ type: "text", text: "Your thought has been logged." }],
      },
      action_id: "test-think-action-1",
    };

    const initialUiEvents = [mockMessageEvent, mockThinkAction];
    const result = handleEventForUI(mockThinkObservation, initialUiEvents);

    // ThinkObservation should NOT be added - ThinkAction should remain
    expect(result).toEqual([mockMessageEvent, mockThinkAction]);
    expect(result).not.toBe(initialUiEvents);
  });

  describe("ACPToolCallEvent dedup", () => {
    const mockInProgress: ACPToolCallEvent = {
      kind: "ACPToolCallEvent",
      id: "acp-evt-1",
      timestamp: "2026-04-16T19:32:29.828069",
      source: "agent",
      tool_call_id: "toolu_ABC",
      title: "gh pr diff 490",
      tool_kind: "execute",
      status: "in_progress",
      raw_input: { command: "gh pr diff 490" },
      raw_output: null,
      content: null,
      is_error: false,
    };

    const mockCompleted: ACPToolCallEvent = {
      ...mockInProgress,
      id: "acp-evt-2",
      status: "completed",
      raw_output: "output text",
    };

    it("appends the first tool call for a new tool_call_id", () => {
      const result = handleEventForUI(mockInProgress, [mockMessageEvent]);

      expect(result).toEqual([mockMessageEvent, mockInProgress]);
    });

    it("replaces a later status event at the original position", () => {
      const result = handleEventForUI(mockCompleted, [
        mockMessageEvent,
        mockInProgress,
      ]);

      expect(result).toEqual([mockMessageEvent, mockCompleted]);
    });

    it("leaves tool calls with different tool_call_ids untouched", () => {
      const other: ACPToolCallEvent = {
        ...mockInProgress,
        id: "acp-evt-99",
        tool_call_id: "toolu_XYZ",
        title: "ls -la",
      };
      const result = handleEventForUI(mockCompleted, [
        mockMessageEvent,
        other,
        mockInProgress,
      ]);

      expect(result).toEqual([mockMessageEvent, other, mockCompleted]);
    });
  });

  it("should NOT add ThinkObservation even when ThinkAction is not found", () => {
    const mockThinkObservation: ObservationEvent = {
      id: "test-think-observation-1",
      timestamp: Date.now().toString(),
      source: "environment",
      tool_name: "think",
      tool_call_id: "call_think_1",
      observation: {
        kind: "ThinkObservation",
        content: [{ type: "text", text: "Your thought has been logged." }],
      },
      action_id: "test-think-action-not-found",
    };

    const initialUiEvents = [mockMessageEvent];
    const result = handleEventForUI(mockThinkObservation, initialUiEvents);

    // ThinkObservation should never be added to uiEvents
    expect(result).toEqual([mockMessageEvent]);
    expect(result).not.toBe(initialUiEvents);
  });
});

describe("handleEventForUI - streaming deltas", () => {
  let nextId = 0;
  const delta = (
    content: string | null,
    reasoning_content: string | null = null,
  ): StreamingDeltaEvent => {
    nextId += 1;
    return {
      id: `delta-${nextId}`,
      timestamp: `2026-06-29T00:00:0${nextId}.000Z`,
      source: "agent",
      kind: "StreamingDeltaEvent",
      content,
      reasoning_content,
    };
  };

  const userMessage: MessageEvent = {
    id: "user-1",
    timestamp: "2026-06-29T00:00:00.000Z",
    source: "user",
    llm_message: { role: "user", content: [{ type: "text", text: "hi" }] },
    activated_microagents: [],
    extended_content: [],
  };

  const assistantMessage = (text: string): MessageEvent => ({
    id: "assistant-1",
    timestamp: "2026-06-29T00:00:09.000Z",
    source: "agent",
    llm_message: { role: "assistant", content: [{ type: "text", text }] },
    activated_microagents: [],
    extended_content: [],
  });

  const stateUpdate = (id: string): OpenHandsEvent =>
    ({
      id,
      timestamp: "2026-06-29T00:00:05.500Z",
      source: "environment",
      kind: "ConversationStateUpdateEvent",
      key: "stats",
      value: {},
    }) as unknown as OpenHandsEvent;

  it("merges consecutive deltas into one growing bubble keeping the first id", () => {
    let ui: OpenHandsEvent[] = [userMessage];
    const first = delta("Hello");
    ui = handleEventForUI(first, ui);
    ui = handleEventForUI(delta(", "), ui);
    ui = handleEventForUI(delta("world"), ui);

    expect(ui).toHaveLength(2);
    const merged = ui[1];
    expect(isStreamingDeltaEvent(merged)).toBe(true);
    expect((merged as StreamingDeltaEvent).content).toBe("Hello, world");
    expect(merged.id).toBe(first.id);
  });

  it("merges deltas into one bubble even when a state update interleaves", () => {
    let ui: OpenHandsEvent[] = [userMessage];
    const first = delta("Hello");
    ui = handleEventForUI(first, ui);
    ui = handleEventForUI(stateUpdate("state-1"), ui);
    ui = handleEventForUI(delta(", world"), ui);

    const deltas = ui.filter(isStreamingDeltaEvent) as StreamingDeltaEvent[];
    expect(deltas).toHaveLength(1);
    expect(deltas[0].content).toBe("Hello, world");
    expect(deltas[0].id).toBe(first.id);
  });

  it("concatenates reasoning_content across deltas", () => {
    let ui: OpenHandsEvent[] = [userMessage];
    ui = handleEventForUI(delta(null, "think "), ui);
    ui = handleEventForUI(delta(null, "more"), ui);

    expect(ui).toHaveLength(2);
    expect((ui[1] as StreamingDeltaEvent).reasoning_content).toBe("think more");
  });

  it("drops empty boundary deltas that carry no text", () => {
    const ui = handleEventForUI(delta(null, null), [userMessage]);
    expect(ui).toEqual([userMessage]);
  });

  it("reconciles the final assistant message into the delta without duplicating the bubble", () => {
    let ui: OpenHandsEvent[] = [userMessage];
    const first = delta("Hello");
    ui = handleEventForUI(first, ui);
    ui = handleEventForUI(delta(", world"), ui);
    ui = handleEventForUI(assistantMessage("Hello, world"), ui);

    expect(ui).toHaveLength(2); // user message + the (reconciled) delta bubble
    const finalBubble = ui[1];
    expect(isStreamingDeltaEvent(finalBubble)).toBe(true);
    expect((finalBubble as StreamingDeltaEvent).content).toBe("Hello, world");
    expect(finalBubble.id).toBe(first.id);
  });

  it("appends the final message's unstreamed suffix to the last delta", () => {
    let ui: OpenHandsEvent[] = [userMessage];
    ui = handleEventForUI(delta("Hel"), ui);
    // The provider buffered the tail: final text has content never streamed.
    ui = handleEventForUI(assistantMessage("Hello, world"), ui);

    expect(ui).toHaveLength(2);
    expect((ui[1] as StreamingDeltaEvent).content).toBe("Hello, world");
  });

  it("appends the final message normally when nothing was streamed (stream=False)", () => {
    const final = assistantMessage("Complete answer");
    const ui = handleEventForUI(final, [userMessage]);

    expect(ui).toHaveLength(2);
    expect(ui[1]).toBe(final);
    expect(isStreamingDeltaEvent(ui[1])).toBe(false);
  });
});
