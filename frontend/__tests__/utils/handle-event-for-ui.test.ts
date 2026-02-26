import { describe, expect, it } from "vitest";
import {
  ActionEvent,
  ObservationEvent,
  MessageEvent,
  SecurityRisk,
  OpenHandsEvent,
} from "#/types/v1/core";
import { handleEventForUI } from "#/utils/handle-event-for-ui";

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
