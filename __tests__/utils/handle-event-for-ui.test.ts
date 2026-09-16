import { describe, expect, it } from "vitest";
import {
  ActionEvent,
  ObservationEvent,
  MessageEvent,
  SecurityRisk,
  OpenHandsEvent,
} from "#/types/agent-server/core";
import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import type {
  DeltaFrame,
  ItemStartedFrame,
} from "#/types/agent-server/session-frames";
import { isStreamingDeltaEvent } from "#/types/agent-server/type-guards";
import {
  abortStreamingSlot,
  appendStreamingDeltas,
  clearStreamingSlots,
  handleEventForUI,
  openStreamingSlot,
} from "#/utils/handle-event-for-ui";

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
    activated_skills: [],
    extended_content: [],
  };

  const mockFinishActionEvent: ActionEvent = {
    id: "test-finish-action-1",
    timestamp: Date.now().toString(),
    source: "agent",
    thought: [],
    thinking_blocks: [],
    action: {
      kind: "FinishAction",
      message: "I'll start working on that. Done.",
    },
    tool_name: "finish",
    tool_call_id: "call_finish_1",
    tool_call: {
      id: "call_finish_1",
      type: "function",
      function: {
        name: "finish",
        arguments: JSON.stringify({
          message: "I'll start working on that. Done.",
        }),
      },
    },
    llm_response_id: "response_finish",
    security_risk: SecurityRisk.UNKNOWN,
  };

  const mockAgentMessageEvent: MessageEvent = {
    id: "test-agent-message-1",
    timestamp: Date.now().toString(),
    source: "agent",
    llm_message: {
      role: "assistant",
      content: [{ type: "text", text: "I'll start working on that. Done." }],
    },
    activated_skills: [],
    extended_content: [],
  };

  const makeStreamingDelta = (
    id: string,
    content: string | null,
  ): StreamingDeltaEvent => ({
    id,
    kind: "StreamingDeltaEvent",
    timestamp: Date.now().toString(),
    source: "agent",
    content,
    reasoning_content: null,
  });

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

  describe("streaming slots", () => {
    const started = (
      itemId: string,
      extra: Partial<ItemStartedFrame> = {},
    ): ItemStartedFrame => ({
      type: "item_started",
      item_id: itemId,
      attempt: 1,
      ...extra,
    });

    const delta = (
      itemId: string,
      content: string,
      extra: Partial<DeltaFrame> = {},
    ): DeltaFrame => ({
      type: "delta",
      item_id: itemId,
      attempt: 1,
      order: 0,
      kind: "text",
      content,
      ...extra,
    });

    const slotsOf = (events: OpenHandsEvent[]) =>
      events.filter((event): event is StreamingDeltaEvent =>
        isStreamingDeltaEvent(event),
      );

    it("opens a slot keyed by item_id and accumulates its deltas", () => {
      const opened = openStreamingSlot(started("item-1"), [mockMessageEvent]);
      const result = appendStreamingDeltas(
        [delta("item-1", "I'll start "), delta("item-1", "working on that.")],
        opened,
      );

      expect(slotsOf(result)).toHaveLength(1);
      expect(slotsOf(result)[0]).toMatchObject({
        id: "item-1",
        content: "I'll start working on that.",
        reasoning_content: null,
      });
    });

    it("routes reasoning deltas to reasoning_content, independently of text", () => {
      const opened = openStreamingSlot(started("item-1"), []);
      const result = appendStreamingDeltas(
        [
          delta("item-1", "thinking", { kind: "reasoning" }),
          delta("item-1", "answer"),
          delta("item-1", " more", { kind: "reasoning" }),
        ],
        opened,
      );

      expect(slotsOf(result)[0]).toMatchObject({
        content: "answer",
        reasoning_content: "thinking more",
      });
    });

    it("retires the slot when the durable event carries its id", () => {
      const opened = openStreamingSlot(started("test-agent-message-1"), [
        mockMessageEvent,
      ]);
      const streamed = appendStreamingDeltas(
        [delta("test-agent-message-1", "partial text")],
        opened,
      );

      const result = handleEventForUI(mockAgentMessageEvent, streamed);

      // One equality test on the id — the streamed text is never compared.
      expect(result).toEqual([mockMessageEvent, mockAgentMessageEvent]);
    });

    it("retires the slot in place, keeping events that arrived beside it", () => {
      const opened = openStreamingSlot(
        started("test-agent-message-1", { anchor_seq: 0 }),
        [{ ...mockMessageEvent, seq: 0 }],
      );
      const streamed = appendStreamingDeltas(
        [delta("test-agent-message-1", "partial")],
        opened,
      );
      // A user message lands mid-stream, sequenced after the slot's anchor.
      const interrupted = handleEventForUI(
        { ...mockMessageEvent, id: "mid-stream", seq: 1 },
        streamed,
      );

      const result = handleEventForUI(mockAgentMessageEvent, interrupted);

      expect(result.map((event) => event.id)).toEqual([
        mockMessageEvent.id,
        mockAgentMessageEvent.id,
        "mid-stream",
      ]);
    });

    it("keeps a mid-stream event below the slot rather than splitting it (#15433)", () => {
      const opened = openStreamingSlot(started("item-1", { anchor_seq: 0 }), [
        { ...mockMessageEvent, seq: 0 },
      ]);
      const firstHalf = appendStreamingDeltas(
        [delta("item-1", "first half")],
        opened,
      );
      const interrupted = handleEventForUI(
        { ...mockMessageEvent, id: "mid-stream", seq: 1 },
        firstHalf,
      );
      const result = appendStreamingDeltas(
        [delta("item-1", " second half")],
        interrupted,
      );

      // One slot, still whole, still above the interrupting message.
      expect(slotsOf(result)).toHaveLength(1);
      expect(slotsOf(result)[0].content).toBe("first half second half");
      expect(result.map((event) => event.id)).toEqual([
        mockMessageEvent.id,
        "item-1",
        "mid-stream",
      ]);
    });

    it("supersedes the slot's text when a higher attempt streams it again", () => {
      const opened = openStreamingSlot(started("item-1"), []);
      const first = appendStreamingDeltas([delta("item-1", "retried")], opened);

      const reopened = openStreamingSlot(
        started("item-1", { attempt: 2 }),
        first,
      );
      expect(slotsOf(reopened)[0].content).toBeNull();

      const result = appendStreamingDeltas(
        [delta("item-1", "fresh text", { attempt: 2 })],
        reopened,
      );
      expect(slotsOf(result)).toHaveLength(1);
      expect(slotsOf(result)[0].content).toBe("fresh text");
    });

    it("keeps streamed text when the current attempt's start is repeated", () => {
      const opened = openStreamingSlot(started("item-1"), []);
      const streamed = appendStreamingDeltas([delta("item-1", "kept")], opened);

      const result = openStreamingSlot(started("item-1"), streamed);

      expect(result).toBe(streamed);
      expect(slotsOf(result)[0].content).toBe("kept");
    });

    it("places the slot by anchor_seq when a message lands before the first token", () => {
      // The SDK reads the anchor when the step opens, before the LLM call; a
      // user message persisted while waiting for the first token has a higher
      // seq and has already rendered by the time item_started arrives.
      const history = [
        { ...mockMessageEvent, seq: 0 },
        { ...mockMessageEvent, id: "sent-while-waiting", seq: 1 },
      ];

      const result = openStreamingSlot(
        started("item-1", { anchor_seq: 0 }),
        history,
      );

      expect(result.map((event) => event.id)).toEqual([
        mockMessageEvent.id,
        "item-1",
        "sent-while-waiting",
      ]);
      // Borrowing the anchor's timestamp keeps it there through a re-sort.
      expect(slotsOf(result)[0].timestamp).toBe(mockMessageEvent.timestamp);
    });

    it("borrows the anchor event's timestamp, not its neighbour's", () => {
      const history = [
        {
          ...mockMessageEvent,
          timestamp: "2026-09-16T10:00:00.000001",
          seq: 4,
        },
        // e.g. a transient or client-stamped event with a different clock
        {
          ...mockMessageEvent,
          id: "no-seq",
          timestamp: "2026-09-16T23:59:59Z",
        },
      ];

      const result = openStreamingSlot(
        started("item-1", { anchor_seq: 4 }),
        history,
      );

      expect(slotsOf(result)[0].timestamp).toBe("2026-09-16T10:00:00.000001");
    });

    it("supersedes on a higher-attempt delta even without its item_started", () => {
      const opened = openStreamingSlot(started("item-1"), []);
      const first = appendStreamingDeltas([delta("item-1", "stale")], opened);

      const result = appendStreamingDeltas(
        [
          delta("item-1", "fresh", { attempt: 2 }),
          delta("item-1", "think", { attempt: 2, kind: "reasoning" }),
        ],
        first,
      );

      expect(slotsOf(result)[0]).toMatchObject({
        attempt: 2,
        content: "fresh",
        reasoning_content: "think",
      });
    });

    it("ignores frames from an attempt already superseded", () => {
      const opened = openStreamingSlot(started("item-1", { attempt: 2 }), []);
      const result = appendStreamingDeltas(
        [delta("item-1", "stale", { attempt: 1 })],
        opened,
      );

      expect(slotsOf(result)[0].content).toBeNull();
      expect(openStreamingSlot(started("item-1", { attempt: 1 }), opened)).toBe(
        opened,
      );
    });

    it("opens a slot for a live delta whose item_started predates the connection", () => {
      const result = appendStreamingDeltas(
        [delta("item-late", "streamed after connect")],
        [mockMessageEvent],
        { isFromPlanningAgent: true },
      );

      expect(slotsOf(result)).toHaveLength(1);
      expect(slotsOf(result)[0]).toMatchObject({
        id: "item-late",
        content: "streamed after connect",
        isFromPlanningAgent: true,
      });
    });

    it("does not reopen an item whose durable event already arrived", () => {
      const result = appendStreamingDeltas(
        [delta("test-agent-message-1", "straggler")],
        [mockMessageEvent, mockAgentMessageEvent],
        { isFinished: (itemId) => itemId === mockAgentMessageEvent.id },
      );

      expect(slotsOf(result)).toHaveLength(0);
    });

    it("drops the slot on item_aborted: nothing durable is coming", () => {
      const opened = openStreamingSlot(started("item-1"), [mockMessageEvent]);
      const streamed = appendStreamingDeltas(
        [delta("item-1", "half a sentence")],
        opened,
      );

      expect(abortStreamingSlot("item-1", streamed)).toEqual([
        mockMessageEvent,
      ]);
    });

    it("clears open slots for one socket only", () => {
      const main = openStreamingSlot(started("main-item"), [mockMessageEvent]);
      const both = openStreamingSlot(started("plan-item"), main, {
        isFromPlanningAgent: true,
      });

      expect(clearStreamingSlots(both).map((event) => event.id)).toEqual([
        mockMessageEvent.id,
        "plan-item",
      ]);
      expect(clearStreamingSlots(both, true).map((event) => event.id)).toEqual([
        mockMessageEvent.id,
        "main-item",
      ]);
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

  // Regression for issue #1534: with stream=true an agent step streams its
  // pre-tool-call text as a StreamingDeltaEvent, then the step arrives as an
  // intermediate ActionEvent whose `thought` is that same text. The chat hoists
  // the action's thought into its own message, so the leftover streaming delta
  // must not also render the text or it appears twice.
});
