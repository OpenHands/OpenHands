import { describe, expect, it } from "vitest";
import { shouldRenderEvent } from "#/components/conversation-events/chat/event-content-helpers/should-render-event";
import {
  createPlanningFileEditorActionEvent,
  createOtherActionEvent,
  createPlanningObservationEvent,
  createUserMessageEvent,
} from "test-utils";
import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import {
  ActionEvent,
  ObservationEvent,
  OpenHandsEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { SwitchLLMAction } from "#/types/agent-server/core/base/action";
import { SwitchLLMObservation } from "#/types/agent-server/core/base/observation";

const makeACPEvent = (
  overrides: Partial<ACPToolCallEvent> = {},
): ACPToolCallEvent => ({
  id: "acp-1",
  kind: "ACPToolCallEvent",
  timestamp: "2024-01-01T00:00:00Z",
  source: "agent",
  tool_call_id: "tc-1",
  title: "Run command",
  status: "completed",
  tool_kind: "execute",
  raw_input: { command: "ls" },
  raw_output: "file.txt",
  content: null,
  is_error: false,
  ...overrides,
});

describe("shouldRenderEvent - PlanningFileEditorAction", () => {
  it("should return false for PlanningFileEditorAction", () => {
    const event = createPlanningFileEditorActionEvent("action-1");

    expect(shouldRenderEvent(event)).toBe(false);
  });

  it("should return true for other action types", () => {
    const event = createOtherActionEvent("action-1");

    expect(shouldRenderEvent(event)).toBe(true);
  });

  it("should return true for PlanningFileEditorObservation", () => {
    const event = createPlanningObservationEvent("obs-1");

    // Observations should still render (they're handled separately in event-message)
    expect(shouldRenderEvent(event)).toBe(true);
  });

  it("should return true for user message events", () => {
    const event = createUserMessageEvent("msg-1");

    expect(shouldRenderEvent(event)).toBe(true);
  });
});

describe("shouldRenderEvent - ACPToolCallEvent", () => {
  it("renders the early in_progress 'started' card", () => {
    // The SDK now persists exactly one ``started`` (in_progress) event and
    // one terminal event per ``tool_call_id`` — the action->observation pair.
    // The started event renders the card as "running"; ``handleEventForUI``
    // then replaces it in place when the terminal event arrives. The old
    // half-formed flashing came from the now-removed per-progress fan-out.
    const event = makeACPEvent({ status: "in_progress", raw_input: {} });

    expect(shouldRenderEvent(event)).toBe(true);
  });

  it("renders pending 'started' cards", () => {
    const event = makeACPEvent({ status: "pending" });

    expect(shouldRenderEvent(event)).toBe(true);
  });

  it("renders completed events", () => {
    const event = makeACPEvent({ status: "completed" });

    expect(shouldRenderEvent(event)).toBe(true);
  });

  it("renders failed events", () => {
    const event = makeACPEvent({ status: "failed", is_error: true });

    expect(shouldRenderEvent(event)).toBe(true);
  });

  it("renders events with null status as a running card", () => {
    // A ``null`` status (legacy builds before the field was required) renders
    // as "running" via the absent check mark rather than being hidden — better
    // a card than a silently dropped tool call.
    const event = makeACPEvent({ status: null });

    expect(shouldRenderEvent(event)).toBe(true);
  });
});

describe("shouldRenderEvent - StreamingDeltaEvent", () => {
  const makeStreamingDelta = (
    overrides: Partial<StreamingDeltaEvent> = {},
  ): StreamingDeltaEvent => ({
    id: "delta-1",
    kind: "StreamingDeltaEvent",
    timestamp: "2024-01-01T00:00:00Z",
    source: "agent",
    content: "I'll start working on that.",
    reasoning_content: null,
    ...overrides,
  });

  it("renders text deltas", () => {
    expect(shouldRenderEvent(makeStreamingDelta())).toBe(true);
  });

  it("renders reasoning-only deltas", () => {
    expect(
      shouldRenderEvent(
        makeStreamingDelta({
          content: null,
          reasoning_content: "thinking",
        }),
      ),
    ).toBe(true);
  });

  it("hides empty deltas", () => {
    expect(
      shouldRenderEvent(
        makeStreamingDelta({
          content: null,
          reasoning_content: null,
        }),
      ),
    ).toBe(false);
  });
});

describe("shouldRenderEvent - SwitchLLM", () => {
  const switchAction: ActionEvent<SwitchLLMAction> = {
    id: "switch-action",
    timestamp: "2024-01-01T00:00:00Z",
    source: "agent",
    thought: [],
    thinking_blocks: [],
    action: {
      kind: "SwitchLLMAction",
      profile_name: "haiku",
      reason: "Use a cheaper model.",
    },
    tool_name: "switch_llm",
    tool_call_id: "tool-switch",
    tool_call: {
      id: "tool-switch",
      type: "function",
      function: {
        name: "switch_llm",
        arguments: JSON.stringify({ profile_name: "haiku" }),
      },
    },
    llm_response_id: "response-switch",
    security_risk: SecurityRisk.LOW,
  };

  const makeSwitchObservation = (
    overrides: Partial<SwitchLLMObservation> = {},
  ): ObservationEvent<SwitchLLMObservation> => ({
    id: "switch-observation",
    timestamp: "2024-01-01T00:00:01Z",
    source: "environment",
    tool_name: "switch_llm",
    tool_call_id: "tool-switch",
    action_id: "switch-action",
    observation: {
      kind: "SwitchLLMObservation",
      content: [{ type: "text", text: "Switched." }],
      is_error: false,
      profile_name: "haiku",
      reason: "Use a cheaper model.",
      active_model: "anthropic/claude-haiku-4-5",
      ...overrides,
    },
  });

  it("hides switch actions and successful observations for the shared model UI", () => {
    expect(shouldRenderEvent(switchAction)).toBe(false);
    expect(shouldRenderEvent(makeSwitchObservation())).toBe(false);
  });

  it("keeps failed switch observations visible", () => {
    expect(
      shouldRenderEvent(
        makeSwitchObservation({
          is_error: true,
          content: [{ type: "text", text: "Profile was not found." }],
        }),
      ),
    ).toBe(true);
  });
});

describe("shouldRenderEvent - /goal status updates", () => {
  const makeGoalEvent = (active: boolean): OpenHandsEvent =>
    ({
      id: "goal-1",
      kind: "ConversationStateUpdateEvent",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      key: "goal",
      value: {
        active,
        status: active ? "running" : "complete",
        iteration: 1,
        max_iterations: 10,
        objective: "make pytest pass",
        verdict: null,
      },
    }) as unknown as OpenHandsEvent;

  const makeStatsEvent = (): OpenHandsEvent =>
    ({
      id: "stats-1",
      kind: "ConversationStateUpdateEvent",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      key: "stats",
      value: {},
    }) as unknown as OpenHandsEvent;

  it("renders the terminal goal status inline", () => {
    expect(shouldRenderEvent(makeGoalEvent(false))).toBe(true);
  });

  it("hides the in-progress goal status (shown by the live banner)", () => {
    expect(shouldRenderEvent(makeGoalEvent(true))).toBe(false);
  });

  it("hides non-goal state updates", () => {
    expect(shouldRenderEvent(makeStatsEvent())).toBe(false);
  });
});

describe("shouldRenderEvent - /goal loop re-prompts", () => {
  const makeUserMessage = (text: string): OpenHandsEvent =>
    ({
      id: "m1",
      kind: "MessageEvent",
      source: "user",
      timestamp: "2024-01-01T00:00:00Z",
      llm_message: { role: "user", content: [{ type: "text", text }] },
    }) as unknown as OpenHandsEvent;

  it("hides the per-round follow-up re-prompt", () => {
    const followup = makeUserMessage(
      "The goal is NOT yet complete (audit iteration 1).\nOutstanding: needs tests",
    );
    expect(shouldRenderEvent(followup)).toBe(false);
  });

  it("hides the resume re-prompt", () => {
    expect(
      shouldRenderEvent(
        makeUserMessage("Resuming a goal that was paused or interrupted. ..."),
      ),
    ).toBe(false);
  });

  it("still renders the objective and ordinary user messages", () => {
    expect(
      shouldRenderEvent(makeUserMessage("create roman.py with to_roman(n)")),
    ).toBe(true);
    expect(shouldRenderEvent(makeUserMessage("hello"))).toBe(true);
  });
});
