import { describe, expect, it } from "vitest";
import { shouldRenderEvent } from "#/components/v1/chat/event-content-helpers/should-render-event";
import {
  createPlanningFileEditorActionEvent,
  createOtherActionEvent,
  createPlanningObservationEvent,
  createUserMessageEvent,
} from "test-utils";
import { ACPToolCallEvent } from "#/types/v1/core/events/acp-tool-call-event";

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
  it("should return false for in_progress events (suppress empty-args flash)", () => {
    const event = makeACPEvent({ status: "in_progress", raw_input: {} });

    expect(shouldRenderEvent(event)).toBe(false);
  });

  it("should return true for completed events", () => {
    const event = makeACPEvent({ status: "completed" });

    expect(shouldRenderEvent(event)).toBe(true);
  });

  it("should return true for failed events", () => {
    const event = makeACPEvent({ status: "failed", is_error: true });

    expect(shouldRenderEvent(event)).toBe(true);
  });

  it("should return false for null status (pre-terminal — no production events yet)", () => {
    // ACP feature flag has never shipped to production with the GUI, so
    // there are no legacy null-status events in the wild. Treat null as
    // pre-terminal and suppress to avoid flashing an empty card during
    // the intermediate updates some ACP servers emit before settling.
    const event = makeACPEvent({ status: null });

    expect(shouldRenderEvent(event)).toBe(false);
  });
});
