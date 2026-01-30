import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  usePlanPreviewEvents,
  shouldShowPlanPreview,
} from "#/components/v1/chat/hooks/use-plan-preview-events";
import {
  OpenHandsEvent,
  MessageEvent,
  ObservationEvent,
  PlanningFileEditorObservation,
} from "#/types/v1/core";

// Helper to create a user message event
const createUserMessageEvent = (id: string): MessageEvent => ({
  id,
  timestamp: new Date().toISOString(),
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "User message" }],
  },
  activated_microagents: [],
  extended_content: [],
});

// Helper to create a PlanningFileEditorObservation event
const createPlanningObservationEvent = (
  id: string,
  actionId: string = "action-1",
): ObservationEvent<PlanningFileEditorObservation> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "planning_file_editor",
  tool_call_id: "call-1",
  action_id: actionId,
  observation: {
    kind: "PlanningFileEditorObservation",
    content: [{ type: "text", text: "Plan content" }],
    is_error: false,
    command: "create",
    path: "/workspace/PLAN.md",
    prev_exist: false,
    old_content: null,
    new_content: "Plan content",
  },
});

// Helper to create a non-planning observation event
const createOtherObservationEvent = (id: string): ObservationEvent => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: "call-1",
  action_id: "action-1",
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "output" }],
    command: "echo test",
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
});

describe("usePlanPreviewEvents", () => {
  it("should return empty set when no events provided", () => {
    const { result } = renderHook(() => usePlanPreviewEvents([]));

    expect(result.current).toBeInstanceOf(Set);
    expect(result.current.size).toBe(0);
  });

  it("should return empty set when no PlanningFileEditorObservation events exist", () => {
    const events: OpenHandsEvent[] = [
      createUserMessageEvent("user-1"),
      createOtherObservationEvent("obs-1"),
    ];

    const { result } = renderHook(() => usePlanPreviewEvents(events));

    expect(result.current.size).toBe(0);
  });

  it("should return event ID for single PlanningFileEditorObservation in one phase", () => {
    const events: OpenHandsEvent[] = [
      createUserMessageEvent("user-1"),
      createPlanningObservationEvent("plan-obs-1"),
    ];

    const { result } = renderHook(() => usePlanPreviewEvents(events));

    expect(result.current.size).toBe(1);
    expect(result.current.has("plan-obs-1")).toBe(true);
  });

  it("should return only the last PlanningFileEditorObservation when multiple exist in one phase", () => {
    const events: OpenHandsEvent[] = [
      createUserMessageEvent("user-1"),
      createPlanningObservationEvent("plan-obs-1"),
      createPlanningObservationEvent("plan-obs-2"),
      createPlanningObservationEvent("plan-obs-3"),
      createOtherObservationEvent("other-obs-1"),
    ];

    const { result } = renderHook(() => usePlanPreviewEvents(events));

    // Should only include the last one in the phase
    expect(result.current.size).toBe(1);
    expect(result.current.has("plan-obs-1")).toBe(false);
    expect(result.current.has("plan-obs-2")).toBe(false);
    expect(result.current.has("plan-obs-3")).toBe(true);
  });

  it("should return one event ID per phase when multiple phases exist", () => {
    const events: OpenHandsEvent[] = [
      createUserMessageEvent("user-1"),
      createPlanningObservationEvent("plan-obs-1"),
      createPlanningObservationEvent("plan-obs-2"),
      createUserMessageEvent("user-2"),
      createPlanningObservationEvent("plan-obs-3"),
      createPlanningObservationEvent("plan-obs-4"),
    ];

    const { result } = renderHook(() => usePlanPreviewEvents(events));

    // Should have one preview per phase (last observation in each phase)
    expect(result.current.size).toBe(2);
    expect(result.current.has("plan-obs-2")).toBe(true); // Last in phase 1
    expect(result.current.has("plan-obs-4")).toBe(true); // Last in phase 2
    expect(result.current.has("plan-obs-1")).toBe(false);
    expect(result.current.has("plan-obs-3")).toBe(false);
  });

  it("should handle phase with no PlanningFileEditorObservation", () => {
    const events: OpenHandsEvent[] = [
      createUserMessageEvent("user-1"),
      createOtherObservationEvent("obs-1"),
      createUserMessageEvent("user-2"),
      createPlanningObservationEvent("plan-obs-1"),
    ];

    const { result } = renderHook(() => usePlanPreviewEvents(events));

    // Only phase 2 has a planning observation
    expect(result.current.size).toBe(1);
    expect(result.current.has("plan-obs-1")).toBe(true);
  });

  it("should handle events starting with non-user message", () => {
    const events: OpenHandsEvent[] = [
      createOtherObservationEvent("obs-1"),
      createUserMessageEvent("user-1"),
      createPlanningObservationEvent("plan-obs-1"),
    ];

    const { result } = renderHook(() => usePlanPreviewEvents(events));

    // Events before first user message should be in first phase
    expect(result.current.size).toBe(1);
    expect(result.current.has("plan-obs-1")).toBe(true);
  });
});

describe("shouldShowPlanPreview", () => {
  it("should return true when event ID is in the set", () => {
    const planPreviewEventIds = new Set(["event-1", "event-2", "event-3"]);

    expect(shouldShowPlanPreview("event-2", planPreviewEventIds)).toBe(true);
  });

  it("should return false when event ID is not in the set", () => {
    const planPreviewEventIds = new Set(["event-1", "event-2"]);

    expect(shouldShowPlanPreview("event-3", planPreviewEventIds)).toBe(false);
  });

  it("should return false when set is empty", () => {
    const planPreviewEventIds = new Set<string>();

    expect(shouldShowPlanPreview("event-1", planPreviewEventIds)).toBe(false);
  });
});
