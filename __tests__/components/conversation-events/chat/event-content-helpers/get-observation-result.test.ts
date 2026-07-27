import { describe, expect, it } from "vitest";
import {
  getACPToolCallResult,
  getObservationResult,
} from "#/components/conversation-events/chat/event-content-helpers/get-observation-result";
import { ACPToolCallEvent } from "#/types/agent-server/core/events/acp-tool-call-event";
import { ObservationEvent } from "#/types/agent-server/core";

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

describe("getACPToolCallResult", () => {
  it("maps completed → success", () => {
    expect(getACPToolCallResult(makeACPEvent({ status: "completed" }))).toBe(
      "success",
    );
  });

  it("maps failed → error", () => {
    expect(getACPToolCallResult(makeACPEvent({ status: "failed" }))).toBe(
      "error",
    );
  });

  it("maps is_error → error even when status is completed", () => {
    expect(
      getACPToolCallResult(makeACPEvent({ status: "completed", is_error: true })),
    ).toBe("error");
  });

  it.each(["pending", "in_progress"] as const)(
    "maps non-terminal status %s → undefined (running card)",
    (status) => {
      expect(getACPToolCallResult(makeACPEvent({ status }))).toBeUndefined();
    },
  );

  it("maps null status → undefined (running card)", () => {
    expect(getACPToolCallResult(makeACPEvent({ status: null }))).toBeUndefined();
  });
});

const makeObs = (
  observation: ObservationEvent["observation"],
): ObservationEvent => ({
  id: "obs-1",
  timestamp: "2024-01-01T00:00:00Z",
  source: "environment",
  tool_name: "tool",
  tool_call_id: "tc-1",
  action_id: "act-1",
  observation,
});

describe("getObservationResult", () => {
  it("maps InvokeSkillObservation is_error → error, otherwise success", () => {
    expect(
      getObservationResult(
        makeObs({
          kind: "InvokeSkillObservation",
          skill_name: "s",
          content: [],
          is_error: true,
        }),
      ),
    ).toBe("error");
    expect(
      getObservationResult(
        makeObs({
          kind: "InvokeSkillObservation",
          skill_name: "s",
          content: [],
          is_error: false,
        }),
      ),
    ).toBe("success");
  });

  it("maps TaskObservation is_error or failed status → error, otherwise success", () => {
    const task = (extra: { status: string; is_error?: boolean }) =>
      makeObs({
        kind: "TaskObservation",
        content: [],
        task_id: "t1",
        subagent: "code-explorer",
        ...extra,
      });
    expect(getObservationResult(task({ status: "completed" }))).toBe("success");
    expect(getObservationResult(task({ status: "failed" }))).toBe("error");
    expect(
      getObservationResult(task({ status: "completed", is_error: true })),
    ).toBe("error");
  });

  it("maps CanvasUIObservation is_error → error, otherwise success", () => {
    expect(
      getObservationResult(
        makeObs({ kind: "CanvasUIObservation", content: [], is_error: true }),
      ),
    ).toBe("error");
    expect(
      getObservationResult(
        makeObs({ kind: "CanvasUIObservation", content: [], is_error: false }),
      ),
    ).toBe("success");
  });
});
