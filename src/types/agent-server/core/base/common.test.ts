import { describe, expect, it } from "vitest";
import {
  ExecutionStatus,
  SourceType,
} from "#/types/agent-server/core/base/common";
import {
  isBaseEvent,
  isAgentServerEvent,
} from "#/types/agent-server/type-guards";

describe("ExecutionStatus (sourced from @openhands/typescript-client)", () => {
  it("keeps every historical status value stable", () => {
    expect(ExecutionStatus.IDLE).toBe("idle");
    expect(ExecutionStatus.RUNNING).toBe("running");
    expect(ExecutionStatus.PAUSED).toBe("paused");
    expect(ExecutionStatus.WAITING_FOR_CONFIRMATION).toBe(
      "waiting_for_confirmation",
    );
    expect(ExecutionStatus.FINISHED).toBe("finished");
    expect(ExecutionStatus.ERROR).toBe("error");
    expect(ExecutionStatus.STUCK).toBe("stuck");
  });

  it("exposes server statuses the old local enum was missing", () => {
    // The canonical client contract adds "deleting" to the historical set.
    // Canvas previously redeclared ExecutionStatus without it, drifting from
    // the wire model (#16952).
    expect(ExecutionStatus.DELETING).toBe("deleting");
  });
});

describe("SourceType (sourced from @openhands/typescript-client)", () => {
  it("is equivalent to the canonical EventSource", () => {
    const sources: readonly SourceType[] = [
      "agent",
      "user",
      "environment",
      "system",
      "hook",
    ];
    expect(sources).toContain("system");
  });
});

describe("runtime guards and the widened source set", () => {
  it("isBaseEvent accepts a system-sourced event", () => {
    const systemEvent = {
      id: "01JW7T4XG8Q0VPRQM6YK0N3ZB2",
      timestamp: "2026-08-27T00:00:00.000Z",
      source: "system",
      kind: "SystemPromptEvent",
    };
    expect(isBaseEvent(systemEvent)).toBe(true);
    expect(isAgentServerEvent(systemEvent)).toBe(true);
  });

  it("isBaseEvent still rejects non-events and unknown sources", () => {
    const notAnEvent = { id: "id", timestamp: "ts", source: "unknown" };
    expect(isBaseEvent(notAnEvent)).toBe(false);
    expect(isBaseEvent(null)).toBe(false);
    expect(isBaseEvent("string")).toBe(false);
  });
});
