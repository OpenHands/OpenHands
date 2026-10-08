import { describe, expect, it } from "vitest";
import { isPlanningMode } from "#/utils/conversation-mode";

describe("isPlanningMode", () => {
  it("treats plain plan mode as a planning mode regardless of phase", () => {
    expect(isPlanningMode("plan")).toBe(true);
    expect(isPlanningMode("plan", null)).toBe(true);
    expect(isPlanningMode("plan", "requirements")).toBe(true);
  });

  it("treats code mode as a non-planning mode", () => {
    expect(isPlanningMode("code")).toBe(false);
    expect(isPlanningMode("code", "analysis")).toBe(false);
  });

  it("treats Deep Plan with an active planning phase as a planning mode", () => {
    expect(isPlanningMode("deep-plan", "analysis")).toBe(true);
    expect(isPlanningMode("deep-plan", "requirements")).toBe(true);
  });

  it("hands the Implementation phase to the code agent", () => {
    expect(isPlanningMode("deep-plan", "implementation")).toBe(false);
  });

  it("treats Deep Plan with no active phase as non-planning", () => {
    // No phase means no planner has been provisioned yet, so a message must
    // not be routed to an empty planner id.
    expect(isPlanningMode("deep-plan", null)).toBe(false);
    expect(isPlanningMode("deep-plan")).toBe(false);
  });
});
