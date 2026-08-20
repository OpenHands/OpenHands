import { describe, expect, it } from "vitest";
import { isAutomationsRoute } from "#/manifests/automation-interface";

describe("isAutomationsRoute", () => {
  it("matches the automations list and nested routes", () => {
    expect(isAutomationsRoute("/automations")).toBe(true);
    expect(isAutomationsRoute("/automations/templates")).toBe(true);
    expect(isAutomationsRoute("/automations/auto-1")).toBe(true);
    expect(isAutomationsRoute("/automations/new/github-pr-reviewer")).toBe(
      true,
    );
  });

  it("does not match unrelated routes", () => {
    expect(isAutomationsRoute("/")).toBe(false);
    expect(isAutomationsRoute("/conversations")).toBe(false);
    expect(isAutomationsRoute("/conversations/abc")).toBe(false);
    expect(isAutomationsRoute("/settings")).toBe(false);
  });
});
