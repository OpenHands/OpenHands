import { beforeEach, describe, expect, it } from "vitest";
import {
  consumeAutomationSetupHandoff,
  markAutomationSetupHandoff,
} from "#/api/automation-setup-handoff-store";

describe("automation setup handoff store", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("keeps a handoff readable for the browser session", () => {
    markAutomationSetupHandoff("conversation-1");

    expect(consumeAutomationSetupHandoff("conversation-1")).toBe(true);
    expect(consumeAutomationSetupHandoff("conversation-1")).toBe(true);
    expect(consumeAutomationSetupHandoff("conversation-2")).toBe(false);
  });
});
