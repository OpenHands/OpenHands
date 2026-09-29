import { beforeEach, describe, expect, it } from "vitest";
import {
  clearAutomationSetupHandoff,
  consumeAutomationSetupHandoff,
  markAutomationSetupHandoff,
} from "#/api/automation-setup-handoff-store";

describe("automation setup handoff store", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("keeps a handoff readable until it is explicitly cleared", () => {
    markAutomationSetupHandoff("conversation-1");

    expect(consumeAutomationSetupHandoff("conversation-1")).toBe(true);
    expect(consumeAutomationSetupHandoff("conversation-1")).toBe(true);

    clearAutomationSetupHandoff("conversation-1");

    expect(consumeAutomationSetupHandoff("conversation-1")).toBe(false);
  });
});
