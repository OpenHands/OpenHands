import { beforeEach, describe, expect, it } from "vitest";
import {
  clearAgentNotificationPendingPrompts,
  readAgentNotificationPendingPrompts,
  writeAgentNotificationPendingPrompts,
} from "#/components/features/chat/agent-notifications-pending-prompts";

describe("agent-notifications-pending-prompts", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("writes and reads pending prompts for a conversation", () => {
    writeAgentNotificationPendingPrompts("conv-1", [
      "Create skill A",
      "Create automation B",
    ]);

    expect(readAgentNotificationPendingPrompts("conv-1")).toEqual([
      "Create skill A",
      "Create automation B",
    ]);
  });

  it("clears pending prompts for a conversation", () => {
    writeAgentNotificationPendingPrompts("conv-1", ["Prompt"]);

    clearAgentNotificationPendingPrompts("conv-1");

    expect(readAgentNotificationPendingPrompts("conv-1")).toEqual([]);
  });
});
