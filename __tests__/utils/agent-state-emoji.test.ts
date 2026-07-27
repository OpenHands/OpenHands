import { describe, it, expect } from "vitest";

import { getAgentStateEmoji } from "#/utils/agent-state-emoji";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

describe("getAgentStateEmoji", () => {
  it.each([
    [ExecutionStatus.RUNNING, "🟢"],
    [ExecutionStatus.FINISHED, "✅"],
    [ExecutionStatus.IDLE, "✅"],
    [ExecutionStatus.WAITING_FOR_CONFIRMATION, "✅"],
    [ExecutionStatus.PAUSED, "⚪"],
    [ExecutionStatus.ERROR, "🔴"],
    [ExecutionStatus.STUCK, "🔴"],
  ])("returns %s for execution status %s", (status, emoji) => {
    expect(getAgentStateEmoji(status)).toBe(emoji);
  });

  it("returns null for an unknown / missing status", () => {
    expect(getAgentStateEmoji(null)).toBeNull();
    expect(getAgentStateEmoji(undefined)).toBeNull();
  });
});
