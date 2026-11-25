import { describe, expect, it } from "vitest";
import {
  computeMessageGaps,
  isNewLocalDay,
} from "#/utils/compute-message-gaps";
import type { MessageEvent } from "#/types/agent-server/core";

const message = (
  id: string,
  source: MessageEvent["source"],
  timestamp: string,
): MessageEvent => ({
  id,
  source,
  timestamp,
  llm_message: {
    role: source === "user" ? "user" : "assistant",
    content: [{ type: "text", text: id }],
  },
  activated_skills: [],
  extended_content: [],
});

describe("isNewLocalDay", () => {
  it("detects a day change across midnight in local time", () => {
    const previous = new Date(2026, 0, 1, 23, 59);
    const current = new Date(2026, 0, 2, 0, 1);
    expect(isNewLocalDay(previous, current)).toBe(true);
  });

  it("ignores time differences within the same day", () => {
    const previous = new Date(2026, 0, 1, 9, 0);
    const current = new Date(2026, 0, 1, 23, 0);
    expect(isNewLocalDay(previous, current)).toBe(false);
  });
});

describe("computeMessageGaps", () => {
  it("flags a message that arrives 5+ minutes after the previous one", () => {
    const events = [
      message("u1", "user", "2026-01-01T10:00:00Z"),
      message("a1", "agent", "2026-01-01T10:06:00Z"),
    ];
    expect(computeMessageGaps(events)).toEqual(new Set(["a1"]));
  });

  it("does not flag messages within a 5-minute gap on the same day", () => {
    const events = [
      message("u1", "user", "2026-01-01T10:00:00Z"),
      message("a1", "agent", "2026-01-01T10:03:00Z"),
      message("a2", "agent", "2026-01-01T10:04:00Z"),
    ];
    expect(computeMessageGaps(events)).toEqual(new Set());
  });

  it("measures the gap from the previous chat message, ignoring other events", () => {
    const events = [
      message("u1", "user", "2026-01-01T10:00:00Z"),
      message("env1", "environment", "2026-01-01T10:00:30Z"),
      message("a1", "agent", "2026-01-01T10:06:00Z"),
    ];
    // a1 is 6 minutes after u1 (the last chat message), so it is flagged.
    expect(computeMessageGaps(events)).toEqual(new Set(["a1"]));
  });
});
