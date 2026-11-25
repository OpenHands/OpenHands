import { describe, expect, it } from "vitest";
import { computeAssistantProcessingTimes } from "#/utils/compute-message-processing-times";
import type { MessageEvent } from "#/types/agent-server/core";

const message = (
  id: string,
  source: "user" | "agent",
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

describe("computeAssistantProcessingTimes", () => {
  it("measures each reply from its preceding user message", () => {
    const events = [
      message("u1", "user", "2026-01-01T00:00:00Z"),
      message("a1", "agent", "2026-01-01T00:00:12Z"),
      message("u2", "user", "2026-01-01T00:01:00Z"),
      message("a2", "agent", "2026-01-01T00:02:05Z"),
    ];

    const result = computeAssistantProcessingTimes(events);
    expect(result.get("a1")).toBe(12_000);
    expect(result.get("a2")).toBe(65_000);
  });

  it("skips replies with no preceding user message", () => {
    const events = [message("a1", "agent", "2026-01-01T00:00:12Z")];
    expect(computeAssistantProcessingTimes(events).has("a1")).toBe(false);
  });

  it("reads offset-less timestamps as UTC", () => {
    const events = [
      message("u1", "user", "2026-01-01T00:00:00"),
      message("a1", "agent", "2026-01-01T00:00:12"),
    ];
    expect(computeAssistantProcessingTimes(events).get("a1")).toBe(12_000);
  });

  it("clamps negative deltas to zero", () => {
    const events = [
      message("u1", "user", "2026-01-01T00:00:12Z"),
      message("a1", "agent", "2026-01-01T00:00:00Z"),
    ];
    expect(computeAssistantProcessingTimes(events).get("a1")).toBe(0);
  });
});
