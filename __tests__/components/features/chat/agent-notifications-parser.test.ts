import { describe, expect, it } from "vitest";
import { extractAgentNotifications } from "#/components/features/chat/agent-notifications-parser";

describe("extractAgentNotifications", () => {
  it("returns the original text when no agent-notification fences are present", () => {
    const text = "Here is a normal assistant reply.";

    expect(extractAgentNotifications(text)).toEqual({
      message: text,
      notifications: [],
    });
  });

  it("parses a valid fenced notification and strips it from the message", () => {
    const text = [
      "I finished the workflow.",
      "",
      "```agent-notification",
      '{"id":"skill-standup","kind":"skill","name":"Standup helper","prompt":"Save a reusable skill."}',
      "```",
    ].join("\n");

    const result = extractAgentNotifications(text);

    expect(result.message).toBe("I finished the workflow.");
    expect(result.notifications).toEqual([
      expect.objectContaining({
        id: "skill-standup",
        kind: "skill",
        name: "Standup helper",
        prompt: "Save a reusable skill.",
      }),
    ]);
  });

  it("drops malformed fenced blocks silently", () => {
    const text = [
      "Done.",
      "",
      "```agent-notification",
      "{not-json}",
      "```",
    ].join("\n");

    expect(extractAgentNotifications(text)).toEqual({
      message: "Done.",
      notifications: [],
    });
  });
});
