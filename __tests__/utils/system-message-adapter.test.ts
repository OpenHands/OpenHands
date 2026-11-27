import { describe, it, expect } from "vitest";
import { adaptSystemMessage } from "#/utils/system-message-adapter";
import { EventState } from "#/stores/use-event-store";

const v1Event: EventState["events"] = [
  {
    id: "v1-id",
    timestamp: "2025-12-30T12:00:00Z",
    source: "agent",
    system_prompt: {
      type: "text",
      text: "v1 prompt",
    },
    tools: [
      {
        type: "function",
        function: {
          name: "bash",
          description: "Execute bash",
          parameters: {},
        },
      },
    ],
  },
];

describe("adaptSystemMessage", () => {
  it("should correctly adapt the v1 system_prompt event structure", () => {
    const result = adaptSystemMessage(v1Event);
    expect(result).not.toBeNull();
    expect(result?.content).toBe("v1 prompt");
  });

  it("should return null when no system message is present in events", () => {
    expect(adaptSystemMessage([])).toBeNull();
  });

  it("should leave content unchanged when dynamic_context is absent", () => {
    const result = adaptSystemMessage(v1Event);
    expect(result?.content).toBe("v1 prompt");
  });

  it("should append dynamic_context to the system prompt content", () => {
    const events: EventState["events"] = [
      {
        id: "v1-id",
        timestamp: "2025-12-30T12:00:00Z",
        source: "agent",
        system_prompt: { type: "text", text: "v1 prompt" },
        tools: [],
        dynamic_context: { type: "text", text: "<SKILLS>my-skill</SKILLS>" },
      },
    ];
    const result = adaptSystemMessage(events);
    expect(result?.content).toContain("v1 prompt");
    expect(result?.content).toContain("<SKILLS>my-skill</SKILLS>");
  });

  it("should redact unmasked custom secret values in dynamic_context", () => {
    const events: EventState["events"] = [
      {
        id: "v1-id",
        timestamp: "2025-12-30T12:00:00Z",
        source: "agent",
        system_prompt: { type: "text", text: "v1 prompt" },
        tools: [],
        dynamic_context: {
          type: "text",
          text: "<CUSTOM_SECRETS>\nMY_API_KEY=super-secret-value\n</CUSTOM_SECRETS>",
        },
      },
    ];
    const result = adaptSystemMessage(events);
    expect(result?.content).not.toContain("super-secret-value");
    expect(result?.content).toContain("MY_API_KEY=<secret-hidden>");
  });
});
