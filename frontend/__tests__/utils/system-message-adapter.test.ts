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
});
