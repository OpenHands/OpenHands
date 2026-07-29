import { describe, expect, it } from "vitest";
import {
  TABLE_DEMO_CONVERSATION_ID,
  TABLE_DEMO_EVENTS,
} from "#/fixtures/table-demo-conversation";

describe("table demo conversation fixture", () => {
  it("seeds a user message and an agent reply with a wide markdown table", () => {
    expect(TABLE_DEMO_CONVERSATION_ID).toBe("table-demo");
    expect(TABLE_DEMO_EVENTS).toHaveLength(2);
    expect(TABLE_DEMO_EVENTS[0]?.source).toBe("user");
    expect(TABLE_DEMO_EVENTS[1]?.source).toBe("agent");

    const agentText =
      "llm_message" in TABLE_DEMO_EVENTS[1]!
        ? TABLE_DEMO_EVENTS[1].llm_message.content[0]
        : null;
    expect(agentText).toMatchObject({ type: "text" });
    if (agentText && "text" in agentText) {
      expect(agentText.text).toContain("| Feature | OpenHands |");
      expect(agentText.text).toContain("| Continue |");
    }
  });
});
