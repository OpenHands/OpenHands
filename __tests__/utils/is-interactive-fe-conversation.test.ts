import { describe, expect, it } from "vitest";
import {
  CONVERSATION_ORIGIN_TAG_KEY,
  isInteractiveFeConversation,
} from "#/utils/is-interactive-fe-conversation";
import { ACP_SERVER_TAG_KEY } from "#/api/agent-server-adapter";

describe("isInteractiveFeConversation", () => {
  it("returns false when conversation is missing", () => {
    expect(isInteractiveFeConversation(null)).toBe(false);
    expect(isInteractiveFeConversation(undefined)).toBe(false);
  });

  it("allows local FE conversations with null trigger and no origin tag", () => {
    expect(
      isInteractiveFeConversation({ trigger: null, tags: null }),
    ).toBe(true);
    expect(
      isInteractiveFeConversation({
        trigger: null,
        tags: { [ACP_SERVER_TAG_KEY]: "claude-code" },
      }),
    ).toBe(true);
  });

  it("allows Cloud gui-triggered conversations", () => {
    expect(
      isInteractiveFeConversation({ trigger: "gui", tags: null }),
    ).toBe(true);
  });

  it.each([
    "resolver",
    "suggested_task",
    "microagent_management",
  ] as const)("rejects Cloud trigger %s", (trigger) => {
    expect(isInteractiveFeConversation({ trigger, tags: null })).toBe(false);
  });

  it("rejects conversations stamped with an origin tag", () => {
    expect(
      isInteractiveFeConversation({
        trigger: null,
        tags: { [CONVERSATION_ORIGIN_TAG_KEY]: "slack" },
      }),
    ).toBe(false);
    expect(
      isInteractiveFeConversation({
        trigger: "gui",
        tags: { [CONVERSATION_ORIGIN_TAG_KEY]: "automation" },
      }),
    ).toBe(false);
  });

  it("ignores blank origin tag values", () => {
    expect(
      isInteractiveFeConversation({
        trigger: null,
        tags: { [CONVERSATION_ORIGIN_TAG_KEY]: "   " },
      }),
    ).toBe(true);
  });
});
