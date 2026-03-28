import { describe, expect, it } from "vitest";
import {
  createSkillReadyEvent,
  isSkillReadyEvent,
} from "#/components/v1/chat/event-content-helpers/create-skill-ready-event";
import { MessageEvent } from "#/types/v1/core";

const makeMessageEvent = (
  overrides: Partial<MessageEvent> = {},
): MessageEvent =>
  ({
    id: "msg-1",
    timestamp: "2024-01-01T00:00:00Z",
    source: "user",
    message: { role: "user", content: [{ type: "text", text: "test" }] },
    activated_microagents: [],
    extended_content: [],
    ...overrides,
  }) as MessageEvent;

describe("createSkillReadyEvent", () => {
  it("includes _skillReadyItems with structured skill data", () => {
    const event = makeMessageEvent({
      activated_microagents: ["docker"],
      extended_content: [
        { type: "text", text: "<EXTRA_INFO>Docker guide</EXTRA_INFO>" },
      ],
    });

    const result = createSkillReadyEvent(event);

    expect(result._skillReadyItems).toEqual([
      { name: "docker", content: "Docker guide" },
    ]);
  });

  it("sets correct id and source", () => {
    const event = makeMessageEvent({
      id: "msg-42",
      activated_microagents: ["skill1"],
      extended_content: [
        { type: "text", text: "<EXTRA_INFO>content</EXTRA_INFO>" },
      ],
    });

    const result = createSkillReadyEvent(event);

    expect(result.id).toBe("msg-42-skill-ready");
    expect(result.source).toBe("agent");
    expect(result._isSkillReadyEvent).toBe(true);
  });

  it("throws when no skills and no extended content", () => {
    const event = makeMessageEvent();

    expect(() => createSkillReadyEvent(event)).toThrow(
      "Cannot create skill ready event",
    );
  });
});

describe("isSkillReadyEvent", () => {
  it("returns true for valid SkillReadyEvent", () => {
    const event = makeMessageEvent({
      activated_microagents: ["skill1"],
      extended_content: [
        { type: "text", text: "<EXTRA_INFO>content</EXTRA_INFO>" },
      ],
    });

    expect(isSkillReadyEvent(createSkillReadyEvent(event))).toBe(true);
  });

  it("returns false for plain objects", () => {
    expect(isSkillReadyEvent({})).toBe(false);
    expect(isSkillReadyEvent(null)).toBe(false);
    expect(isSkillReadyEvent({ _isSkillReadyEvent: false })).toBe(false);
  });
});
