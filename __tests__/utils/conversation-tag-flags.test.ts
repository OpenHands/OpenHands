import { describe, expect, it } from "vitest";
import {
  ARCHIVED_CONVERSATION_TAG_KEY,
  conversationTagFlagPatch,
  getConversationResolution,
  hasConversationTagFlag,
  parseAutoResolveNote,
  PINNED_CONVERSATION_TAG_KEY,
  RESOLVED_CONVERSATION_TAG_KEY,
  withConversationTagFlags,
} from "#/utils/conversation-tag-flags";
import { getDisplayConversationTags } from "#/api/agent-server-adapter";

describe("conversation tag flags", () => {
  it("uses squashed lowercase keys that satisfy the backend tag-key rule", () => {
    const tagKeyRule = /^[a-z0-9]+$/;
    expect(PINNED_CONVERSATION_TAG_KEY).toMatch(tagKeyRule);
    expect(ARCHIVED_CONVERSATION_TAG_KEY).toMatch(tagKeyRule);
  });

  it("reads only the exact 'true' value as enabled", () => {
    expect(
      hasConversationTagFlag({ pinned: "true" }, PINNED_CONVERSATION_TAG_KEY),
    ).toBe(true);
    expect(
      hasConversationTagFlag({ pinned: "false" }, PINNED_CONVERSATION_TAG_KEY),
    ).toBe(false);
    expect(hasConversationTagFlag({}, PINNED_CONVERSATION_TAG_KEY)).toBe(false);
    expect(hasConversationTagFlag(null, PINNED_CONVERSATION_TAG_KEY)).toBe(
      false,
    );
    expect(hasConversationTagFlag(undefined, PINNED_CONVERSATION_TAG_KEY)).toBe(
      false,
    );
  });

  it("builds a full replacement map without mutating the input", () => {
    const current = { origin: "slack", acpserver: "claude-code" };
    const next = withConversationTagFlags(current, {
      [PINNED_CONVERSATION_TAG_KEY]: true,
    });
    expect(next).toEqual({
      origin: "slack",
      acpserver: "claude-code",
      pinned: "true",
    });
    expect(current).toEqual({ origin: "slack", acpserver: "claude-code" });

    expect(
      withConversationTagFlags(next, {
        [PINNED_CONVERSATION_TAG_KEY]: false,
        [ARCHIVED_CONVERSATION_TAG_KEY]: true,
      }),
    ).toEqual({
      origin: "slack",
      acpserver: "claude-code",
      archived: "true",
    });
    expect(withConversationTagFlags(null, { pinned: true })).toEqual({
      pinned: "true",
    });
  });

  it("builds a merge patch that deletes disabled flags", () => {
    expect(
      conversationTagFlagPatch({
        [PINNED_CONVERSATION_TAG_KEY]: true,
        [ARCHIVED_CONVERSATION_TAG_KEY]: false,
      }),
    ).toEqual({ pinned: "true", archived: null });
  });

  it("keeps pin/archive flags out of the display tag chips", () => {
    expect(
      getDisplayConversationTags({
        pinned: "true",
        archived: "true",
        origin: "slack",
      }),
    ).toEqual([["origin", "slack"]]);
  });
});

describe("getConversationResolution", () => {
  it("reads the user and auto resolutions", () => {
    expect(getConversationResolution({ resolved: "user" })).toBe("user");
    expect(getConversationResolution({ resolved: "auto" })).toBe("auto");
  });

  it("treats a missing or unknown value as unresolved", () => {
    expect(getConversationResolution(undefined)).toBeNull();
    expect(getConversationResolution(null)).toBeNull();
    expect(getConversationResolution({})).toBeNull();
    expect(getConversationResolution({ resolved: "" })).toBeNull();
    expect(getConversationResolution({ resolved: "true" })).toBeNull();
  });

  it("removes resolved like any other flag when reopening", () => {
    expect(
      withConversationTagFlags(
        { resolved: "auto", project: "abc" },
        { [RESOLVED_CONVERSATION_TAG_KEY]: false },
      ),
    ).toEqual({ project: "abc" });
    expect(
      conversationTagFlagPatch({ [RESOLVED_CONVERSATION_TAG_KEY]: false }),
    ).toEqual({ resolved: null });
  });

  it("reads the server's auto-resolve note", () => {
    expect(parseAutoResolveNote("0.93 nothing")).toEqual({
      percent: 93,
      source: "judge",
    });
    expect(parseAutoResolveNote("1.00 nothing")).toEqual({
      percent: 100,
      source: "judge",
    });
    expect(parseAutoResolveNote("0.95 critic")).toEqual({
      percent: 95,
      source: "critic",
    });
    // resolve_conversation: the agent resolved its own conversation.
    expect(parseAutoResolveNote("agent")).toEqual({ source: "agent" });
    for (const bad of [null, undefined, "", "sure", "1.5 nothing", "0.9"]) {
      expect(parseAutoResolveNote(bad)).toBeNull();
    }
  });
});
