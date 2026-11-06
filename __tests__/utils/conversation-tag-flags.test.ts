import { describe, expect, it } from "vitest";
import {
  ARCHIVED_CONVERSATION_TAG_KEY,
  conversationTagFlagPatch,
  hasConversationTagFlag,
  PINNED_CONVERSATION_TAG_KEY,
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
