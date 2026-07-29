import { describe, expect, it } from "vitest";
import {
  hasConversationStarted,
  resolvePickerKind,
} from "#/components/features/chat/components/resolve-picker-kind";

// The pill is always an LLM selector (OSS-5735): agent-profile switching lives
// in the "+" tools menu, and neither the backend nor the conversation's state
// branches here any more — whether a pick is actionable is decided in
// useChatInputLlmProfileState, so cloud never falls back to a raw model chip.
describe("resolvePickerKind", () => {
  it("shows the constrained model picker in an ACP context", () => {
    expect(resolvePickerKind({ isAcp: true })).toBe("model");
  });

  it("shows the LLM-profile picker for an OpenHands context", () => {
    expect(resolvePickerKind({ isAcp: false })).toBe("llm-profile");
  });
});

describe("hasConversationStarted", () => {
  const empty = {
    isLoadingHistory: false,
    hasUserEvents: false,
    hasPendingUserMessages: false,
    hasSubstantiveAgentActions: false,
    hasModelEntries: false,
  };

  it("keeps a fully loaded blank conversation in pre-run state", () => {
    expect(hasConversationStarted(empty)).toBe(false);
  });

  it.each([
    "isLoadingHistory",
    "hasUserEvents",
    "hasPendingUserMessages",
    "hasSubstantiveAgentActions",
    "hasModelEntries",
  ] as const)("treats %s as started", (field) => {
    expect(hasConversationStarted({ ...empty, [field]: true })).toBe(true);
  });
});
