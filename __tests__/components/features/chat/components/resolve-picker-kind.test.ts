import { describe, expect, it } from "vitest";
import {
  hasConversationStarted,
  resolvePickerKind,
  type ResolvePickerKindInput,
} from "#/components/features/chat/components/resolve-picker-kind";

const base: ResolvePickerKindInput = {
  hasConversation: false,
  isCloud: false,
  isAcp: false,
  profilesAvailable: true,
};

describe("resolvePickerKind", () => {
  describe("home (no active conversation)", () => {
    it("shows the agent-profile picker when profiles are available", () => {
      expect(resolvePickerKind({ ...base, hasConversation: false })).toBe(
        "agent-profile",
      );
      // isCloud / isAcp don't matter once profiles exist.
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: false,
          isCloud: true,
          isAcp: true,
        }),
      ).toBe("agent-profile");
    });

    it("falls back to the LLM-profile picker on local when no profiles exist", () => {
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: false,
          isCloud: false,
          profilesAvailable: false,
        }),
      ).toBe("llm-profile");
    });

    it("falls back to the model picker on cloud when no profiles exist", () => {
      // Cloud has no home LLM-profile activate path.
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: false,
          isCloud: true,
          profilesAvailable: false,
        }),
      ).toBe("model");
    });
  });

  describe("inside a conversation", () => {
    it("shows the agent-profile picker for an unstarted conversation when profiles exist", () => {
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: true,
          hasStartedConversation: false,
          isCloud: false,
          isAcp: false,
          profilesAvailable: true,
        }),
      ).toBe("agent-profile");
    });

    it("uses the pre-run fallback for an unstarted conversation when profiles are unavailable", () => {
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: true,
          hasStartedConversation: false,
          isCloud: false,
          profilesAvailable: false,
        }),
      ).toBe("llm-profile");
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: true,
          hasStartedConversation: false,
          isCloud: true,
          profilesAvailable: false,
        }),
      ).toBe("model");
    });

    it("shows the model picker for an ACP conversation regardless of backend", () => {
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: true,
          hasStartedConversation: true,
          isCloud: false,
          isAcp: true,
        }),
      ).toBe("model");
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: true,
          hasStartedConversation: true,
          isCloud: true,
          isAcp: true,
        }),
      ).toBe("model");
    });

    it("shows the LLM-profile picker for an OpenHands conversation regardless of backend", () => {
      // /switch_profile is a real endpoint on both backends (cloud proxies
      // POST /api/v1/app-conversations/{id}/switch_profile) — no cloud
      // restriction here.
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: true,
          hasStartedConversation: true,
          isCloud: false,
          isAcp: false,
        }),
      ).toBe("llm-profile");
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: true,
          hasStartedConversation: true,
          isCloud: true,
          isAcp: false,
        }),
      ).toBe("llm-profile");
    });

    it("ignores profilesAvailable once a conversation is active", () => {
      expect(
        resolvePickerKind({
          ...base,
          hasConversation: true,
          hasStartedConversation: true,
          isCloud: false,
          isAcp: false,
          profilesAvailable: false,
        }),
      ).toBe("llm-profile");
    });
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
