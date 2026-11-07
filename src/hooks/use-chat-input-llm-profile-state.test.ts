import { describe, expect, it } from "vitest";
import type { ProfileInfo } from "@openhands/typescript-client";
import { stampedProfileMatches } from "./use-chat-input-llm-profile-state";

const profiles = [
  { name: "gpt-profile", model: "gpt-6" },
  { name: "claude-profile", model: "claude-x" },
] as unknown as ProfileInfo[];

describe("stampedProfileMatches", () => {
  it("accepts the stamp while its model matches the backend model", () => {
    expect(stampedProfileMatches("gpt-profile", profiles, "gpt-6")).toBe(true);
  });

  it("rejects the stamp when the backend model has diverged", () => {
    expect(stampedProfileMatches("gpt-profile", profiles, "claude-x")).toBe(
      false,
    );
  });

  it("keeps the previous trust level when the backend model is unknown", () => {
    expect(stampedProfileMatches("gpt-profile", profiles, null)).toBe(true);
    expect(stampedProfileMatches("gpt-profile", profiles, undefined)).toBe(
      true,
    );
  });

  it("rejects a stamp naming a profile that no longer exists", () => {
    expect(stampedProfileMatches("deleted-profile", profiles, "gpt-6")).toBe(
      false,
    );
    expect(stampedProfileMatches("deleted-profile", profiles, null)).toBe(
      false,
    );
  });

  it("rejects an empty or missing stamp", () => {
    expect(stampedProfileMatches(null, profiles, null)).toBe(false);
    expect(stampedProfileMatches(undefined, profiles, null)).toBe(false);
    expect(stampedProfileMatches("", profiles, null)).toBe(false);
  });
});
