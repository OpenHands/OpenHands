import { describe, expect, it } from "vitest";
import type { ProfileListResponse } from "#/api/profiles-service/profiles-service.api";
import { resolveTitleLlmProfile } from "./title-llm-profile";

const profiles = {
  profiles: [{ name: "fast" }, { name: "powerful" }],
  active_profile: "fast",
} as unknown as ProfileListResponse;

describe("resolveTitleLlmProfile", () => {
  it("prefers the user's explicit title model over everything else", () => {
    expect(resolveTitleLlmProfile("powerful", profiles, "fast")).toBe(
      "powerful",
    );
  });

  it("falls back to the agent profile's pinned LLM, not the active one", () => {
    expect(resolveTitleLlmProfile(null, profiles, "powerful")).toBe("powerful");
  });

  it("falls back to the active profile when the agent has no pinned LLM", () => {
    expect(resolveTitleLlmProfile(null, profiles, null)).toBe("fast");
  });

  it("ignores refs that name a profile which no longer exists", () => {
    expect(resolveTitleLlmProfile(null, profiles, "deleted")).toBe("fast");
  });

  it("returns undefined without a profile list", () => {
    expect(
      resolveTitleLlmProfile("powerful", undefined, "fast"),
    ).toBeUndefined();
  });
});
