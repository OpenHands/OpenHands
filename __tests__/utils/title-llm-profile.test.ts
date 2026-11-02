import { describe, expect, it } from "vitest";
import { resolveTitleLlmProfile } from "#/utils/title-llm-profile";

const profiles = {
  profiles: [
    {
      name: "Default",
      model: "anthropic/claude-sonnet-4",
      base_url: null,
      api_key_set: true,
    },
    {
      name: "Titles",
      model: "anthropic/claude-haiku-3-5",
      base_url: null,
      api_key_set: true,
    },
  ],
  active_profile: "Default",
};

describe("resolveTitleLlmProfile", () => {
  it("uses an available explicit preference", () => {
    expect(resolveTitleLlmProfile("Titles", profiles)).toBe("Titles");
  });

  it("uses the active profile in automatic mode", () => {
    expect(resolveTitleLlmProfile(null, profiles)).toBe("Default");
  });

  it("falls back to the active profile when the preference is stale", () => {
    expect(resolveTitleLlmProfile("Deleted", profiles)).toBe("Default");
  });

  it("omits the value when no profile can be resolved", () => {
    expect(
      resolveTitleLlmProfile(null, {
        profiles: [],
        active_profile: null,
      }),
    ).toBeUndefined();
  });

  describe("agentLlmProfileRef priority", () => {
    it("uses agentLlmProfileRef when no explicit preference is set", () => {
      expect(resolveTitleLlmProfile(null, profiles, "Titles")).toBe("Titles");
    });

    it("explicit preference takes priority over agentLlmProfileRef", () => {
      expect(resolveTitleLlmProfile("Titles", profiles, "Default")).toBe(
        "Titles",
      );
    });

    it("falls back to active_profile when agentLlmProfileRef is not in the list", () => {
      expect(resolveTitleLlmProfile(null, profiles, "Deleted")).toBe("Default");
    });

    it("ignores null agentLlmProfileRef and falls back to active_profile", () => {
      expect(resolveTitleLlmProfile(null, profiles, null)).toBe("Default");
    });
  });
});
