import { describe, expect, it, test } from "vitest";
import { hasAdvancedSettingsSet } from "#/utils/has-advanced-settings-set";
import { DEFAULT_SETTINGS } from "#/services/settings";

describe("hasAdvancedSettingsSet", () => {
  it("should return false by default", () => {
    expect(hasAdvancedSettingsSet(DEFAULT_SETTINGS)).toBe(false);
  });

  it("should return false if an empty object", () => {
    expect(hasAdvancedSettingsSet({})).toBe(false);
  });

  describe("should be true if", () => {
    test("llm_base_url is set", () => {
      expect(
        hasAdvancedSettingsSet({
          ...DEFAULT_SETTINGS,
          llm_base_url: "test",
        }),
      ).toBe(true);
    });

    test("agent is not default value", () => {
      expect(
        hasAdvancedSettingsSet({
          ...DEFAULT_SETTINGS,
          agent: "test",
        }),
      ).toBe(true);
    });
  });
});
