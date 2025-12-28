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

    test("enable_default_condenser is disabled", () => {
      // Arrange
      const settings = {
        ...DEFAULT_SETTINGS,
        enable_default_condenser: false,
      };

      // Act
      const result = hasAdvancedSettingsSet(settings);

      // Assert
      expect(result).toBe(true);
    });

    test("condenser_max_size is customized above default", () => {
      // Arrange
      const settings = {
        ...DEFAULT_SETTINGS,
        condenser_max_size: 200,
      };

      // Act
      const result = hasAdvancedSettingsSet(settings);

      // Assert
      expect(result).toBe(true);
    });

    test("condenser_max_size is customized below default", () => {
      // Arrange
      const settings = {
        ...DEFAULT_SETTINGS,
        condenser_max_size: 50,
      };

      // Act
      const result = hasAdvancedSettingsSet(settings);

      // Assert
      expect(result).toBe(true);
    });

    test("search_api_key is set to non-empty value", () => {
      // Arrange
      const settings = {
        ...DEFAULT_SETTINGS,
        search_api_key: "test-api-key-123",
      };

      // Act
      const result = hasAdvancedSettingsSet(settings);

      // Assert
      expect(result).toBe(true);
    });

    test("search_api_key with whitespace is treated as set", () => {
      // Arrange
      const settings = {
        ...DEFAULT_SETTINGS,
        search_api_key: "  test-key  ",
      };

      // Act
      const result = hasAdvancedSettingsSet(settings);

      // Assert
      expect(result).toBe(true);
    });
  });
});
