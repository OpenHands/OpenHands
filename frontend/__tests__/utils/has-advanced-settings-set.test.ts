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
    test("llm.base_url is set", () => {
      const as = DEFAULT_SETTINGS.agent_settings as Record<string, unknown>;
      const llm = (as?.llm ?? {}) as Record<string, unknown>;
      expect(
        hasAdvancedSettingsSet({
          ...DEFAULT_SETTINGS,
          agent_settings: {
            ...as,
            llm: { ...llm, base_url: "test" },
          },
        }),
      ).toBe(true);
    });

    test("agent is not default value", () => {
      expect(
        hasAdvancedSettingsSet({
          ...DEFAULT_SETTINGS,
          agent_settings: {
            ...DEFAULT_SETTINGS.agent_settings,
            agent: "test",
          },
        }),
      ).toBe(true);
    });

    test("condenser.enabled is disabled", () => {
      const as = DEFAULT_SETTINGS.agent_settings as Record<string, unknown>;
      const condenser = (as?.condenser ?? {}) as Record<string, unknown>;
      const settings = {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...as,
          condenser: { ...condenser, enabled: false },
        },
      };

      const result = hasAdvancedSettingsSet(settings);

      expect(result).toBe(true);
    });

    test("condenser.max_size is customized above default", () => {
      const as = DEFAULT_SETTINGS.agent_settings as Record<string, unknown>;
      const condenser = (as?.condenser ?? {}) as Record<string, unknown>;
      const settings = {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...as,
          condenser: { ...condenser, max_size: 200 },
        },
      };

      const result = hasAdvancedSettingsSet(settings);

      expect(result).toBe(true);
    });

    test("condenser.max_size is customized below default", () => {
      const as = DEFAULT_SETTINGS.agent_settings as Record<string, unknown>;
      const condenser = (as?.condenser ?? {}) as Record<string, unknown>;
      const settings = {
        ...DEFAULT_SETTINGS,
        agent_settings: {
          ...as,
          condenser: { ...condenser, max_size: 50 },
        },
      };

      const result = hasAdvancedSettingsSet(settings);

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
