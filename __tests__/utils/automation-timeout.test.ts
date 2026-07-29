import { describe, it, expect } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { validateAutomationTimeout } from "#/utils/automation-timeout";

describe("validateAutomationTimeout", () => {
  it("treats a blank value as 'use the server default'", () => {
    // Arrange / Act
    const result = validateAutomationTimeout("   ");

    // Assert
    expect(result).toEqual({ value: null });
  });

  it("accepts a positive integer up to the maximum", () => {
    // Arrange / Act
    const result = validateAutomationTimeout("1800");

    // Assert
    expect(result).toEqual({ value: 1800 });
  });

  it("rejects a non-integer value", () => {
    // Arrange / Act
    const result = validateAutomationTimeout("12.5");

    // Assert
    expect(result).toEqual({
      errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_INVALID_NUMBER,
    });
  });

  it("rejects a non-positive value", () => {
    // Arrange / Act
    const result = validateAutomationTimeout("0");

    // Assert
    expect(result).toEqual({
      errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_POSITIVE,
    });
  });

  it("rejects a value above the maximum", () => {
    // Arrange / Act
    const result = validateAutomationTimeout("1801");

    // Assert
    expect(result).toEqual({
      errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_MAX_EXCEEDED,
    });
  });
});
