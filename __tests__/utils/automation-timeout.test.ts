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

  it("accepts a positive integer up to a deployment-provided maximum", () => {
    // Arrange / Act
    const result = validateAutomationTimeout("900", 900);

    // Assert
    expect(result).toEqual({ value: 900 });
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
    const result = validateAutomationTimeout("901", 900);

    // Assert
    expect(result).toEqual({
      errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_MAX_EXCEEDED,
    });
  });

  it("validates against a ceiling the interface manifest lowered", () => {
    // Arrange / Act — a manifest max of 900 moves the boundary: 900 stays
    // in range while 901, fine under the 1800 default, is now rejected.
    const atCeiling = validateAutomationTimeout("900", 900);
    const overCeiling = validateAutomationTimeout("901", 900);

    // Assert
    expect(atCeiling).toEqual({ value: 900 });
    expect(overCeiling).toEqual({
      errorKey: I18nKey.AUTOMATIONS$ERROR_TIMEOUT_MAX_EXCEEDED,
    });
  });
});
