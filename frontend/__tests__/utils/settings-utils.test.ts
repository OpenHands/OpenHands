import { describe, it, expect } from "vitest";
import { parseMaxBudgetPerTask, extractSettings } from "#/utils/settings-utils";

describe("parseMaxBudgetPerTask", () => {
  it("should return null for empty string", () => {
    expect(parseMaxBudgetPerTask("")).toBeNull();
  });

  it("should return null for whitespace-only string", () => {
    expect(parseMaxBudgetPerTask("   ")).toBeNull();
  });

  it("should return null for non-numeric string", () => {
    expect(parseMaxBudgetPerTask("abc")).toBeNull();
  });

  it("should return null for values less than 1", () => {
    expect(parseMaxBudgetPerTask("0")).toBeNull();
    expect(parseMaxBudgetPerTask("0.5")).toBeNull();
    expect(parseMaxBudgetPerTask("-1")).toBeNull();
    expect(parseMaxBudgetPerTask("-10.5")).toBeNull();
  });

  it("should return the parsed value for valid numbers >= 1", () => {
    expect(parseMaxBudgetPerTask("1")).toBe(1);
    expect(parseMaxBudgetPerTask("1.0")).toBe(1);
    expect(parseMaxBudgetPerTask("1.5")).toBe(1.5);
    expect(parseMaxBudgetPerTask("10")).toBe(10);
    expect(parseMaxBudgetPerTask("100.99")).toBe(100.99);
  });

  it("should handle string numbers with leading/trailing whitespace", () => {
    expect(parseMaxBudgetPerTask("  1  ")).toBe(1);
    expect(parseMaxBudgetPerTask("  10.5  ")).toBe(10.5);
  });

  it("should return null for edge cases", () => {
    expect(parseMaxBudgetPerTask("0.999")).toBeNull();
    expect(parseMaxBudgetPerTask("NaN")).toBeNull();
    expect(parseMaxBudgetPerTask("Infinity")).toBeNull();
    expect(parseMaxBudgetPerTask("-Infinity")).toBeNull();
  });

  it("should handle scientific notation", () => {
    expect(parseMaxBudgetPerTask("1e0")).toBe(1);
    expect(parseMaxBudgetPerTask("1.5e1")).toBe(15);
    expect(parseMaxBudgetPerTask("5e-1")).toBeNull(); // 0.5, which is < 1
  });
});

describe("extractSettings", () => {
  it("should preserve model name case when extracting settings", () => {
    // Test cases with various model name formats
    const testCases = [
      { provider: "sambanova", model: "Meta-Llama-3.1-8B-Instruct" },
      { provider: "openai", model: "GPT-4o" },
      { provider: "anthropic", model: "Claude-3-5-Sonnet" },
      { provider: "openrouter", model: "CamelCaseModel" },
    ];

    testCases.forEach(({ provider, model }) => {
      const formData = new FormData();
      formData.set("llm-provider-input", provider);
      formData.set("llm-model-input", model);

      const settings = extractSettings(formData);

      // Verify that the model name case is preserved
      const expectedModel = `${provider}/${model}`;
      expect(settings.llm_model).toBe(expectedModel);
      // Only test that it's not lowercased if the original has uppercase letters
      if (expectedModel !== expectedModel.toLowerCase()) {
        expect(settings.llm_model).not.toBe(expectedModel.toLowerCase());
      }
    });
  });

  it("should handle custom model without lowercasing", () => {
    const formData = new FormData();
    formData.set("llm-provider-input", "sambanova");
    formData.set("llm-model-input", "Meta-Llama-3.1-8B-Instruct");
    formData.set("use-advanced-options", "true");
    formData.set("custom-model", "Custom-Model-Name");

    const settings = extractSettings(formData);

    // Custom model should take precedence and preserve case
    expect(settings.llm_model).toBe("Custom-Model-Name");
    expect(settings.llm_model).not.toBe("custom-model-name");
  });
});
