import { describe, expect, it } from "vitest";
import { isCustomModel } from "#/utils/is-custom-model";

describe("isCustomModel", () => {
  const models = ["anthropic/claude-3.5", "openai/gpt-3.5-turbo", "gpt-4o"];
  const prefixedModels = ["anthropic/claude-3.5", "openai/gpt-4o"];

  it("should return false by default", () => {
    expect(isCustomModel(models, "")).toBe(false);
  });

  it("should be true if it is a custom model", () => {
    expect(isCustomModel(models, "some/model")).toBe(true);
  });

  it("should be false if it is not a custom model", () => {
    expect(isCustomModel(models, "anthropic/claude-3.5")).toBe(false);
    expect(isCustomModel(models, "openai/gpt-3.5-turbo")).toBe(false);
    expect(isCustomModel(models, "openai/gpt-4o")).toBe(false);
  });

  it("treats bare and prefixed OpenAI models as the same model", () => {
    expect(isCustomModel(prefixedModels, "gpt-4o")).toBe(false);
  });
});
