import { describe, it, expect } from "vitest";
import { extractSettings } from "#/utils/settings-utils";

function getLlmModel(settings: ReturnType<typeof extractSettings>): unknown {
  const as = settings.agent_settings as Record<string, unknown> | undefined;
  const llm = as?.llm as Record<string, unknown> | undefined;
  return llm?.model;
}

describe("Model name case preservation", () => {
  it("should preserve the original case of model names in extractSettings", () => {
    const formData = new FormData();
    formData.set("llm-provider-input", "SambaNova");
    formData.set("llm-model-input", "Meta-Llama-3.1-8B-Instruct");
    formData.set("agent", "CodeActAgent");
    formData.set("language", "en");

    const settings = extractSettings(formData);

    expect(getLlmModel(settings)).toBe(
      "SambaNova/Meta-Llama-3.1-8B-Instruct",
    );
  });

  it("should preserve openai model case", () => {
    const formData = new FormData();
    formData.set("llm-provider-input", "openai");
    formData.set("llm-model-input", "gpt-4o");
    formData.set("agent", "CodeActAgent");
    formData.set("language", "en");

    const settings = extractSettings(formData);
    expect(getLlmModel(settings)).toBe("openai/gpt-4o");
  });

  it("should preserve anthropic model case", () => {
    const formData = new FormData();
    formData.set("llm-provider-input", "anthropic");
    formData.set("llm-model-input", "claude-sonnet-4-20250514");
    formData.set("agent", "CodeActAgent");
    formData.set("language", "en");

    const settings = extractSettings(formData);
    expect(getLlmModel(settings)).toBe("anthropic/claude-sonnet-4-20250514");
  });

  it("should not automatically lowercase model names", () => {
    const formData = new FormData();
    formData.set("llm-provider-input", "SambaNova");
    formData.set("llm-model-input", "Meta-Llama-3.1-8B-Instruct");
    formData.set("agent", "CodeActAgent");
    formData.set("language", "en");

    const settings = extractSettings(formData);

    expect(getLlmModel(settings)).not.toBe(
      "sambanova/meta-llama-3.1-8b-instruct",
    );
    expect(getLlmModel(settings)).toBe(
      "SambaNova/Meta-Llama-3.1-8B-Instruct",
    );
  });
});
