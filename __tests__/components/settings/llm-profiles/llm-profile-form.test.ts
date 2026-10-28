import { describe, expect, it } from "vitest";
import {
  buildCreateProfileBaseConfig,
  buildProfileLlmConfig,
} from "#/components/features/settings/llm-profiles/llm-profile-form";

describe("buildCreateProfileBaseConfig", () => {
  it("returns an empty object when active settings are null or undefined", () => {
    expect(buildCreateProfileBaseConfig(null)).toEqual({});
    expect(buildCreateProfileBaseConfig(undefined)).toEqual({});
  });

  it("strips credentials, model, auth types, and connection id while preserving other options", () => {
    const activeLlmSettings = {
      model: "openai/gpt-4o",
      api_key: "sk-secret-key",
      base_url: "https://api.openai.com/v1",
      auth_type: "api_key",
      subscription_vendor: "openai",
      provider_connection_id: "conn-123",
      temperature: 0.4,
      max_output_tokens: 4096,
      top_p: 0.95,
    };

    const baseConfig = buildCreateProfileBaseConfig(activeLlmSettings);

    expect(baseConfig).toEqual({
      temperature: 0.4,
      max_output_tokens: 4096,
      top_p: 0.95,
    });
    expect(baseConfig).not.toHaveProperty("model");
    expect(baseConfig).not.toHaveProperty("api_key");
    expect(baseConfig).not.toHaveProperty("base_url");
    expect(baseConfig).not.toHaveProperty("auth_type");
    expect(baseConfig).not.toHaveProperty("subscription_vendor");
    expect(baseConfig).not.toHaveProperty("provider_connection_id");
  });

  it("integrates with buildProfileLlmConfig to retain non-dirty displayed options on create", () => {
    const baseConfig = buildCreateProfileBaseConfig({
      model: "openai/gpt-4o",
      api_key: "sk-secret",
      temperature: 0.4,
      max_output_tokens: 4096,
    });

    const { llmConfig } = buildProfileLlmConfig({
      baseConfig,
      dirtyLlm: {
        api_key: "new-key",
      },
      values: {
        "llm.model": "anthropic/claude-3-5-sonnet",
        "llm.api_key": "new-key",
      },
      view: "basic",
      supportsConnections: false,
      isCloud: false,
    });

    expect(llmConfig.model).toBe("anthropic/claude-3-5-sonnet");
    expect(llmConfig.temperature).toBe(0.4);
    expect(llmConfig.max_output_tokens).toBe(4096);
    expect(llmConfig.api_key).toBe("new-key");
  });
});
