import { describe, expect, it } from "vitest";
import { isUsableLlmProfile } from "#/utils/usable-llm-profile";

describe("isUsableLlmProfile", () => {
  it("treats an API-key profile as usable", () => {
    expect(
      isUsableLlmProfile({
        model: "openai/gpt-5.2",
        base_url: null,
        api_key_set: true,
      }),
    ).toBe(true);
  });

  it("treats a live provider-connection link as usable even without an inline key", () => {
    expect(
      isUsableLlmProfile({
        model: "ollama/llama3.2",
        base_url: null,
        api_key_set: false,
        provider_connection_id: "conn-ollama",
      }),
    ).toBe(true);
  });

  it("rejects a broken provider-connection link", () => {
    expect(
      isUsableLlmProfile({
        model: "ollama/llama3.2",
        base_url: null,
        api_key_set: false,
        provider_connection_id: "conn-gone",
        provider_connection_broken: true,
      }),
    ).toBe(false);
  });

  it("treats a keyless Ollama profile with a base URL as usable", () => {
    expect(
      isUsableLlmProfile({
        model: "ollama/qwen3-coder:16b",
        base_url: "http://gpu.home:11434",
        api_key_set: false,
      }),
    ).toBe(true);
  });

  it("treats a ChatGPT subscription profile as usable from detail config", () => {
    expect(
      isUsableLlmProfile(
        {
          model: "gpt-5.2",
          base_url: null,
          api_key_set: false,
        },
        {
          config: {
            auth_type: "subscription",
            subscription_vendor: "openai",
          },
        },
      ),
    ).toBe(true);
  });

  it("rejects a keyless, unlinked, non-subscription profile", () => {
    expect(
      isUsableLlmProfile({
        model: "openai/gpt-5.5",
        base_url: null,
        api_key_set: false,
      }),
    ).toBe(false);
  });
});
