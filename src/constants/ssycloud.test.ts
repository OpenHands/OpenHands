import { describe, expect, it } from "vitest";
import {
  fromSSYCloudRuntimeModel,
  isSSYCloudBaseUrl,
  SSYCLOUD_BASE_URL,
  toSSYCloudRuntimeModel,
  toSSYCloudSelectorModel,
} from "./ssycloud";

describe("SSYCloud model configuration", () => {
  it("wraps SSYCloud model IDs with LiteLLM's OpenAI-compatible prefix", () => {
    expect(toSSYCloudRuntimeModel("deepseek/deepseek-v4-flash")).toBe(
      "openai/deepseek/deepseek-v4-flash",
    );
    expect(fromSSYCloudRuntimeModel("openai/openai/gpt-5.2")).toBe(
      "openai/gpt-5.2",
    );
  });

  it("restores the synthetic provider ID for the model selector", () => {
    expect(toSSYCloudSelectorModel("openai/anthropic/claude-sonnet-4.6")).toBe(
      "ssycloud/anthropic/claude-sonnet-4.6",
    );
  });

  it("recognizes the default base URL with optional trailing slashes", () => {
    expect(isSSYCloudBaseUrl(SSYCLOUD_BASE_URL)).toBe(true);
    expect(isSSYCloudBaseUrl(`${SSYCLOUD_BASE_URL}///`)).toBe(true);
    expect(isSSYCloudBaseUrl("https://api.openai.com/v1")).toBe(false);
  });
});
