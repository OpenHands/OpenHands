import { describe, expect, it } from "vitest";
import {
  buildProfileLlmConfig,
  isOpenRouterBaseUrl,
  qualifyOpenRouterModel,
} from "#/components/features/settings/llm-profiles/llm-profile-form";

const OPENROUTER_CONNECTION = { id: "conn-or", provider: "openrouter" };
const OPENAI_CONNECTION = { id: "conn-oa", provider: "openai" };

function build(llm: Record<string, unknown>, connectionId = "") {
  return buildProfileLlmConfig({
    baseConfig: {},
    dirtyLlm: llm,
    values: {
      "llm.model": llm.model as string,
      "llm.provider_connection_id": connectionId,
    },
    view: "advanced",
    supportsConnections: true,
    isCloud: false,
    providerConnections: [OPENROUTER_CONNECTION, OPENAI_CONNECTION],
  }).llmConfig;
}

describe("OpenRouter model qualification", () => {
  it.each([
    ["https://openrouter.ai/api/v1", true],
    ["https://eu.openrouter.ai/api/v1", true],
    [" https://openrouter.ai/api/v1 ", true],
    ["https://api.openai.com/v1", false],
    ["https://openrouter.ai.example.com/v1", false],
    ["not a url", false],
    ["", false],
    [null, false],
  ])("isOpenRouterBaseUrl(%j) is %s", (url, expected) => {
    expect(isOpenRouterBaseUrl(url)).toBe(expected);
  });

  it("prefixes an OpenRouter model ID once", () => {
    expect(qualifyOpenRouterModel("thinkingmachines/inkling-small:free")).toBe(
      "openrouter/thinkingmachines/inkling-small:free",
    );
    expect(qualifyOpenRouterModel("openrouter/auto")).toBe("openrouter/auto");
  });

  it("qualifies the model of a profile linked to an OpenRouter connection", () => {
    expect(
      build({ model: "thinkingmachines/inkling-small:free" }, "conn-or").model,
    ).toBe("openrouter/thinkingmachines/inkling-small:free");
  });

  it("qualifies the model of a profile whose base URL is OpenRouter", () => {
    const config = build({
      model: "z-ai/glm-4.6",
      base_url: "https://openrouter.ai/api/v1",
      api_key: "sk-test",
    });
    expect(config.model).toBe("openrouter/z-ai/glm-4.6");
    expect(config.base_url).toBe("https://openrouter.ai/api/v1");
  });

  it("keeps a model that already has the openrouter/ prefix", () => {
    expect(
      build({ model: "openrouter/anthropic/claude-sonnet-4" }, "conn-or").model,
    ).toBe("openrouter/anthropic/claude-sonnet-4");
  });

  it.each([
    ["another provider's connection", { model: "gpt-4o" }, "conn-oa"],
    [
      "a non-OpenRouter base URL",
      { model: "qwen/qwen3-coder", base_url: "http://localhost:11434/v1" },
      "",
    ],
    ["no base URL", { model: "anthropic/claude-sonnet-4" }, ""],
    [
      "an unknown connection, even with an OpenRouter base URL left in the form",
      {
        model: "anthropic/claude-sonnet-4",
        base_url: "https://openrouter.ai/api/v1",
      },
      "conn-unknown",
    ],
  ])("leaves the model unchanged for %s", (_label, llm, connectionId) => {
    expect(build(llm, connectionId).model).toBe(llm.model);
  });
});
