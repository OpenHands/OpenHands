import { describe, expect, it } from "vitest";
import {
  getProviderConnectionOption,
  listProviderConnectionOptions,
  resolveProviderConnectionFields,
} from "#/constants/provider-connection-options";

describe("provider connection options", () => {
  it("lists chat providers a pasted key or ChatGPT login can actually drive", () => {
    const ids = listProviderConnectionOptions().map((option) => option.id);

    expect(ids).toContain("openai");
    expect(ids).toContain("anthropic");
    expect(ids).toContain("cursor-cli");
    expect(ids).toContain("opencode");
    expect(ids).toContain("openhands");
    expect(ids).toContain("ollama");
    expect(ids).toContain("custom");

    expect(ids).not.toContain("azure");
    expect(ids).not.toContain("bedrock");
    expect(ids).not.toContain("vertex_ai");
    expect(ids).not.toContain("chatgpt");
    expect(ids).not.toContain("github_copilot");
    expect(ids).not.toContain("a2a");
  });

  it("offers ChatGPT on OpenAI, Claude login on Anthropic, and CLI login on Cursor/OpenCode", () => {
    expect(getProviderConnectionOption("openai")?.authModes).toEqual([
      "api_key",
      "openai_subscription",
    ]);
    expect(getProviderConnectionOption("anthropic")?.authModes).toEqual([
      "api_key",
      "anthropic_subscription",
    ]);
    expect(getProviderConnectionOption("cursor-cli")?.authModes).toEqual([
      "cli_subscription",
    ]);
    expect(getProviderConnectionOption("opencode")?.authModes).toEqual([
      "cli_subscription",
    ]);
  });

  it("requires a key for hosted APIs, a URL for Ollama/custom, and neither for ChatGPT login", () => {
    expect(resolveProviderConnectionFields("openai", "api_key")).toEqual({
      apiKey: "required",
      baseUrl: "optional",
      showOpenHandsHelp: false,
    });
    expect(
      resolveProviderConnectionFields("openai", "openai_subscription"),
    ).toEqual({
      apiKey: "hidden",
      baseUrl: "hidden",
      showOpenHandsHelp: false,
    });
    expect(resolveProviderConnectionFields("openhands", "api_key")).toEqual({
      apiKey: "required",
      baseUrl: "optional",
      showOpenHandsHelp: true,
    });
    expect(resolveProviderConnectionFields("ollama", "api_key")).toEqual({
      apiKey: "optional",
      baseUrl: "required",
      showOpenHandsHelp: false,
    });
    expect(resolveProviderConnectionFields("custom", "api_key")).toEqual({
      apiKey: "required",
      baseUrl: "required",
      showOpenHandsHelp: false,
    });
    expect(
      resolveProviderConnectionFields("anthropic", "anthropic_subscription"),
    ).toEqual({
      apiKey: "hidden",
      baseUrl: "hidden",
      showOpenHandsHelp: false,
    });
    expect(
      resolveProviderConnectionFields("cursor-cli", "cli_subscription"),
    ).toEqual({
      apiKey: "optional",
      baseUrl: "hidden",
      showOpenHandsHelp: false,
    });
    expect(
      resolveProviderConnectionFields("opencode", "cli_subscription"),
    ).toEqual({
      apiKey: "hidden",
      baseUrl: "hidden",
      showOpenHandsHelp: false,
    });
  });
});
