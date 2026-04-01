import { describe, it, expect } from "vitest";
import { extractModelAndProvider } from "../../src/utils/extract-model-and-provider";

describe("extractModelAndProvider", () => {
  it("should split on / separator", () => {
    expect(extractModelAndProvider("azure/ada")).toEqual({
      provider: "azure",
      model: "ada",
      separator: "/",
    });

    expect(
      extractModelAndProvider("azure/standard/1024-x-1024/dall-e-2"),
    ).toEqual({
      provider: "azure",
      model: "standard/1024-x-1024/dall-e-2",
      separator: "/",
    });

    expect(extractModelAndProvider("vertex_ai_beta/chat-bison")).toEqual({
      provider: "vertex_ai_beta",
      model: "chat-bison",
      separator: "/",
    });

    expect(
      extractModelAndProvider(
        "cloudflare/@cf/mistral/mistral-7b-instruct-v0.1",
      ),
    ).toEqual({
      provider: "cloudflare",
      model: "@cf/mistral/mistral-7b-instruct-v0.1",
      separator: "/",
    });
  });

  it("should return dotted or bare models as-is", () => {
    expect(extractModelAndProvider("cohere.command-r-v1:0")).toEqual({
      provider: "",
      model: "cohere.command-r-v1:0",
      separator: "",
    });

    expect(extractModelAndProvider("together-ai-21.1b-41b")).toEqual({
      provider: "",
      model: "together-ai-21.1b-41b",
      separator: "",
    });

    expect(extractModelAndProvider("gpt-4o-mini")).toEqual({
      provider: "",
      model: "gpt-4o-mini",
      separator: "",
    });

    expect(extractModelAndProvider("claude-opus-4-6")).toEqual({
      provider: "",
      model: "claude-opus-4-6",
      separator: "",
    });
  });

  it("should correctly parse already-prefixed models", () => {
    expect(extractModelAndProvider("openai/gpt-5.2")).toEqual({
      provider: "openai",
      model: "gpt-5.2",
      separator: "/",
    });

    expect(extractModelAndProvider("anthropic/claude-opus-4-6")).toEqual({
      provider: "anthropic",
      model: "claude-opus-4-6",
      separator: "/",
    });

    expect(extractModelAndProvider("openhands/claude-opus-4-5-20251101")).toEqual({
      provider: "openhands",
      model: "claude-opus-4-5-20251101",
      separator: "/",
    });
  });
});
