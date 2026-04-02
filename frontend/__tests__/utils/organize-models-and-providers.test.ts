import { expect, test } from "vitest";
import { organizeModelsAndProviders } from "../../src/utils/organize-models-and-providers";

test("organizeModelsAndProviders — models already prefixed by backend", () => {
  // The backend now assigns provider prefixes to bare LiteLLM names.
  // The frontend only needs to parse slash-prefixed provider/model strings.
  const models = [
    "azure/ada",
    "azure/gpt-35-turbo",
    "azure/gpt-3-turbo",
    "azure/standard/1024-x-1024/dall-e-2",
    "vertex_ai_beta/chat-bison",
    "vertex_ai_beta/chat-bison-32k",
    "sagemaker/meta-textgeneration-llama-2-13b",
    "cohere.command-r-v1:0",
    "cloudflare/@cf/mistral/mistral-7b-instruct-v0.1",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20241022",
    "together-ai-21.1b-41b",
  ];

  const object = organizeModelsAndProviders(models);

  expect(object).toEqual({
    azure: {
      separator: "/",
      models: [
        "ada",
        "gpt-35-turbo",
        "gpt-3-turbo",
        "standard/1024-x-1024/dall-e-2",
      ],
    },
    vertex_ai_beta: {
      separator: "/",
      models: ["chat-bison", "chat-bison-32k"],
    },
    sagemaker: { separator: "/", models: ["meta-textgeneration-llama-2-13b"] },
    cloudflare: {
      separator: "/",
      models: ["@cf/mistral/mistral-7b-instruct-v0.1"],
    },
    openai: {
      separator: "/",
      models: ["gpt-4o", "gpt-4o-mini"],
    },
    anthropic: {
      separator: "/",
      models: ["claude-3-5-sonnet-20241022"],
    },
    other: {
      separator: "",
      models: ["cohere.command-r-v1:0", "together-ai-21.1b-41b"],
    },
  });
});
