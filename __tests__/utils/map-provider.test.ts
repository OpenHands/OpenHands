import { test, expect } from "vitest";
import { getProviderId, mapProvider } from "../../src/utils/map-provider";

test("mapProvider", () => {
  expect(mapProvider("azure")).toBe("Azure");
  expect(mapProvider("azure_ai")).toBe("Azure AI Studio");
  expect(mapProvider("vertex_ai")).toBe("VertexAI");
  expect(mapProvider("palm")).toBe("PaLM");
  expect(mapProvider("gemini")).toBe("Gemini");
  expect(mapProvider("anthropic")).toBe("Anthropic");
  expect(mapProvider("sagemaker")).toBe("AWS SageMaker");
  expect(mapProvider("bedrock")).toBe("AWS Bedrock");
  expect(mapProvider("mistral")).toBe("Mistral AI");
  expect(mapProvider("anyscale")).toBe("Anyscale");
  expect(mapProvider("databricks")).toBe("Databricks");
  expect(mapProvider("ollama")).toBe("Ollama");
  expect(mapProvider("perplexity")).toBe("Perplexity AI");
  expect(mapProvider("friendliai")).toBe("FriendliAI");
  expect(mapProvider("groq")).toBe("Groq");
  expect(mapProvider("fireworks_ai")).toBe("Fireworks AI");
  expect(mapProvider("cloudflare")).toBe("Cloudflare Workers AI");
  expect(mapProvider("deepinfra")).toBe("DeepInfra");
  expect(mapProvider("ai21")).toBe("AI21");
  expect(mapProvider("replicate")).toBe("Replicate");
  expect(mapProvider("voyage")).toBe("Voyage AI");
  expect(mapProvider("openrouter")).toBe("OpenRouter");
  expect(mapProvider("clarifai")).toBe("Clarifai");
  expect(mapProvider("moonshot")).toBe("Moonshot");
});

test("getProviderId round-trips the display name back to the provider id", () => {
  // The backend addresses providers by their litellm id (`perplexity`), so the
  // display name must map back to that id, not to a misspelling of it.
  expect(getProviderId("Perplexity AI")).toBe("perplexity");
  expect(getProviderId("OpenAI")).toBe("openai");
  expect(getProviderId("Some New Provider")).toBe("Some New Provider");
});
