import { afterEach, describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import { server } from "#/mocks/node";
import { fetchOpenRouterModels } from "./openrouter-models-service";

const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";

afterEach(() => {
  server.resetHandlers();
});

describe("fetchOpenRouterModels", () => {
  it("parses capability metadata alongside each catalog id", async () => {
    server.use(
      http.get(OPENROUTER_MODELS_URL, () =>
        HttpResponse.json({
          data: [
            {
              id: "anthropic/claude-3.7-sonnet",
              context_length: 200000,
              supported_parameters: ["tools", "reasoning", "temperature"],
              architecture: {
                input_modalities: ["text", "image"],
                output_modalities: ["text"],
              },
              top_provider: { max_completion_tokens: 8192 },
              pricing: { prompt: "0.000003", completion: "0.000015" },
            },
          ],
        }),
      ),
    );

    const models = await fetchOpenRouterModels();

    expect(models).toEqual([
      {
        id: "anthropic/claude-3.7-sonnet",
        contextLength: 200000,
        maxOutputTokens: 8192,
        supportedParameters: ["tools", "reasoning", "temperature"],
        inputModalities: ["text", "image"],
        outputModalities: ["text"],
        pricing: { prompt: "0.000003", completion: "0.000015" },
      },
    ]);
  });

  it("returns only the id when OpenRouter omits capability fields, never inventing values", async () => {
    server.use(
      http.get(OPENROUTER_MODELS_URL, () =>
        HttpResponse.json({ data: [{ id: "vendor/bare-model" }] }),
      ),
    );

    const models = await fetchOpenRouterModels();

    expect(models).toEqual([{ id: "vendor/bare-model" }]);
  });

  it("preserves an explicit empty supported_parameters as known-empty, not unknown", async () => {
    server.use(
      http.get(OPENROUTER_MODELS_URL, () =>
        HttpResponse.json({
          data: [
            { id: "vendor/no-parameters-published", supported_parameters: [] },
          ],
        }),
      ),
    );

    const models = await fetchOpenRouterModels();

    expect(models).toEqual([
      { id: "vendor/no-parameters-published", supportedParameters: [] },
    ]);
  });

  it("ignores malformed capability fields instead of throwing", async () => {
    server.use(
      http.get(OPENROUTER_MODELS_URL, () =>
        HttpResponse.json({
          data: [
            {
              id: "vendor/odd-model",
              context_length: "not-a-number",
              supported_parameters: "tools",
              architecture: { input_modalities: [1, 2] },
              pricing: "not-an-object",
            },
          ],
        }),
      ),
    );

    const models = await fetchOpenRouterModels();

    expect(models).toEqual([{ id: "vendor/odd-model" }]);
  });

  it("dedupes repeated ids, keeping first-seen order with the latest metadata", async () => {
    server.use(
      http.get(OPENROUTER_MODELS_URL, () =>
        HttpResponse.json({
          data: [
            { id: "vendor/model-a" },
            { id: "vendor/model-b", context_length: 4096 },
            { id: "vendor/model-a", context_length: 128000 },
          ],
        }),
      ),
    );

    const models = await fetchOpenRouterModels();

    expect(models.map((m) => m.id)).toEqual([
      "vendor/model-a",
      "vendor/model-b",
    ]);
    expect(models[0]).toEqual({
      id: "vendor/model-a",
      contextLength: 128000,
    });
  });

  it("rejects a catalog whose entries have no usable id", async () => {
    server.use(
      http.get(OPENROUTER_MODELS_URL, () =>
        HttpResponse.json({ data: [{ name: "not-an-id" }] }),
      ),
    );

    await expect(fetchOpenRouterModels()).rejects.toThrow(
      "OpenRouter returned an invalid model catalog",
    );
  });

  it("never sends credentials or an authorization header", async () => {
    let authorization: string | null | undefined;
    let cookieHeaderSeen = false;
    server.use(
      http.get(OPENROUTER_MODELS_URL, ({ request }) => {
        authorization = request.headers.get("authorization");
        cookieHeaderSeen = request.headers.has("cookie");
        return HttpResponse.json({ data: [{ id: "vendor/model" }] });
      }),
    );

    await fetchOpenRouterModels();

    expect(authorization).toBeNull();
    expect(cookieHeaderSeen).toBe(false);
  });
});
