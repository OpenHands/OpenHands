import { describe, expect, it } from "vitest";
import {
  getFirstLlmConfigValidationError,
  getLlmConfigValidationErrors,
} from "#/utils/llm-config-validation";

describe("LLM config validation", () => {
  it("allows provider-specific API keys with a reasonable length", () => {
    expect(getLlmConfigValidationErrors({ api_key: "provider-key" })).toEqual(
      {},
    );
  });

  it("rejects an API key that is too short", () => {
    expect(getLlmConfigValidationErrors({ api_key: "short" })).toMatchObject({
      apiKey: "API key must be at least 8 characters.",
    });
  });

  it("allows an empty optional base URL", () => {
    expect(getLlmConfigValidationErrors({ base_url: "  " })).toEqual({});
  });

  it.each(["not-a-url", "ftp://api.example.com", "https://"])(
    "rejects an invalid base URL: %s",
    (baseUrl) => {
      expect(getLlmConfigValidationErrors({ base_url: baseUrl })).toMatchObject(
        {
          baseUrl: "Base URL must be a valid HTTP(S) URL.",
        },
      );
    },
  );

  it("accepts local and hosted HTTP(S) endpoints", () => {
    expect(
      getLlmConfigValidationErrors({
        base_url: "http://localhost:8000/v1",
        api_key: "provider-key",
      }),
    ).toEqual({});
    expect(
      getFirstLlmConfigValidationError({
        base_url: "https://api.example.com/v1",
      }),
    ).toBeNull();
  });
});
