import { describe, expect, it } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import {
  MIN_LLM_API_KEY_LENGTH,
  validateLlmApiKey,
  validateLlmBaseUrl,
  validateLlmCredentials,
} from "#/utils/validate-llm-credentials";

describe("validateLlmBaseUrl", () => {
  it.each([undefined, null, "", "   "])(
    "accepts %s (the field is optional)",
    (value) => {
      expect(validateLlmBaseUrl(value)).toBeNull();
    },
  );

  it.each([
    "https://api.openai.com",
    "https://api.openai.com/v1",
    "http://localhost:11434/v1",
    "http://127.0.0.1:8000",
    "  https://api.openai.com/v1  ",
  ])("accepts the valid base URL %s", (value) => {
    expect(validateLlmBaseUrl(value)).toBeNull();
  });

  it.each([".", "..", "not a url", "https://.", "https://..."])(
    "rejects the malformed base URL %s",
    (value) => {
      expect(validateLlmBaseUrl(value)).toBe(
        I18nKey.SETTINGS$LLM_BASE_URL_INVALID,
      );
    },
  );

  it.each(["localhost:11434", "ftp://example.com", "file:///tmp/x"])(
    "rejects %s because it is not http(s)",
    (value) => {
      expect(validateLlmBaseUrl(value)).toBe(
        I18nKey.SETTINGS$LLM_BASE_URL_INVALID_PROTOCOL,
      );
    },
  );
});

describe("validateLlmApiKey", () => {
  it.each([undefined, null, "", "   "])(
    "accepts %s — an empty key means 'leave the stored key unchanged'",
    (value) => {
      expect(validateLlmApiKey(value)).toBeNull();
    },
  );

  it.each([
    "sk-proj-abcdef0123456789",
    "EMPTY",
    "ollama",
    "sk-1234",
    "  sk-1234  ",
  ])("accepts the plausible API key %s", (value) => {
    expect(validateLlmApiKey(value)).toBeNull();
  });

  it.each(["-", ".", "ab", "sk"])(
    "rejects the too-short API key %s",
    (value) => {
      expect(validateLlmApiKey(value)).toBe(
        I18nKey.SETTINGS$LLM_API_KEY_TOO_SHORT,
      );
    },
  );

  it.each(["----", "......", "sk-1234 5678", "my key here"])(
    "rejects the malformed API key %s",
    (value) => {
      expect(validateLlmApiKey(value)).toBe(
        I18nKey.SETTINGS$LLM_API_KEY_INVALID,
      );
    },
  );

  it("does not reject keys at exactly the minimum length", () => {
    expect(validateLlmApiKey("a".repeat(MIN_LLM_API_KEY_LENGTH))).toBeNull();
    expect(
      validateLlmApiKey("a".repeat(MIN_LLM_API_KEY_LENGTH - 1)),
    ).not.toBeNull();
  });
});

describe("validateLlmCredentials", () => {
  it("ignores credential fields the config does not carry", () => {
    expect(validateLlmCredentials({})).toBeNull();
    expect(
      validateLlmCredentials({ api_key: null, base_url: undefined }),
    ).toBeNull();
  });

  it("accepts a valid config", () => {
    expect(
      validateLlmCredentials({
        api_key: "sk-proj-abcdef0123456789",
        base_url: "https://api.openai.com/v1",
      }),
    ).toBeNull();
  });

  it("reports the base URL failure first", () => {
    expect(validateLlmCredentials({ api_key: "-", base_url: "." })).toBe(
      I18nKey.SETTINGS$LLM_BASE_URL_INVALID,
    );
  });

  it("reports an API key failure when the base URL is fine", () => {
    expect(
      validateLlmCredentials({ api_key: "-", base_url: "https://x.example" }),
    ).toBe(I18nKey.SETTINGS$LLM_API_KEY_TOO_SHORT);
  });
});
