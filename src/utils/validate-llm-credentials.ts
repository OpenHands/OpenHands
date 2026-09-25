import { I18nKey } from "#/i18n/declaration";

/**
 * Shortest LLM API key we accept. Real provider keys are far longer, but
 * OpenAI-compatible local servers are routinely configured with short
 * placeholders ("none", "EMPTY", "sk-1234"), so the floor only has to reject
 * input that is clearly not a key at all — e.g. the single "-" that saved
 * silently before this check existed.
 */
export const MIN_LLM_API_KEY_LENGTH = 4;

/** A key or URL that carries no letter or digit cannot be a real value. */
const ALPHANUMERIC = /[a-z0-9]/i;

const WHITESPACE = /\s/;

/**
 * Validate the LLM Base URL. Empty is valid — the field is optional and means
 * "use the provider default". Returns the translation key of the failure, or
 * `null` when the value is acceptable.
 */
export function validateLlmBaseUrl(
  baseUrl: string | null | undefined,
): I18nKey | null {
  const trimmed = baseUrl?.trim() ?? "";
  if (!trimmed) return null;

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return I18nKey.SETTINGS$LLM_BASE_URL_INVALID;
  }

  // A scheme-less "localhost:11434" parses with protocol "localhost:" and an
  // empty hostname, so both checks are needed to reject it.
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return I18nKey.SETTINGS$LLM_BASE_URL_INVALID_PROTOCOL;
  }
  if (!ALPHANUMERIC.test(url.hostname)) {
    return I18nKey.SETTINGS$LLM_BASE_URL_INVALID;
  }

  return null;
}

/**
 * Validate an LLM API key. Empty is valid — across the settings form and the
 * profile editor an empty key means "leave the stored key unchanged", never
 * "save an empty key". Returns the translation key of the failure, or `null`
 * when the value is acceptable.
 *
 * Failure messages interpolate `min`, so callers translate with
 * `t(key, { min: MIN_LLM_API_KEY_LENGTH })`.
 */
export function validateLlmApiKey(
  apiKey: string | null | undefined,
): I18nKey | null {
  const trimmed = apiKey?.trim() ?? "";
  if (!trimmed) return null;

  if (trimmed.length < MIN_LLM_API_KEY_LENGTH) {
    return I18nKey.SETTINGS$LLM_API_KEY_TOO_SHORT;
  }
  // Punctuation-only input ("----") and a key mangled by a bad paste (internal
  // whitespace) are never valid credentials.
  if (!ALPHANUMERIC.test(trimmed) || WHITESPACE.test(trimmed)) {
    return I18nKey.SETTINGS$LLM_API_KEY_INVALID;
  }

  return null;
}

/**
 * Validate the credential fields of an LLM config about to be persisted.
 * Only string values are checked, so a config whose `api_key` / `base_url` was
 * deliberately omitted (subscription auth, a provider-connection link, the
 * cloud OpenHands provider) is left alone.
 *
 * @returns the translation key of the first failure, or `null` when valid.
 */
export function validateLlmCredentials(llm: {
  api_key?: unknown;
  base_url?: unknown;
}): I18nKey | null {
  if (typeof llm.base_url === "string") {
    const baseUrlError = validateLlmBaseUrl(llm.base_url);
    if (baseUrlError) return baseUrlError;
  }
  if (typeof llm.api_key === "string") {
    const apiKeyError = validateLlmApiKey(llm.api_key);
    if (apiKeyError) return apiKeyError;
  }
  return null;
}
