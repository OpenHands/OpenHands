export interface LlmConfigValidationErrors {
  apiKey?: string;
  baseUrl?: string;
}

const MIN_API_KEY_LENGTH = 8;

export function getLlmConfigValidationErrors(
  config: Record<string, unknown>,
): LlmConfigValidationErrors {
  const errors: LlmConfigValidationErrors = {};
  const rawApiKey = config.api_key ?? config["llm.api_key"];
  const rawBaseUrl = config.base_url ?? config["llm.base_url"];
  const apiKey = typeof rawApiKey === "string" ? rawApiKey.trim() : "";
  const baseUrl = typeof rawBaseUrl === "string" ? rawBaseUrl.trim() : "";

  // API keys differ between providers, so validate only the minimum useful
  // shape instead of enforcing a provider-specific prefix.
  if (apiKey && apiKey.length < MIN_API_KEY_LENGTH) {
    errors.apiKey = "API key must be at least 8 characters.";
  }

  if (baseUrl) {
    try {
      const parsedUrl = new URL(baseUrl);
      if (!parsedUrl.hostname || !/^https?:$/.test(parsedUrl.protocol)) {
        errors.baseUrl = "Base URL must be a valid HTTP(S) URL.";
      }
    } catch {
      errors.baseUrl = "Base URL must be a valid HTTP(S) URL.";
    }
  }

  return errors;
}

export function getFirstLlmConfigValidationError(
  config: Record<string, unknown>,
): string | null {
  const errors = getLlmConfigValidationErrors(config);
  return errors.baseUrl ?? errors.apiKey ?? null;
}
