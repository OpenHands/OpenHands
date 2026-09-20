import { extractModelAndProvider } from "./extract-model-and-provider";

const PROVIDER_DEFAULT_BASE_URLS: Partial<Record<string, readonly string[]>> = {
  openai: ["https://api.openai.com", "https://api.openai.com/v1"],
  moonshot: ["https://api.kimi.com/coding/v1"],
  openrouter: ["https://openrouter.ai/api/v1"],
};
const DEFAULT_API_KEY_HELP_URL =
  "https://docs.openhands.dev/usage/local-setup#getting-an-api-key";
const PROVIDER_API_KEY_URLS: Partial<Record<string, string>> = {
  openrouter: "https://openrouter.ai/keys",
};

export function buildModelId(provider: string | null, model: string | null) {
  const selectedProvider = provider?.trim();
  const modelId = model?.trim();
  if (!selectedProvider || !modelId) return null;
  const prefix = `${selectedProvider}/`;
  // OpenRouter itself is also a model namespace (e.g. openrouter/auto).
  // Its SDK-qualified IDs have a provider prefix AND a vendor/model path.
  const isQualified =
    modelId.startsWith(prefix) &&
    (selectedProvider !== "openrouter" ||
      modelId.slice(prefix.length).includes("/"));
  return isQualified ? modelId : `${prefix}${modelId}`;
}

export function isProviderDefaultBaseUrl(model: string, baseUrl: string) {
  const { provider } = extractModelAndProvider(model);
  try {
    const url = new URL(baseUrl);
    if (url.search || url.hash || url.username || url.password) return false;
    const normalized = `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
    return PROVIDER_DEFAULT_BASE_URLS[provider]?.includes(normalized) ?? false;
  } catch {
    return false;
  }
}

export function getProviderApiKeyHelpUrl(model: string) {
  const { provider } = extractModelAndProvider(model);
  return PROVIDER_API_KEY_URLS[provider] ?? DEFAULT_API_KEY_HELP_URL;
}

export function getProviderBaseUrlPlaceholder(provider: string | null) {
  return (
    PROVIDER_DEFAULT_BASE_URLS[provider ?? ""]?.[0] ??
    PROVIDER_DEFAULT_BASE_URLS.openai![0]
  );
}
