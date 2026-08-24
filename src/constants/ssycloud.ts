export const SSYCLOUD_PROVIDER_ID = "ssycloud";
export const SSYCLOUD_DISPLAY_NAME = "SSYCloud";
export const SSYCLOUD_BASE_URL = "https://router.shengsuanyun.com/api/v1";
export const SSYCLOUD_MODELS_URL = `${SSYCLOUD_BASE_URL}/models`;
export const SSYCLOUD_API_KEY_URL =
  "https://www.shengsuanyun.com/login?redirect=%2F%3Ffrom%3DCH_FGM22X9G";

const OPENAI_COMPATIBLE_PROVIDER_PREFIX = "openai/";

const normalizeBaseUrl = (baseUrl: string) =>
  baseUrl.trim().replace(/\/+$/, "");

export const isSSYCloudBaseUrl = (baseUrl: string | null | undefined) =>
  normalizeBaseUrl(baseUrl ?? "") === SSYCLOUD_BASE_URL;

/**
 * LiteLLM uses the first ``openai/`` segment as its routing discriminator.
 * SSYCloud model IDs already contain their upstream provider prefix, so a
 * model such as ``deepseek/deepseek-v4-flash`` is persisted as
 * ``openai/deepseek/deepseek-v4-flash``.
 */
export const toSSYCloudRuntimeModel = (modelId: string) =>
  `${OPENAI_COMPATIBLE_PROVIDER_PREFIX}${modelId}`;

export const fromSSYCloudRuntimeModel = (model: string) =>
  model.startsWith(OPENAI_COMPATIBLE_PROVIDER_PREFIX)
    ? model.slice(OPENAI_COMPATIBLE_PROVIDER_PREFIX.length)
    : model;

export const toSSYCloudSelectorModel = (runtimeModel: string) =>
  `${SSYCLOUD_PROVIDER_ID}/${fromSSYCloudRuntimeModel(runtimeModel)}`;
