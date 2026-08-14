/**
 * Path constants for the agent-server Model Provider endpoints
 * (OpenHands/OpenHands#15492).
 *
 * Kept separate from the service so the service file never has a literal
 * "/api/..." string adjacent to its HTTP call (see the
 * no-direct-agent-server-calls guard), mirroring the LLM subscription
 * service pattern.
 *
 * Note the base path is `/model-providers`, not `/providers`: the latter is
 * the read-only list of *available provider kinds* used by the add-provider
 * picker, so the configured-provider records live under their own path.
 */
export const MODEL_PROVIDERS_PATH = "/api/llm/model-providers";
export const MODEL_PROVIDER_PATH = (id: string) =>
  `${MODEL_PROVIDERS_PATH}/${encodeURIComponent(id)}`;
export const MODEL_PROVIDER_TEST_PATH = (id: string) =>
  `${MODEL_PROVIDER_PATH(id)}/test`;
export const MODEL_PROVIDER_MODELS_PATH = (id: string) =>
  `${MODEL_PROVIDER_PATH(id)}/models`;
export const MODEL_PROVIDER_MODEL_PATH = (id: string, model: string) =>
  `${MODEL_PROVIDER_MODELS_PATH(id)}/${encodeURIComponent(model)}`;
