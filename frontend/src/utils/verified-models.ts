// Here are the list of verified models and providers that we know work well with OpenHands.
export const VERIFIED_PROVIDERS = [
  "openhands",
  "anthropic",
  "openai",
  "mistral",
  "lemonade",
  "clarifai",
];
export const VERIFIED_MODELS = [
  "o3-mini-2025-01-31",
  "o3-2025-04-16",
  "o3",
  "o4-mini-2025-04-16",
  "claude-3-5-sonnet-20241022",
  "claude-3-7-sonnet-20250219",
  "claude-sonnet-4-20250514",
  "claude-sonnet-4-5-20250929",
  "claude-haiku-4-5-20251001",
  "claude-opus-4-20250514",
  "claude-opus-4-1-20250805",
  "claude-opus-4-5-20251101",
  "gemini-2.5-pro",
  "o4-mini",
  "deepseek-chat",
  "devstral-small-2505",
  "devstral-small-2507",
  "devstral-medium-2507",
  "kimi-k2-0711-preview",
  "qwen3-coder-480b",
  "gpt-5.2",
  "gpt-5.2-codex",
];

// LiteLLM does not return OpenAI models with the provider, so we list them here to set them ourselves for consistency
// (e.g., they return `gpt-4o` instead of `openai/gpt-4o`)
export const VERIFIED_OPENAI_MODELS = [
  "gpt-5.2",
  "gpt-5.2-codex",
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4.1",
  "gpt-4.1-2025-04-14",
  "o3",
  "o3-2025-04-16",
  "o4-mini",
  "o4-mini-2025-04-16",
  "codex-mini-latest",
];

// LiteLLM does not return the compatible Anthropic models with the provider, so we list them here to set them ourselves
// (e.g., they return `claude-3-5-sonnet-20241022` instead of `anthropic/claude-3-5-sonnet-20241022`)
export const VERIFIED_ANTHROPIC_MODELS = [
  "claude-3-5-sonnet-20240620",
  "claude-3-5-sonnet-20241022",
  "claude-3-5-haiku-20241022",
  "claude-3-7-sonnet-20250219",
  "claude-sonnet-4-20250514",
  "claude-sonnet-4-5-20250929",
  "claude-haiku-4-5-20251001",
  "claude-opus-4-20250514",
  "claude-opus-4-1-20250805",
  "claude-opus-4-5-20251101",
];

// LiteLLM does not return the compatible Mistral models with the provider, so we list them here to set them ourselves
// (e.g., they return `devstral-small-2505` instead of `mistral/devstral-small-2505`)
export const VERIFIED_MISTRAL_MODELS = [
  "devstral-small-2507",
  "devstral-medium-2507",
  "devstral-small-2505",
];

// LiteLLM does not return the compatible OpenHands models with the provider, so we list them here to set them ourselves
// (e.g., they return `claude-sonnet-4-20250514` instead of `openhands/claude-sonnet-4-20250514`)
export const VERIFIED_OPENHANDS_MODELS = [
  "claude-sonnet-4-20250514",
  "claude-sonnet-4-5-20250929",
  "claude-haiku-4-5-20251001",
  "gpt-5.2",
  "gpt-5.2-codex",
  "claude-opus-4-20250514",
  "claude-opus-4-1-20250805",
  "claude-opus-4-5-20251101",
  "gemini-2.5-pro",
  "o3",
  "o4-mini",
  "devstral-small-2507",
  "devstral-medium-2507",
  "devstral-small-2505",
  "kimi-k2-0711-preview",
  "qwen3-coder-480b",
];

// Default model for OpenHands provider
export const DEFAULT_OPENHANDS_MODEL = "openhands/claude-opus-4-5-20251101";
