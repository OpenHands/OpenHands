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
  "claude-opus-4-5-20251101",
  "claude-sonnet-4-5-20250929",
  "gpt-5.2-codex",
  "gpt-5.2",
  "minimax-m2.5",
  "gemini-3-pro-preview",
  "gemini-3-flash-preview",
  "deepseek-chat",
  "devstral-medium-2512",
  "kimi-k2-0711-preview",
  "qwen3-coder-480b",
];

// LiteLLM does not return OpenAI models with the provider, so we list them here to set them ourselves for consistency
// (e.g., they return `gpt-4o` instead of `openai/gpt-4o`)
export const VERIFIED_OPENAI_MODELS = [
  "gpt-5.2",
  "gpt-5.2-codex",
  "gpt-4o",
  "gpt-4o-mini",
];

// LiteLLM does not return the compatible Anthropic models with the provider, so we list them here to set them ourselves
// (e.g., they return `claude-3-5-sonnet-20241022` instead of `anthropic/claude-3-5-sonnet-20241022`)
export const VERIFIED_ANTHROPIC_MODELS = [
  "claude-opus-4-5-20251101",
  "claude-sonnet-4-5-20250929",
  "claude-3-5-sonnet-20240620",
  "claude-3-5-sonnet-20241022",
  "claude-3-7-sonnet-20250219",
  "claude-sonnet-4-20250514",
  "claude-opus-4-20250514",
  "claude-opus-4-1-20250805",
];

// LiteLLM does not return the compatible Mistral models with the provider, so we list them here to set them ourselves
// (e.g., they return `devstral-small-2505` instead of `mistral/devstral-small-2505`)
export const VERIFIED_MISTRAL_MODELS = ["devstral-medium-2512"];

// LiteLLM does not return the compatible OpenHands models with the provider, so we list them here to set them ourselves
// (e.g., they return `claude-sonnet-4-20250514` instead of `openhands/claude-sonnet-4-20250514`)
export const VERIFIED_OPENHANDS_MODELS = [
  "claude-opus-4-5-20251101",
  "claude-sonnet-4-5-20250929",
  "gpt-5.2-codex",
  "gpt-5.2",
  "minimax-m2.5",
  "gemini-3-pro-preview",
  "gemini-3-flash-preview",
  "devstral-medium-2512",
  "kimi-k2-0711-preview",
  "qwen3-coder-480b",
];

// Default model for OpenHands provider
export const DEFAULT_OPENHANDS_MODEL = "openhands/claude-opus-4-5-20251101";
