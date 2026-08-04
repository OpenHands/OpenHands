export const FREE_MODEL_BADGE_LABEL = "Free";

export const FREE_OPENHANDS_MODELS = {
  "openhands/glm-5.2": "OpenHands GLM-5.2 (free)",
  "openhands/deepseek-v4-flash": "OpenHands DeepSeek V4 Flash (free)",
  "openhands/minimax-m2.7": "OpenHands MiniMax M2.7 (free)",
} as const;

export const FREE_OPENHANDS_MODEL_IDS = Object.keys(FREE_OPENHANDS_MODELS);
export const FREE_OPENHANDS_MODEL_NOTE = `Free OpenHands models: ${FREE_OPENHANDS_MODEL_IDS.join(
  ", ",
)}. Other provider endpoints with similar model names may require separate billing.`;

export const isFreeOpenHandsModel = (
  model: string | null | undefined,
): model is keyof typeof FREE_OPENHANDS_MODELS =>
  Boolean(model && model in FREE_OPENHANDS_MODELS);

export function formatModelNameForDisplay(
  model: string | null | undefined,
): string | null {
  if (!model) return null;
  return isFreeOpenHandsModel(model) ? FREE_OPENHANDS_MODELS[model] : model;
}

export function formatProviderModelNameForDisplay(
  provider: string | null | undefined,
  model: string | null | undefined,
): string | null {
  if (!model) return null;
  const fullModel = provider ? `${provider}/${model}` : model;
  return isFreeOpenHandsModel(fullModel)
    ? FREE_OPENHANDS_MODELS[fullModel]
    : model;
}

/**
 * Format a native (OpenHands-kind) routing model string for display, stripping
 * the provider route prefix (e.g. ``"anthropic/claude-sonnet-4-5-20250929"`` →
 * ``"claude-sonnet-4-5-20250929"``, ``"litellm_proxy/openai/gpt-4o"`` →
 * ``"gpt-4o"``) so a conversation chip shows a meaningful model name rather than
 * the full routing path.
 *
 * Returns ``null`` for an empty/nullish input, and falls back to the original
 * string when stripping the prefix would leave nothing (e.g. a trailing slash)
 * — never an empty string, which would collapse the chip text.
 *
 * Display-only: unlike {@link deriveProfileNameFromModel} this does not sanitize
 * to an identifier, so it keeps the real model id intact for the chip.
 */
export function formatNativeModelName(
  model: string | null | undefined,
): string | null {
  if (!model) return null;
  if (isFreeOpenHandsModel(model)) return FREE_OPENHANDS_MODELS[model];
  const lastSegment = model.split("/").pop();
  return lastSegment || model;
}
