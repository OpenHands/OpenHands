export const OPENHANDS_FREE_GLM_MODEL_ID = "openhands/glm-5.2";
export const OPENHANDS_FREE_GLM_MODEL_LABEL = "OpenHands GLM-5.2 (free)";
export const OPENHANDS_FREE_GLM_BADGE_LABEL = "Free";
export const OPENHANDS_FREE_GLM_MODEL_NOTE =
  "Only openhands/glm-5.2 is free. Other GLM-5.2 endpoints may require their own billing.";

export function isOpenHandsFreeGlmModel(
  model: string | null | undefined,
): boolean {
  return model === OPENHANDS_FREE_GLM_MODEL_ID;
}

export function formatModelNameForDisplay(
  model: string | null | undefined,
): string | null {
  if (!model) return null;
  return isOpenHandsFreeGlmModel(model)
    ? OPENHANDS_FREE_GLM_MODEL_LABEL
    : model;
}

export function formatProviderModelNameForDisplay(
  provider: string | null | undefined,
  model: string | null | undefined,
): string | null {
  if (!model) return null;
  const fullModel = provider ? `${provider}/${model}` : model;
  return isOpenHandsFreeGlmModel(fullModel)
    ? OPENHANDS_FREE_GLM_MODEL_LABEL
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
  if (isOpenHandsFreeGlmModel(model)) return OPENHANDS_FREE_GLM_MODEL_LABEL;
  const lastSegment = model.split("/").pop();
  return lastSegment || model;
}
