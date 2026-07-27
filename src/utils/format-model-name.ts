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
  const lastSegment = model.split("/").pop();
  return lastSegment || model;
}
