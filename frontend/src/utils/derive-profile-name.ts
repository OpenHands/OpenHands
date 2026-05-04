// Backend-enforced pattern: ^[A-Za-z0-9._-]{1,64}$
// Matches settings_router.py — keep them in sync.
const ALLOWED_CHARS = /[^A-Za-z0-9._-]/g;

export const PROFILE_NAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;

/**
 * Build a profile name from an LLM model string.
 *
 * Examples:
 *   "openai/gpt-4o"                -> "openai_gpt-4o"
 *   "anthropic/claude-3-5-sonnet"  -> "anthropic_claude-3-5-sonnet"
 *   "openai/gpt-4o:custom"         -> "openai_gpt-4o_custom"
 *
 * Returns null if the input has no usable characters (e.g. empty or all
 * non-alphanumeric). The caller should then skip the auto-profile step
 * rather than falling back to a generic placeholder name.
 */
export function deriveProfileNameFromModel(model: string): string | null {
  // "/" is the canonical provider separator; mapping it to "_" keeps the
  // human-readable boundary without running afoul of the backend regex.
  const sanitized = model
    .trim()
    .replace(/\//g, "_")
    .replace(ALLOWED_CHARS, "_")
    .replace(/_+/g, "_")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, 64);
  return sanitized.length > 0 ? sanitized : null;
}
