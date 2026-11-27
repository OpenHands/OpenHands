/**
 * Profile name validation pattern.
 * Profile names: 1-64 chars, must start with alphanumeric, then alphanumerics
 * or '.', '_', '-'. Blocks empty names, path separators, leading dots
 * (hidden files / path traversal), and shell-special characters.
 */
export const PROFILE_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

/**
 * Shared profile name validation. Any whitespace (including leading/trailing)
 * makes the value invalid. When `isRequired`, empty values are invalid;
 * otherwise empty is valid.
 */
export function isProfileNameValid(
  value: string,
  { isRequired = false }: { isRequired?: boolean } = {},
): boolean {
  if (value === "") return !isRequired;
  return PROFILE_NAME_PATTERN.test(value);
}

/**
 * Derive a profile name from a model string.
 * Extracts the model portion after the provider prefix (e.g., "openai/gpt-4" -> "gpt-4")
 * and sanitizes it to match the profile name pattern.
 */
export function deriveProfileNameFromModel(model: string): string {
  // Extract the model name after the last slash (provider/model format)
  const parts = model.split("/");
  const modelName = parts[parts.length - 1] || model;

  // Sanitize: replace invalid characters with dashes
  let sanitized = modelName
    .replace(/[^A-Za-z0-9._-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");

  // Ensure it starts with alphanumeric
  if (sanitized && !/^[A-Za-z0-9]/.test(sanitized)) {
    sanitized = `profile-${sanitized}`;
  }

  // Truncate to 64 characters
  if (sanitized.length > 64) {
    sanitized = sanitized.substring(0, 64).replace(/-+$/, "");
  }

  return sanitized || "default-profile";
}
