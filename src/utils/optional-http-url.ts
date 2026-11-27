/**
 * Optional http(s) URL check used by LLM settings save paths.
 *
 * Empty / whitespace-only values are valid (the field is optional). A
 * non-empty value must parse as `http:` or `https:` with a hostname.
 * Surrounding whitespace is ignored for the check; callers that persist
 * should trim before sending.
 */
export function isValidOptionalHttpUrl(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed) {
    return true;
  }

  try {
    const url = new URL(trimmed);
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      url.hostname.length > 0
    );
  } catch {
    return false;
  }
}
