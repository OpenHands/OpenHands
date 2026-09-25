/**
 * Validate that a URL is safe to open in a popup.
 * Prevents XSS via javascript: URLs or other malicious schemes.
 */
export function isValidVerificationUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:";
  } catch {
    return false;
  }
}