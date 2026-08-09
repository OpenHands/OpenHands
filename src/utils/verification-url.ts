/**
 * Validate that a device-login verification URL is safe to open / link.
 * Prevents XSS via javascript: / data: / other non-https schemes.
 */
export function isValidVerificationUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:";
  } catch {
    return false;
  }
}
