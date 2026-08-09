/**
 * Helper function to transform VS Code URLs
 *
 * This function checks if a VS Code URL points to localhost and replaces it with
 * the current window's hostname if they don't match.
 *
 * Only http:, https:, and vscode: schemes are returned. Invalid or non-allowlisted
 * inputs (e.g. javascript:) yield null so callers that window.open do not navigate
 * to attacker-controlled schemes.
 *
 * @param vsCodeUrl The original VS Code URL from the backend
 * @returns The transformed URL with the correct hostname, or null if unsafe/invalid
 */
const ALLOWED_VSCODE_URL_PROTOCOLS = new Set(["http:", "https:", "vscode:"]);

export function transformVSCodeUrl(vsCodeUrl: string | null): string | null {
  if (!vsCodeUrl) return null;

  try {
    const url = new URL(vsCodeUrl);

    if (!ALLOWED_VSCODE_URL_PROTOCOLS.has(url.protocol)) {
      return null;
    }

    // Check if the URL points to localhost
    if (
      url.hostname === "localhost" &&
      window.location.hostname !== "localhost"
    ) {
      // Replace localhost with the current hostname
      url.hostname = window.location.hostname;
      return url.toString();
    }

    return vsCodeUrl;
  } catch {
    // Invalid URLs must not be passed through to window.open
    return null;
  }
}
