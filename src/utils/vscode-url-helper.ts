/**
 * Schemes a VS Code URL may use. The value is backend-controlled and is handed
 * straight to `window.open`, so anything outside this set — `javascript:`,
 * `data:`, an OS-registered handler — is dropped rather than navigated to.
 */
const ALLOWED_VSCODE_URL_PROTOCOLS = new Set(["http:", "https:"]);

/**
 * Helper function to transform VS Code URLs
 *
 * This function checks if a VS Code URL points to localhost and replaces it with
 * the current window's hostname if they don't match.
 *
 * @param vsCodeUrl The original VS Code URL from the backend
 * @returns The transformed URL with the correct hostname, or `null` when the
 *   input is absent, unparsable, or uses a scheme outside the allowlist
 */
export function transformVSCodeUrl(vsCodeUrl: string | null): string | null {
  if (!vsCodeUrl) return null;

  let url: URL;
  try {
    url = new URL(vsCodeUrl);
  } catch {
    // A string that doesn't parse can't be scheme-checked, so it never reaches
    // the caller's `window.open`.
    return null;
  }

  if (!ALLOWED_VSCODE_URL_PROTOCOLS.has(url.protocol)) return null;

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
}
