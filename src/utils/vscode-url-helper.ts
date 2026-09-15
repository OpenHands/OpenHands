/**
 * Schemes a VS Code URL is allowed to use.
 *
 * `transformVSCodeUrl` is the last gate before the URL reaches `window.open`,
 * so a URL that is not plain HTTP(S) is rejected here rather than forwarded to
 * the caller. Without this, a `javascript:` or `data:` URL from a compromised
 * or misconfigured backend would be opened as-is.
 */
const ALLOWED_PROTOCOLS = new Set(["http:", "https:"]);

/**
 * Helper function to transform VS Code URLs
 *
 * This function checks if a VS Code URL points to localhost and replaces it with
 * the current window's hostname if they don't match.
 *
 * @param vsCodeUrl The original VS Code URL from the backend
 * @returns The transformed URL, or `null` when the input is absent, unparseable,
 *   or does not use an allowed scheme
 */
export function transformVSCodeUrl(vsCodeUrl: string | null): string | null {
  if (!vsCodeUrl) return null;

  try {
    const url = new URL(vsCodeUrl);

    // Reject anything that is not HTTP(S) instead of passing it through.
    if (!ALLOWED_PROTOCOLS.has(url.protocol)) {
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
    // Not a parseable absolute URL, so it cannot be shown to be safe to open.
    return null;
  }
}
