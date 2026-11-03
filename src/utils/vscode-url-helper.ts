const ALLOWED_PROTOCOLS = new Set(["http:", "https:"]);

/**
 * Helper function to transform VS Code URLs
 *
 * This function checks if a VS Code URL points to localhost and replaces it with
 * the current window's hostname if they don't match. URLs that do not parse or
 * are not http(s) are rejected, since the value comes from the backend and is
 * opened in a new tab.
 *
 * @param vsCodeUrl The original VS Code URL from the backend
 * @returns The transformed URL with the correct hostname, or null if rejected
 */
export function transformVSCodeUrl(vsCodeUrl: string | null): string | null {
  if (!vsCodeUrl) return null;

  let url: URL;
  try {
    url = new URL(vsCodeUrl);
  } catch {
    return null;
  }

  if (!ALLOWED_PROTOCOLS.has(url.protocol)) return null;

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
