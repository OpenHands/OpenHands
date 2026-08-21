/**
 * Helper function to transform VS Code URLs
 *
 * This function checks if a VS Code URL points to localhost and replaces it with
 * the current window's hostname, port, and protocol if they don't match.
 *
 * @param vsCodeUrl The original VS Code URL from the backend
 * @returns The transformed URL with the correct hostname, port, and protocol
 */
export function transformVSCodeUrl(vsCodeUrl: string | null): string | null {
  if (!vsCodeUrl) return null;

  try {
    const url = new URL(vsCodeUrl);

    // Check if the URL points to localhost
    if (
      url.hostname === "localhost" &&
      window.location.hostname !== "localhost"
    ) {
      // Replace localhost with the host exposed to the browser.
      url.hostname = window.location.hostname;
      url.port = window.location.port;
      url.protocol = window.location.protocol;
      return url.toString();
    }

    return vsCodeUrl;
  } catch {
    // Silently handle the error and return the original URL
    return vsCodeUrl;
  }
}
