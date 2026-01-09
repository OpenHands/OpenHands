/**
 * Extracts the base host from conversation URL
 * @param conversationUrl The conversation URL containing host/port (e.g., "http://localhost:3000/api/conversations/123")
 * @returns Base host (e.g., "localhost:3000") or window.location.host as fallback
 */
export function extractBaseHost(
  conversationUrl: string | null | undefined,
): string {
  if (conversationUrl && !conversationUrl.startsWith("/")) {
    try {
      const url = new URL(conversationUrl);
      return url.host; // e.g., "localhost:3000"
    } catch {
      return window.location.host;
    }
  }
  return window.location.host;
}

/**
 * Extracts the path prefix from conversation URL (everything before /api/conversations)
 * This is needed for proxy deployments where agent-servers are accessed via paths like /runtime/{port}/
 * @param conversationUrl The conversation URL (e.g., "http://localhost:3000/runtime/55313/api/conversations/123")
 * @returns Path prefix without trailing slash (e.g., "/runtime/55313") or empty string
 */
export function extractPathPrefix(
  conversationUrl: string | null | undefined,
): string {
  if (conversationUrl && !conversationUrl.startsWith("/")) {
    try {
      const url = new URL(conversationUrl);
      const pathBeforeApi = url.pathname.split("/api/conversations")[0] || "";
      return pathBeforeApi.replace(/\/$/, ""); // Remove trailing slash
    } catch {
      return "";
    }
  }
  return "";
}

/**
 * Builds the HTTP base URL for V1 API calls
 * @param conversationUrl The conversation URL containing host/port
 * @returns HTTP base URL (e.g., "http://localhost:3000" or "http://localhost:3000/runtime/55313")
 */
export function buildHttpBaseUrl(
  conversationUrl: string | null | undefined,
): string {
  const baseHost = extractBaseHost(conversationUrl);
  const pathPrefix = extractPathPrefix(conversationUrl);
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";
  return `${protocol}//${baseHost}${pathPrefix}`;
}

/**
 * Builds the WebSocket URL for V1 conversations (without query params)
 * @param conversationId The conversation ID
 * @param conversationUrl The conversation URL containing host/port (e.g., "http://localhost:3000/api/conversations/123")
 * @returns WebSocket URL or null if inputs are invalid
 */
export function buildWebSocketUrl(
  conversationId: string | undefined,
  conversationUrl: string | null | undefined,
): string | null {
  if (!conversationId) {
    return null;
  }

  const baseHost = extractBaseHost(conversationUrl);
  const pathPrefix = extractPathPrefix(conversationUrl);

  // Build WebSocket URL: ws://host:port[/path-prefix]/sockets/events/{conversationId}
  // The path prefix (e.g., /runtime/55313) is needed for proxy deployments
  // Note: Query params should be passed via the useWebSocket hook options
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";

  return `${protocol}//${baseHost}${pathPrefix}/sockets/events/${conversationId}`;
}
