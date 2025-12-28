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
 * Builds the HTTP base URL for V1 API calls
 * @param conversationUrl The conversation URL containing host/port
 * @returns HTTP base URL (e.g., "http://localhost:3000")
 */
export function buildHttpBaseUrl(
  conversationUrl: string | null | undefined,
): string {
  const baseHost = extractBaseHost(conversationUrl);
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";
  return `${protocol}//${baseHost}`;
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

  // Build WebSocket URL: ws://host:port/sockets/events/{conversationId}
  // Note: Query params should be passed via the useWebSocket hook options
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";

  return `${protocol}//${baseHost}/sockets/events/${conversationId}`;
}
