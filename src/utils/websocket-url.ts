/**
 * Extracts the base host from conversation URL
 * @param conversationUrl The conversation URL containing host/port (e.g., "http://localhost:3000/api/conversations/123")
 * @returns Base host (e.g., "localhost:3000") or window.location.host as fallback
 */
function extractBaseHost(conversationUrl: string | null | undefined): string {
  if (conversationUrl && !conversationUrl.startsWith("/")) {
    try {
      const url = new URL(conversationUrl);
      // If the conversation URL points to localhost but we're accessing from external,
      // use the browser's hostname with the conversation URL's port
      const urlHostname = url.hostname;
      const browserHostname =
        window.location.hostname ?? window.location.host?.split(":")[0];
      if (
        browserHostname &&
        (urlHostname === "localhost" || urlHostname === "127.0.0.1") &&
        browserHostname !== "localhost" &&
        browserHostname !== "127.0.0.1"
      ) {
        return `${browserHostname}:${url.port}`;
      }
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
function extractPathPrefix(conversationUrl: string | null | undefined): string {
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
  const pageIsSecure = window.location.protocol === "https:";
  const targetIsSecure =
    getConversationUrlProtocol(conversationUrl) === "https:";
  const protocol = pageIsSecure || targetIsSecure ? "https:" : "http:";
  return `${protocol}//${baseHost}${pathPrefix}`;
}

function getConversationUrlProtocol(
  conversationUrl: string | null | undefined,
): string | null {
  if (!conversationUrl || conversationUrl.startsWith("/")) {
    return null;
  }

  try {
    return new URL(conversationUrl).protocol;
  } catch {
    return null;
  }
}

/**
 * Builds the WebSocket URL for the agent-server's bash-events endpoint.
 * The URL is derived from the same host and path prefix as the conversation
 * events socket so it works in both direct-connect and reverse-proxy deployments.
 *
 * @param conversationUrl The conversation URL containing host/port
 * @param sessionApiKey Optional session API key (appended as query param)
 * @returns WebSocket URL for the bash-events endpoint
 */
export function buildBashWebSocketUrl(
  conversationUrl: string | null | undefined,
  sessionApiKey?: string | null,
): string {
  const baseHost = extractBaseHost(conversationUrl);
  const pathPrefix = extractPathPrefix(conversationUrl);

  const pageIsSecure = window.location.protocol === "https:";
  const targetIsSecure =
    getConversationUrlProtocol(conversationUrl) === "https:";
  const protocol = pageIsSecure || targetIsSecure ? "wss:" : "ws:";

  const base = `${protocol}//${baseHost}${pathPrefix}/sockets/bash-events`;
  if (sessionApiKey) {
    return `${base}?session_api_key=${encodeURIComponent(sessionApiKey)}`;
  }
  return base;
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
  //
  // Protocol selection follows the actual HTTP access path. A page served
  // over HTTPS must use WSS, but an HTTP page that reaches a remote dev ingress
  // over plain HTTP (for example a Tailscale hostname) must use WS; forcing WSS
  // sends a TLS handshake to the HTTP-only ingress and Node reports it as a
  // malformed HTTP method.
  const pageIsSecure = window.location.protocol === "https:";
  const targetIsSecure =
    getConversationUrlProtocol(conversationUrl) === "https:";
  const protocol = pageIsSecure || targetIsSecure ? "wss:" : "ws:";

  return `${protocol}//${baseHost}${pathPrefix}/sockets/events/${conversationId}`;
}
