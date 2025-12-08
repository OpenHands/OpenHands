import { ws } from "msw";
import { setupServer } from "msw/node";

/**
 * Creates a WebSocket link for MSW testing
 * @param url - WebSocket URL to mock (default: "ws://localhost/events/socket")
 * @returns MSW WebSocket link
 */
export const createWebSocketLink = (url = "ws://localhost/events/socket") =>
  ws.link(url);

/**
 * Creates and configures an MSW server for WebSocket testing
 * @param wsLink - WebSocket link to use for the server
 * @returns Configured MSW server
 */
export const createWebSocketMockServer = (wsLink: ReturnType<typeof ws.link>) =>
  setupServer(
    wsLink.addEventListener("connection", ({ server }) => {
      server.connect();
    }),
  );

/**
 * Creates a complete WebSocket testing setup with server and link
 * @param url - WebSocket URL to mock (default: "ws://localhost/events/socket")
 * @returns Object containing the WebSocket link and configured server
 */
export const createWebSocketTestSetup = (
  url = "ws://localhost/events/socket",
) => {
  const wsLink = createWebSocketLink(url);
  const server = createWebSocketMockServer(wsLink);

  return { wsLink, server };
};

/**
 * Standard WebSocket test setup for conversation WebSocket handler tests
 * Updated to use the V1 WebSocket URL pattern: /sockets/events/{conversationId}
 * Uses a wildcard pattern to match any conversation ID
 */
export const conversationWebSocketTestSetup = () =>
  createWebSocketTestSetup("ws://localhost:3000/sockets/events/*");
