// Browsers never time out a socket stuck in CONNECTING on their own, and
// Chrome serializes WebSocket handshakes per host — a hung handshake also
// blocks every other socket to the same origin until it settles. Close it
// after this long so a retry (and any other queued handshake) can proceed.
const HANDSHAKE_TIMEOUT_MS = 10_000;

/**
 * Watches a freshly constructed socket's handshake and closes the socket if it
 * is still in CONNECTING when the timeout elapses. close() fires the socket's
 * own error/close events (code 1006), so the abort flows through the caller's
 * normal close handling rather than a separate failure path.
 *
 * Returns a cancel function. The caller must invoke it once the handshake
 * settles — in its open and close handlers, and in any cleanup that detaches
 * those handlers — so the watchdog never outlives the socket it watches.
 */
export function startHandshakeWatchdog(
  ws: Pick<WebSocket, "readyState" | "close">,
): () => void {
  const timeout = setTimeout(() => {
    if (ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  }, HANDSHAKE_TIMEOUT_MS);
  return () => clearTimeout(timeout);
}
