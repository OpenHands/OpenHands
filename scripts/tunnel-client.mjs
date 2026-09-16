/**
 * MARS port-forward tunnel client (MARSOHS-1427).
 *
 * Runs directly inside Canvas's Electron main process and plays the same
 * role `doctl agents port-forward` plays at the terminal today: given a
 * session id and a remote port inside that session's sandbox, it dials
 * MARS's port-forward WebSocket tunnel and exposes the result as an
 * ordinary local TCP listener that pipes bytes bidirectionally. Anything
 * that connects to that local port gets a plain, single-hop connection to
 * whatever is listening on the guest port.
 *
 * v1 implementation: plain JS, no subprocess. Node's `net` module provides
 * the local TCP listener (the same tool `doctl`'s Go implementation and an
 * earlier Python port both used their own language's equivalent of); the
 * `ws` package provides the WebSocket client. Node's *built-in* `WebSocket`
 * global is not usable here — it deliberately mirrors the browser spec,
 * which does not allow attaching custom headers to a WebSocket handshake,
 * and this tunnel needs to attach `Authorization: Bearer <token>`. `ws`
 * isn't spec-constrained that way and supports it directly.
 *
 * This replaced an earlier prototype that ran the same logic as a spawned
 * Python subprocess (`uv run tools/mars_tunnel.py`). That version worked,
 * but running the tunnel logic in-process here removes an entire layer
 * that existed only to bridge two different languages: no subprocess to
 * spawn and supervise, no hand-rolled stdout readiness protocol, no
 * dependency on `uv` resolving a package at startup. See git history for
 * the Python version if it's ever useful for comparison.
 *
 * Multi-session note: this module starts one local listener per call, for
 * one (session, remote port) pair. Building a `session_id -> tunnel`
 * registry on top of this (port allocation, reuse, targeted teardown,
 * reconnect detection) is a separate, follow-on ticket — out of scope here.
 */

import { createServer } from "node:net";
import WebSocket from "ws";

const REJECTION_BODY_MAX = 300;

function buildTunnelWsUrl(apiUrl, sessionId, remotePort) {
  const url = new URL(apiUrl);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Unsupported API URL scheme: ${url.protocol}`);
  }
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname =
    url.pathname.replace(/\/$/, "") +
    `/v2/agents/sessions/${sessionId}/port-forward/${remotePort}`;
  return url.toString();
}

/**
 * Dial the MARS port-forward tunnel for one session/port pair and resolve
 * once the local listener is bound and accepting connections.
 *
 * @param {object} options
 * @param {string} options.sessionId MARS session id
 * @param {number} options.remotePort Port inside the guest sandbox
 * @param {string} options.accessToken Bearer token for the tunnel
 * @param {string} [options.apiUrl] harness-api base URL (http(s)://...); translated to ws(s)://
 * @param {number} [options.localPort] Local port to listen on (0 lets the OS pick one)
 * @param {string} [options.address] Local bind address
 * @param {(message: string) => void} [options.log]
 * @param {typeof WebSocket} [options.WebSocketImpl] Override for tests
 * @returns {Promise<{sessionId: string, remotePort: number, localPort: number, stop: () => void}>}
 */
export async function startPortForwardTunnel({
  sessionId,
  remotePort,
  accessToken,
  apiUrl = "https://api.digitalocean.com/",
  localPort = 0,
  address = "127.0.0.1",
  log = (message) => console.error(`[mars-tunnel] ${message}`),
  WebSocketImpl = WebSocket,
} = {}) {
  if (!sessionId) {
    throw new Error("sessionId is required");
  }
  if (!Number.isInteger(remotePort) || remotePort <= 0 || remotePort > 65535) {
    throw new Error(`Invalid remote port: ${remotePort}`);
  }
  if (!accessToken) {
    throw new Error("accessToken is required");
  }

  const wsUrl = buildTunnelWsUrl(apiUrl, sessionId, remotePort);
  const headers = { Authorization: `Bearer ${accessToken}` };
  const activeSockets = new Set();

  function bridgeConnection(localSocket) {
    activeSockets.add(localSocket);
    localSocket.once("close", () => activeSockets.delete(localSocket));

    const ws = new WebSocketImpl(wsUrl, { headers });
    let rejected = false;

    // Fires when the server answers the handshake with a non-101 status
    // (e.g. a rejected/invalid token) — doctl's equivalent is
    // serverRejection() reading the same kind of response.
    ws.on("unexpected-response", (_req, res) => {
      rejected = true;
      let body = "";
      res.on("data", (chunk) => {
        body += chunk.toString();
      });
      res.on("end", () => {
        const message = body.trim().slice(0, REJECTION_BODY_MAX);
        log(
          `server rejected tunnel (${res.statusCode} ${res.statusMessage}): ${message}`,
        );
        localSocket.end();
      });
    });

    ws.on("open", () => {
      localSocket.on("data", (chunk) => ws.send(chunk));
      ws.on("message", (data) => localSocket.write(data));
    });

    ws.on("close", () => localSocket.end());
    ws.on("error", (err) => {
      // 'unexpected-response' already logged the rejection and its message;
      // 'ws' also emits a generic error for the same failure, which would
      // otherwise be a duplicate, less useful log line.
      if (!rejected) {
        log(`tunnel dial failed: ${err.message}`);
      }
    });

    localSocket.on("close", () => {
      if (ws.readyState === WebSocketImpl.OPEN || ws.readyState === WebSocketImpl.CONNECTING) {
        ws.close();
      }
    });
    localSocket.on("error", () => ws.close());
  }

  const server = createServer(bridgeConnection);

  return new Promise((resolve, reject) => {
    function onListenError(err) {
      reject(err);
    }
    server.once("error", onListenError);
    server.listen(localPort, address, () => {
      server.off("error", onListenError);
      // Post-startup errors (e.g. a client resetting a connection mid-accept)
      // still need a handler or Node treats them as uncaught.
      server.on("error", (err) => log(`listener error: ${err.message}`));
      const boundPort = server.address().port;
      log(
        `forwarding ${address}:${boundPort} -> port ${remotePort} in session ${sessionId}`,
      );
      resolve({
        sessionId,
        remotePort,
        localPort: boundPort,
        stop() {
          for (const socket of activeSockets) {
            socket.destroy();
          }
          server.close();
        },
      });
    });
  });
}
