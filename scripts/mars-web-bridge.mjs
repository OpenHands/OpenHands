/**
 * MARS for the web build: the Agent Canvas server stands in for Electron's
 * main process.
 *
 * In the desktop app the renderer reaches a DigitalOcean session through
 * `window.marsBridge` (electron/mars-preload.cjs → scripts/mars-tunnel-bridge.mjs),
 * and Electron stamps the DO token onto the renderer's own requests to the
 * session's public ingress URL. A browser can do neither: harness-api sends
 * no CORS headers, a browser WebSocket cannot carry an Authorization header,
 * and the token must never reach the page. So the server that already fronts
 * the web build (scripts/ingress.mjs, scripts/static-server.mjs) hosts the
 * same bridge and exposes it same-origin:
 *
 *   POST /mars/rpc/<method>        — the MarsBridge surface over JSON; the
 *                                    renderer's fetch-backed bridge
 *                                    (src/api/mars/mars-web-bridge.ts) calls
 *                                    this exactly as the preload calls IPC.
 *   ANY  /mars/sessions/<id>/...   — reverse proxy (HTTP + WebSocket) to that
 *                                    session's connected host (its ingress
 *                                    URL, or the tunnel's loopback listener),
 *                                    with `Authorization: Bearer` added
 *                                    server-side and the upstream WebSocket
 *                                    pinged every 30 s: the ingress route has
 *                                    a 20 min idle timeout and drops silently.
 *   GET  /mars/health              — lets the renderer detect this server.
 *
 * `openTunnel` therefore answers with `host` = `<origin>/mars/sessions/<id>`,
 * a path-prefixed backend the renderer already supports (see
 * src/utils/websocket-url.ts `extractPathPrefix`). The agent-server writes
 * its own origin into `conversation_url`, which the renderer uses to open
 * the event WebSocket, so proxied JSON has those URLs rewritten to the prefix.
 *
 * Trust model: same as the rest of the local stack — the server binds
 * loopback by default and anything that can reach it can drive it. Cross-site
 * pages cannot: JSON POSTs preflight and this server sends no CORS headers,
 * and a mismatched `Origin` is refused outright. Credentials are in-memory
 * only (no OS keychain here); `MARS_TOKEN` seeds one at startup.
 */

import { mkdtempSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { tmpdir } from "node:os";
import { join } from "node:path";
import WebSocket, { WebSocketServer } from "ws";

import { createMarsTunnelBridge } from "./mars-tunnel-bridge.mjs";

export const MARS_WEB_PREFIX = "/mars";
const HEALTH_PATH = `${MARS_WEB_PREFIX}/health`;
const RPC_PREFIX = `${MARS_WEB_PREFIX}/rpc/`;
const SESSIONS_PREFIX = `${MARS_WEB_PREFIX}/sessions/`;

/** Ingress routes idle out at 20 min; keep the upstream socket warm. */
export const UPSTREAM_PING_INTERVAL_MS = 30_000;
const RPC_BODY_LIMIT_BYTES = 64 * 1024;
const JSON_REWRITE_LIMIT_BYTES = 8 * 1024 * 1024;

/** Bridge methods a browser may call. Mirrors MARS_TUNNEL_IPC. */
const RPC_METHODS = new Set([
  "getAuthState",
  "savePat",
  "setActiveConnection",
  "signOut",
  "listSessions",
  "listAgentConfigs",
  "listConfigSessions",
  "createOpenHandsAgent",
  "createSession",
  "pauseSession",
  "resumeSession",
  "openTunnel",
  "closeTunnel",
  "getTunnel",
]);

/** Headers that describe one hop and must not be forwarded. */
const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "proxy-connection",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
  "origin",
  "referer",
  "accept-encoding",
]);

export function isMarsWebEnabled(env = process.env) {
  return env.MARS_WEB !== "0";
}

/** `http://host[:port]` of the request as the browser sees it. */
function requestOrigin(req) {
  const proto =
    req.headers["x-forwarded-proto"]?.split(",")[0].trim() || "http";
  return `${proto}://${req.headers.host}`;
}

function sessionProxyBase(req, sessionId) {
  return `${requestOrigin(req)}${SESSIONS_PREFIX}${encodeURIComponent(sessionId)}`;
}

/**
 * A page on another origin cannot be allowed to drive the bridge or read
 * through the proxy. Same-origin requests either carry a matching Origin or
 * none at all (plain navigations, same-origin GETs).
 */
function isCrossOrigin(req) {
  const origin = req.headers.origin;
  if (!origin) return false;
  try {
    return new URL(origin).host !== req.headers.host;
  } catch {
    return true;
  }
}

function writeJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(payload),
    "Cache-Control": "no-store",
  });
  res.end(payload);
}

function readJsonBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error("request body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (chunks.length === 0) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch (error) {
        reject(new Error("request body is not valid JSON"));
      }
    });
    req.on("error", reject);
  });
}

const CONVERSATION_URL_RE =
  /^https?:\/\/[^/?#]+(?:\/[^?#]*?)?(?=\/api\/conversations(?:[/?#]|$))/;

/**
 * Point every agent-server URL in a response at the proxy prefix. The
 * agent-server builds `conversation_url` from its own view of the request,
 * whatever host that is, and the renderer opens the event WebSocket at that
 * host — so without this the page would try to reach the ingress hostname
 * (or the guest) directly. Matches any origin + prefix in front of
 * `/api/conversations`, so it does not depend on what the upstream thinks
 * its hostname is.
 */
export function rewriteAgentServerUrls(value, proxyBase) {
  if (typeof value === "string") {
    return value.replace(CONVERSATION_URL_RE, proxyBase);
  }
  if (Array.isArray(value)) {
    return value.map((item) => rewriteAgentServerUrls(item, proxyBase));
  }
  if (value && typeof value === "object") {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      out[key] = rewriteAgentServerUrls(item, proxyBase);
    }
    return out;
  }
  return value;
}

function parseSessionPath(url) {
  if (!url.startsWith(SESSIONS_PREFIX)) return null;
  const rest = url.slice(SESSIONS_PREFIX.length);
  const slash = rest.indexOf("/");
  const queryStart = rest.search(/[?#]/);
  const idEnd =
    slash === -1 ? (queryStart === -1 ? rest.length : queryStart) : slash;
  const sessionId = decodeURIComponent(rest.slice(0, idEnd));
  if (!sessionId) return null;
  const tail = rest.slice(idEnd);
  return { sessionId, path: tail.startsWith("/") ? tail : `/${tail}` };
}

function forwardHeaders(req, upstream, authorization) {
  const headers = {};
  for (const [name, value] of Object.entries(req.headers)) {
    if (!HOP_BY_HOP_HEADERS.has(name) && value !== undefined) {
      headers[name] = value;
    }
  }
  headers.host = upstream.host;
  // Responses are read as UTF-8 for the URL rewrite; a compressed body
  // would have to be inflated first.
  headers["accept-encoding"] = "identity";
  if (authorization) headers.authorization = authorization;
  return headers;
}

/**
 * @param {object} [options]
 * @param {ReturnType<typeof createMarsTunnelBridge>} [options.bridge] Override for tests
 * @param {Record<string, string | undefined>} [options.env]
 * @param {number} [options.pingIntervalMs]
 * @param {(message: string, error?: unknown) => void} [options.log]
 */
export function createMarsWebBridge({
  bridge: bridgeOverride,
  env = process.env,
  pingIntervalMs = UPSTREAM_PING_INTERVAL_MS,
  log = (message, error) => console.warn(`[mars-web] ${message}`, error ?? ""),
} = {}) {
  const bridge =
    bridgeOverride ??
    createMarsTunnelBridge({
      // Fresh per process: tokens are in-memory anyway (no keychain here),
      // and a surviving connection record with no token would only confuse.
      userDataPath: mkdtempSync(join(tmpdir(), "agent-canvas-mars-")),
      env,
    });

  /** Connected sessions: id → the real upstream base (ingress URL or tunnel). */
  const upstreams = new Map();
  /** Live WebSocket pairs, so dispose() can close them. */
  const sockets = new Set();

  const wss = new WebSocketServer({ noServer: true, perMessageDeflate: false });

  if (env.MARS_TOKEN) {
    bridge
      .savePat({ token: env.MARS_TOKEN, label: "MARS_TOKEN" })
      .catch((error) => log("MARS_TOKEN was not accepted", error));
  }

  function rewriteStatus(req, status) {
    if (!status?.host) return status;
    upstreams.set(status.sessionId, status.host);
    return { ...status, host: sessionProxyBase(req, status.sessionId) };
  }

  /** The bridge surface as the browser sees it. */
  const rpc = {
    async getAuthState() {
      // The implicit-grant flow needs a loopback redirect in a trusted
      // process; the web build takes personal access tokens only.
      return { ...bridge.getAuthState(), canUseOAuth: false };
    },
    savePat: (payload) => bridge.savePat(payload),
    setActiveConnection: (id) => bridge.setActiveConnection(id),
    signOut: (id) => bridge.signOut(id),
    listSessions: (options) => bridge.listSessions(options),
    listAgentConfigs: (options) => bridge.listAgentConfigs(options),
    listConfigSessions: (configId, options) =>
      bridge.listConfigSessions(configId, options),
    createOpenHandsAgent: (payload) => bridge.createOpenHandsAgent(payload),
    createSession: (configId, name) => bridge.createSession(configId, name),
    pauseSession: (sessionId) => bridge.pauseSession(sessionId),
    resumeSession: (sessionId) => bridge.resumeSession(sessionId),
    async openTunnel(req, params) {
      return rewriteStatus(req, await bridge.openTunnel(params));
    },
    async closeTunnel(_req, sessionId) {
      upstreams.delete(sessionId);
      return bridge.closeTunnel(sessionId);
    },
    getTunnel(req, sessionId) {
      return rewriteStatus(req, bridge.getTunnel(sessionId));
    },
  };
  const REQUEST_AWARE = new Set(["openTunnel", "closeTunnel", "getTunnel"]);

  async function handleRpc(req, res, method) {
    if (req.method !== "POST") {
      writeJson(res, 405, { error: { message: "use POST" } });
      return;
    }
    if (!RPC_METHODS.has(method)) {
      writeJson(res, 404, { error: { message: `unknown method ${method}` } });
      return;
    }
    let args;
    try {
      const body = await readJsonBody(req, RPC_BODY_LIMIT_BYTES);
      args = Array.isArray(body.args) ? body.args : [];
    } catch (error) {
      writeJson(res, 400, { error: { message: error.message } });
      return;
    }
    try {
      const result = REQUEST_AWARE.has(method)
        ? await rpc[method](req, ...args)
        : await rpc[method](...args);
      writeJson(res, 200, { result: result ?? null });
    } catch (error) {
      const status =
        typeof error?.status === "number" && error.status >= 400
          ? error.status
          : 500;
      writeJson(res, status, {
        error: { message: error?.message ?? String(error), status },
      });
    }
  }

  function resolveUpstream(sessionId) {
    const base = upstreams.get(sessionId);
    if (!base) return null;
    const upstream = new URL(base);
    return {
      upstream,
      authorization: bridge.ingressAuthorizationHeader(base),
    };
  }

  function proxyHttp(req, res, { sessionId, path }) {
    const target = resolveUpstream(sessionId);
    if (!target) {
      writeJson(res, 409, {
        error: { message: `session ${sessionId} is not connected` },
      });
      return;
    }
    const { upstream, authorization } = target;
    const proxyBase = sessionProxyBase(req, sessionId);
    const isTls = upstream.protocol === "https:";
    const upstreamReq = (isTls ? httpsRequest : httpRequest)(
      {
        protocol: upstream.protocol,
        hostname: upstream.hostname,
        port: upstream.port || (isTls ? 443 : 80),
        method: req.method,
        path: `${upstream.pathname.replace(/\/$/, "")}${path}`,
        headers: forwardHeaders(req, upstream, authorization),
      },
      (upstreamRes) => {
        const headers = { ...upstreamRes.headers };
        delete headers["content-encoding"];
        const isJson = /\bapplication\/json\b/i.test(
          upstreamRes.headers["content-type"] ?? "",
        );
        if (!isJson) {
          res.writeHead(upstreamRes.statusCode ?? 502, headers);
          upstreamRes.pipe(res);
          return;
        }
        const chunks = [];
        let size = 0;
        upstreamRes.on("data", (chunk) => {
          size += chunk.length;
          if (size <= JSON_REWRITE_LIMIT_BYTES) chunks.push(chunk);
        });
        upstreamRes.on("end", () => {
          let body = Buffer.concat(chunks);
          if (size <= JSON_REWRITE_LIMIT_BYTES) {
            try {
              body = Buffer.from(
                JSON.stringify(
                  rewriteAgentServerUrls(
                    JSON.parse(body.toString("utf8")),
                    proxyBase,
                  ),
                ),
              );
            } catch {
              // Not JSON after all; pass it through as received.
            }
          }
          delete headers["transfer-encoding"];
          headers["content-length"] = String(body.length);
          res.writeHead(upstreamRes.statusCode ?? 502, headers);
          res.end(body);
        });
        upstreamRes.on("error", () => res.destroy());
      },
    );
    upstreamReq.on("error", (error) => {
      if (!res.headersSent) {
        writeJson(res, 502, {
          error: { message: `upstream request failed: ${error.message}` },
        });
      } else {
        res.destroy();
      }
    });
    req.pipe(upstreamReq);
  }

  function proxyWebSocket(req, socket, head, { sessionId, path }) {
    const target = resolveUpstream(sessionId);
    if (!target) {
      socket.end("HTTP/1.1 409 Conflict\r\n\r\n");
      return;
    }
    const { upstream, authorization } = target;
    const wsUrl = new URL(
      `${upstream.pathname.replace(/\/$/, "")}${path}`,
      upstream,
    );
    wsUrl.protocol = upstream.protocol === "https:" ? "wss:" : "ws:";
    const protocols = req.headers["sec-websocket-protocol"]
      ?.split(",")
      .map((p) => p.trim())
      .filter(Boolean);
    const headers = {};
    if (authorization) headers.authorization = authorization;
    const upstreamSocket = new WebSocket(wsUrl, protocols, {
      headers,
      perMessageDeflate: false,
    });

    let client = null;
    const pending = [];
    const pingTimer = setInterval(() => {
      if (upstreamSocket.readyState === WebSocket.OPEN) upstreamSocket.ping();
    }, pingIntervalMs);
    const teardown = () => {
      clearInterval(pingTimer);
      sockets.delete(pair);
    };
    const pair = {
      upstreamSocket,
      get client() {
        return client;
      },
    };
    sockets.add(pair);

    const closeClient = (code, reason) => {
      if (!client) {
        socket.destroy();
        return;
      }
      // 1005/1006 are reserved for the receiver and cannot be sent.
      if (code === 1005 || code === 1006 || code === undefined) client.close();
      else client.close(code, reason);
    };

    upstreamSocket.on("unexpected-response", (_req, res) => {
      teardown();
      socket.end(
        `HTTP/1.1 ${res.statusCode} ${res.statusMessage ?? ""}\r\n\r\n`,
      );
    });
    upstreamSocket.on("error", (error) => {
      teardown();
      if (!client) {
        socket.destroy();
      } else {
        log(`upstream websocket error for ${sessionId}`, error);
        closeClient(1011, "upstream error");
      }
    });
    upstreamSocket.on("close", (code, reason) => {
      teardown();
      closeClient(code, reason);
    });
    upstreamSocket.on("message", (data, isBinary) => {
      if (client && client.readyState === WebSocket.OPEN) {
        client.send(data, { binary: isBinary });
      } else {
        pending.push([data, isBinary]);
      }
    });
    upstreamSocket.on("open", () => {
      wss.handleUpgrade(req, socket, head, (ws) => {
        client = ws;
        for (const [data, isBinary] of pending.splice(0)) {
          ws.send(data, { binary: isBinary });
        }
        ws.on("message", (data, isBinary) => {
          if (upstreamSocket.readyState === WebSocket.OPEN) {
            upstreamSocket.send(data, { binary: isBinary });
          }
        });
        ws.on("close", (code, reason) => {
          teardown();
          if (upstreamSocket.readyState === WebSocket.OPEN) {
            if (code === 1005 || code === 1006) upstreamSocket.close();
            else upstreamSocket.close(code, reason);
          }
        });
        ws.on("error", () => {
          teardown();
          upstreamSocket.terminate();
        });
      });
    });
  }

  return {
    /** The bridge in use, for the caller's own dispose() or tests. */
    bridge,

    /** @returns {boolean} true when the request was a MARS request and handled */
    handleHttp(req, res) {
      const url = req.url ?? "/";
      if (!url.startsWith(`${MARS_WEB_PREFIX}/`)) return false;
      if (isCrossOrigin(req)) {
        writeJson(res, 403, {
          error: { message: "cross-origin request refused" },
        });
        return true;
      }
      const pathOnly = url.split(/[?#]/)[0];
      if (pathOnly === HEALTH_PATH) {
        writeJson(res, 200, { ok: true });
        return true;
      }
      if (pathOnly.startsWith(RPC_PREFIX)) {
        void handleRpc(req, res, pathOnly.slice(RPC_PREFIX.length));
        return true;
      }
      const session = parseSessionPath(url);
      if (session) {
        proxyHttp(req, res, session);
        return true;
      }
      writeJson(res, 404, { error: { message: "not found" } });
      return true;
    },

    /** @returns {boolean} true when the upgrade was a MARS request and handled */
    handleUpgrade(req, socket, head) {
      const url = req.url ?? "/";
      if (!url.startsWith(`${MARS_WEB_PREFIX}/`)) return false;
      if (isCrossOrigin(req)) {
        socket.end("HTTP/1.1 403 Forbidden\r\n\r\n");
        return true;
      }
      const session = parseSessionPath(url);
      if (!session) {
        socket.end("HTTP/1.1 404 Not Found\r\n\r\n");
        return true;
      }
      proxyWebSocket(req, socket, head, session);
      return true;
    },

    async dispose() {
      for (const pair of sockets) {
        pair.upstreamSocket.terminate();
        pair.client?.terminate();
      }
      sockets.clear();
      upstreams.clear();
      await bridge.dispose();
    },
  };
}
