/**
 * MARS for the web build: the Agent Canvas server stands in for Electron's
 * main process.
 *
 * In the desktop app the renderer reaches a DigitalOcean session through
 * `window.marsBridge` (electron/preload-main.cjs → scripts/mars-tunnel-bridge.mjs),
 * which holds a port-forward tunnel to the session in the Electron main
 * process. A browser can do neither: harness-api sends no CORS headers, the
 * tunnel needs a Node process, and the token must never reach the page. So
 * the server that already fronts the web build (scripts/ingress.mjs,
 * scripts/static-server.mjs) hosts the same bridge — holding the tunnel
 * itself — and exposes it same-origin:
 *
 *   POST /mars/rpc/<method>        — the MarsBridge surface over JSON; the
 *                                    renderer's fetch-backed bridge
 *                                    (src/api/mars/mars-web-bridge.ts) calls
 *                                    this exactly as the preload calls IPC.
 *   ANY  /mars/sessions/<id>/...   — reverse proxy (HTTP + WebSocket) to that
 *                                    session's tunnel listener on loopback,
 *                                    with the upstream WebSocket pinged
 *                                    every 30 s so an idle stream is not
 *                                    reaped along the way.
 *   GET  /mars/health              — lets the renderer detect this server.
 *
 * `openTunnel` therefore answers with `host` = `<origin>/mars/sessions/<id>`,
 * a path-prefixed backend the renderer already supports (see
 * src/utils/websocket-url.ts `extractPathPrefix`). The agent-server writes
 * its own origin into `conversation_url`, which the renderer uses to open
 * the event WebSocket, so proxied JSON has those URLs rewritten to the prefix.
 *
 * Trust model. These routes hold the DigitalOcean token and can destroy
 * sessions, so they are off unless `MARS_WEB=1`, and when on they want the
 * server's session key on every request: the renderer sends it as
 * `X-Session-API-Key` on `/mars/rpc/*` (the same key it already sends the
 * agent-server), and a request that presents it gets an `HttpOnly;
 * SameSite=Strict; Path=/mars` cookie back, which the browser then attaches
 * on its own to the proxied REST calls and to the WebSocket upgrade — the
 * one request a browser cannot put a header on. The key is `MARS_WEB_KEY`,
 * else the server's `--session-api-key`; with neither, the bridge mounts
 * only on a loopback bind (dev), where a `Host` check keeps a DNS-rebound
 * page from reaching it (cookies are host-scoped, so with a key in play
 * rebinding cannot carry one). Cross-origin requests are refused outright;
 * JSON POSTs preflight and this server sends no CORS headers. Credentials
 * are in-memory only (no OS keychain here); `MARS_TOKEN` seeds one at
 * startup.
 */

import { randomBytes, timingSafeEqual } from "node:crypto";
import { mkdtempSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { tmpdir } from "node:os";
import { join } from "node:path";
import WebSocket, { WebSocketServer } from "ws";

import { DEFAULT_BIND_HOST, isLoopbackBind } from "./bind-host.mjs";
import { createMarsTunnelBridge } from "./mars-tunnel-bridge.mjs";

export const MARS_WEB_PREFIX = "/mars";
const HEALTH_PATH = `${MARS_WEB_PREFIX}/health`;
const RPC_PREFIX = `${MARS_WEB_PREFIX}/rpc/`;
const SESSIONS_PREFIX = `${MARS_WEB_PREFIX}/sessions/`;

/** Keep an idle upstream socket warm so intermediaries do not reap it. */
export const UPSTREAM_PING_INTERVAL_MS = 30_000;
const RPC_BODY_LIMIT_BYTES = 64 * 1024;
/** Past this a JSON body streams through untouched rather than being rewritten. */
const JSON_REWRITE_LIMIT_BYTES = 8 * 1024 * 1024;
/** The session key on HTTP requests; same header the agent-server takes. */
const AUTH_HEADER = "x-session-api-key";
/** Issued to a request that presented the key; carries auth to WebSocket upgrades. */
const AUTH_COOKIE = "mars_web_auth";
/** Response fields whose agent-server URL the renderer follows. */
const URL_FIELDS = new Set(["conversation_url"]);

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
  "destroySession",
  "deleteAgentConfig",
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
  // Belong to the canvas origin, never to the session host.
  "cookie",
  AUTH_HEADER,
]);

/** Off unless asked for: these routes hold the DO token and can destroy sessions. */
export function isMarsWebEnabled(env = process.env) {
  return env.MARS_WEB === "1";
}

/**
 * The key `/mars/*` requires, or null when none is configured. Explicit
 * `MARS_WEB_KEY` wins so a deployment can decouple it from the agent-server
 * key; otherwise the server's own `--session-api-key` is reused, which is
 * what the renderer already holds.
 */
export function resolveMarsWebKey({
  env = process.env,
  sessionApiKey = null,
} = {}) {
  const explicit = env.MARS_WEB_KEY?.trim();
  if (explicit) return explicit;
  const own = typeof sessionApiKey === "string" ? sessionApiKey.trim() : "";
  return own || null;
}

/**
 * Mount the bridge for a server, or return null when it must not be: not
 * opted in, or no key on a non-loopback bind (the routes would be open to
 * the LAN with the DO token behind them).
 */
export function mountMarsWebBridge({
  env = process.env,
  host = DEFAULT_BIND_HOST,
  sessionApiKey = null,
  log = (message) => console.warn(`[mars-web] ${message}`),
} = {}) {
  if (!isMarsWebEnabled(env)) return null;
  const key = resolveMarsWebKey({ env, sessionApiKey });
  if (!key && !isLoopbackBind(host)) {
    log(
      `MARS_WEB=1 but no key and bind host ${host} is not loopback; refusing to mount /mars/* (set MARS_WEB_KEY or --session-api-key).`,
    );
    return null;
  }
  if (!key) log("no key configured; /mars/* is open to local callers only");
  return createMarsWebBridge({ env, host, key });
}

function safeEqual(a, b) {
  const x = Buffer.from(a);
  const y = Buffer.from(b);
  return x.length === y.length && timingSafeEqual(x, y);
}

function readCookie(req, name) {
  const header = req.headers.cookie;
  if (!header) return null;
  for (const part of header.split(";")) {
    const eq = part.indexOf("=");
    if (eq === -1) continue;
    if (part.slice(0, eq).trim() === name) return part.slice(eq + 1).trim();
  }
  return null;
}

/** Hostname of the request's `Host` header, without port or brackets. */
function requestHostname(req) {
  const host = req.headers.host ?? "";
  try {
    return new URL(`http://${host}`).hostname.replace(/^\[|\]$/g, "");
  } catch {
    return "";
  }
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

/**
 * On a loopback bind the only legitimate `Host` values are loopback names;
 * a DNS-rebound page arrives with its own hostname in both `Host` and
 * `Origin`, so the origin check alone would pass it.
 */
function isForeignHost(req, bindHost) {
  if (!isLoopbackBind(bindHost)) return false;
  const hostname = requestHostname(req);
  return !(isLoopbackBind(hostname) || hostname.endsWith(".localhost"));
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
 * Point the agent-server URL fields of a response at the proxy prefix. The
 * agent-server builds `conversation_url` from its own view of the request,
 * whatever host that is, and the renderer opens the event WebSocket at that
 * host — so without this the page would try to reach the guest directly. Only the fields in URL_FIELDS are touched, so a
 * URL that merely appears inside a message or event is left alone. Matches
 * any origin + prefix in front of `/api/conversations`, so it does not
 * depend on what the upstream thinks its hostname is.
 */
export function rewriteAgentServerUrls(value, proxyBase, field = null) {
  if (typeof value === "string") {
    return field !== null && URL_FIELDS.has(field)
      ? value.replace(CONVERSATION_URL_RE, proxyBase)
      : value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => rewriteAgentServerUrls(item, proxyBase, field));
  }
  if (value && typeof value === "object") {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      out[key] = rewriteAgentServerUrls(item, proxyBase, key);
    }
    return out;
  }
  return value;
}

/** Thrown by parseSessionPath for a path that cannot be decoded. */
class BadSessionPath extends Error {}

function parseSessionPath(url) {
  if (!url.startsWith(SESSIONS_PREFIX)) return null;
  const rest = url.slice(SESSIONS_PREFIX.length);
  const slash = rest.indexOf("/");
  const queryStart = rest.search(/[?#]/);
  const idEnd =
    slash === -1 ? (queryStart === -1 ? rest.length : queryStart) : slash;
  let sessionId;
  try {
    sessionId = decodeURIComponent(rest.slice(0, idEnd));
  } catch {
    // A malformed escape must be a 400, not an exception out of the
    // request handler that takes the whole server down.
    throw new BadSessionPath("malformed session id in path");
  }
  if (!sessionId) return null;
  const tail = rest.slice(idEnd);
  return { sessionId, path: tail.startsWith("/") ? tail : `/${tail}` };
}

function forwardHeaders(req, upstream) {
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
  return headers;
}

/**
 * @param {object} [options]
 * @param {ReturnType<typeof createMarsTunnelBridge>} [options.bridge] Override for tests
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string} [options.host] The server's bind host; loopback gets the Host check
 * @param {string | null} [options.key] Session key every request must present; null = open (loopback dev only — see mountMarsWebBridge)
 * @param {number} [options.pingIntervalMs]
 * @param {(message: string, error?: unknown) => void} [options.log]
 */
export function createMarsWebBridge({
  bridge: bridgeOverride,
  env = process.env,
  host = DEFAULT_BIND_HOST,
  key = null,
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

  /** Connected sessions: id → the real upstream base (the tunnel listener). */
  const upstreams = new Map();
  /** Live WebSocket pairs, so dispose() can close them. */
  const sockets = new Set();
  /** Per-process cookie value handed to requests that presented the key. */
  const cookieToken = randomBytes(32).toString("hex");

  /**
   * Whether the request may use the bridge. With no key configured every
   * same-host request may. Otherwise it must carry the key (HTTP) or the
   * cookie a keyed request earned (HTTP or WebSocket upgrade). Returns
   * `"issue"` when the key was presented so the response can set the cookie.
   */
  function authorize(req) {
    if (!key) return "ok";
    const presented = req.headers[AUTH_HEADER];
    if (typeof presented === "string" && safeEqual(presented, key)) {
      return "issue";
    }
    const cookie = readCookie(req, AUTH_COOKIE);
    if (cookie && safeEqual(cookie, cookieToken)) return "ok";
    return "deny";
  }

  function cookieHeader() {
    return `${AUTH_COOKIE}=${cookieToken}; HttpOnly; SameSite=Strict; Path=${MARS_WEB_PREFIX}`;
  }

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
    async destroySession(sessionId) {
      // Only forget the upstream once harness-api agreed: on a 409/423 the
      // session is still running and the user must keep their connection.
      await bridge.destroySession(sessionId);
      upstreams.delete(sessionId);
    },
    deleteAgentConfig: (configId) => bridge.deleteAgentConfig(configId),
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
    return { upstream: new URL(base) };
  }

  function proxyHttp(req, res, { sessionId, path }) {
    const target = resolveUpstream(sessionId);
    if (!target) {
      writeJson(res, 409, {
        error: { message: `session ${sessionId} is not connected` },
      });
      return;
    }
    const { upstream } = target;
    const proxyBase = sessionProxyBase(req, sessionId);
    const isTls = upstream.protocol === "https:";
    const upstreamReq = (isTls ? httpsRequest : httpRequest)(
      {
        protocol: upstream.protocol,
        hostname: upstream.hostname,
        port: upstream.port || (isTls ? 443 : 80),
        method: req.method,
        path: `${upstream.pathname.replace(/\/$/, "")}${path}`,
        headers: forwardHeaders(req, upstream),
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
        // Buffer for the URL rewrite up to the limit; past it, hand the
        // response over untouched (what was buffered first, then the rest
        // piped) rather than truncating it into broken JSON.
        const chunks = [];
        let size = 0;
        let streaming = false;
        upstreamRes.on("data", (chunk) => {
          if (streaming) return;
          size += chunk.length;
          if (size <= JSON_REWRITE_LIMIT_BYTES) {
            chunks.push(chunk);
            return;
          }
          streaming = true;
          res.writeHead(upstreamRes.statusCode ?? 502, headers);
          for (const buffered of chunks.splice(0)) res.write(buffered);
          res.write(chunk);
          upstreamRes.pipe(res);
        });
        upstreamRes.on("end", () => {
          if (streaming) return;
          let body = Buffer.concat(chunks);
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
    const { upstream } = target;
    const wsUrl = new URL(
      `${upstream.pathname.replace(/\/$/, "")}${path}`,
      upstream,
    );
    wsUrl.protocol = upstream.protocol === "https:" ? "wss:" : "ws:";
    const protocols = req.headers["sec-websocket-protocol"]
      ?.split(",")
      .map((p) => p.trim())
      .filter(Boolean);
    const upstreamSocket = new WebSocket(wsUrl, protocols, {
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

    // Until the upstream handshake completes the browser's socket is still
    // raw; if it goes away first, nothing else would notice and the upstream
    // would open anyway and be pinged forever (holding the sandbox awake).
    const abandonBeforeOpen = () => {
      if (client) return;
      teardown();
      upstreamSocket.terminate();
    };
    socket.on("close", abandonBeforeOpen);
    socket.on("error", abandonBeforeOpen);

    upstreamSocket.on("unexpected-response", (upstreamReq, res) => {
      teardown();
      upstreamReq.destroy();
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
      if (isCrossOrigin(req) || isForeignHost(req, host)) {
        writeJson(res, 403, {
          error: { message: "cross-origin request refused" },
        });
        return true;
      }
      const pathOnly = url.split(/[?#]/)[0];
      if (pathOnly === HEALTH_PATH) {
        // Unauthenticated on purpose: it only says the bridge is here.
        writeJson(res, 200, { ok: true, authRequired: key !== null });
        return true;
      }
      const auth = authorize(req);
      if (auth === "deny") {
        writeJson(res, 401, { error: { message: "session key required" } });
        return true;
      }
      if (auth === "issue") res.setHeader("Set-Cookie", cookieHeader());
      if (pathOnly.startsWith(RPC_PREFIX)) {
        void handleRpc(req, res, pathOnly.slice(RPC_PREFIX.length));
        return true;
      }
      let session;
      try {
        session = parseSessionPath(url);
      } catch (error) {
        if (!(error instanceof BadSessionPath)) throw error;
        writeJson(res, 400, { error: { message: error.message } });
        return true;
      }
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
      if (isCrossOrigin(req) || isForeignHost(req, host)) {
        socket.end("HTTP/1.1 403 Forbidden\r\n\r\n");
        return true;
      }
      if (authorize(req) === "deny") {
        socket.end("HTTP/1.1 401 Unauthorized\r\n\r\n");
        return true;
      }
      let session;
      try {
        session = parseSessionPath(url);
      } catch (error) {
        if (!(error instanceof BadSessionPath)) throw error;
        socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
        return true;
      }
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
