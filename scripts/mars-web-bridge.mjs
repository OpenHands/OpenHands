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
 *                                    session's tunnel listener on loopback
 *                                    (the tunnel pings its own upstream
 *                                    socket, see tunnel-client.mjs).
 *   GET  /mars/health              — lets the renderer detect this server.
 *
 * `openTunnel` therefore answers with `host` = `<origin>/mars/sessions/<id>`,
 * a path-prefixed backend the renderer already supports (see
 * src/utils/websocket-url.ts `extractPathPrefix`). The agent-server writes
 * its own origin into `conversation_url`, which the renderer uses to open
 * the event WebSocket, so proxied JSON has those URLs rewritten to the prefix.
 *
 * Trust model. These routes hold the DigitalOcean token and can destroy
 * sessions, so they require the server's session key on every request (or
 * refuse to mount on non-loopback binds when no key is set). Can be explicitly
 * disabled with `MARS_WEB=0`. The renderer sends the key as
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
 * are in-memory only (no OS keychain here, nothing written to disk);
 * `MARS_TOKEN` seeds one at startup.
 */

import { randomBytes, timingSafeEqual } from "node:crypto";

import { DEFAULT_BIND_HOST, isLoopbackBind } from "./bind-host.mjs";
import { readRequestBody } from "./http-body.mjs";
import {
  buildTunnelHost,
  createMarsTunnelBridge,
} from "./mars-tunnel-bridge.mjs";
import { createProxyHandlers, rewriteJsonResponse } from "./proxy-utils.mjs";

export const MARS_WEB_PREFIX = "/mars";
const HEALTH_PATH = `${MARS_WEB_PREFIX}/health`;
const RPC_PREFIX = `${MARS_WEB_PREFIX}/rpc/`;
const SESSIONS_PREFIX = `${MARS_WEB_PREFIX}/sessions/`;

const RPC_BODY_LIMIT_BYTES = 64 * 1024;
/** Past this a JSON body streams through untouched rather than being rewritten. */
const JSON_REWRITE_LIMIT_BYTES = 8 * 1024 * 1024;
/** The session key on HTTP requests; same header the agent-server takes. */
const AUTH_HEADER = "x-session-api-key";
/** Issued to a request that presented the key; carries auth to WebSocket upgrades. */
const AUTH_COOKIE = "mars_web_auth";
/** Response fields whose agent-server URL the renderer follows. */
const URL_FIELDS = new Set(["conversation_url"]);

/**
 * Bridge methods a browser may call — an explicit allow-list, so a method
 * added to the desktop IPC surface is not exposed here until listed. OAuth is
 * left out: its implicit-grant flow needs a loopback redirect in a trusted
 * process.
 */
export const RPC_METHODS = new Set([
  "getAuthState",
  "savePat",
  "setActiveConnection",
  "signOut",
  "listSessions",
  "listAgentConfigs",
  "listConfigSessions",
  "listSandboxSizes",
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

/** Request headers that belong to the canvas origin, never to the session host. */
const CANVAS_ONLY_HEADERS = ["cookie", AUTH_HEADER, "origin", "referer"];

/** Enabled by default: these routes hold the DO token and can destroy sessions. */
export function isMarsWebEnabled(env = process.env) {
  return env.MARS_WEB !== "0";
}

/**
 * The key `/mars/*` requires, or null when none is configured. Explicit
 * `MARS_WEB_KEY` wins so a deployment can decouple it from the agent-server
 * key; otherwise the server's own `--session-api-key` is reused, which is
 * what the renderer already holds.
 *
 * @param {object} [options]
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string | null} [options.sessionApiKey]
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
 *
 * @param {object} [options]
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string} [options.host]
 * @param {string | null} [options.sessionApiKey]
 * @param {(message: string) => void} [options.log]
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
      `No key and bind host ${host} is not loopback; refusing to mount /mars/* (set MARS_WEB_KEY or --session-api-key).`,
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

async function readJsonBody(req, limit) {
  const body = await readRequestBody(req, limit);
  if (body.length === 0) return {};
  try {
    return JSON.parse(body.toString("utf8"));
  } catch {
    throw new Error("request body is not valid JSON");
  }
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

/** Rewrite the request to what the session host should see. */
function toUpstreamRequest(req, path) {
  for (const name of CANVAS_ONLY_HEADERS) delete req.headers[name];
  req.url = path;
}

/**
 * @param {object} [options]
 * @param {ReturnType<typeof createMarsTunnelBridge>} [options.bridge] Override for tests
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string} [options.host] The server's bind host; loopback gets the Host check
 * @param {string | null} [options.key] Session key every request must present; null = open (loopback dev only — see mountMarsWebBridge)
 * @param {(message: string, error?: unknown) => void} [options.log]
 */
export function createMarsWebBridge({
  bridge: bridgeOverride,
  env = process.env,
  host = DEFAULT_BIND_HOST,
  key = null,
  log = (message, error) => console.warn(`[mars-web] ${message}`, error ?? ""),
} = {}) {
  // No userDataPath: credentials stay in this process's memory.
  const bridge =
    bridgeOverride ?? createMarsTunnelBridge({ userDataPath: null, env });

  /** Both ends of proxied WebSockets, so dispose() can close them. */
  const sockets = new Set();
  function track(socket) {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  }
  // Its own proxy rather than the hosting server's: agent-server calls can
  // stay silent far longer than a static backend's timeout, the session host
  // has no use for X-Forwarded-* from the canvas, and errors answer in the
  // same JSON shape as the rest of /mars/*.
  const proxy = createProxyHandlers({
    label: "mars-web",
    timeout: 0,
    proxyTimeout: 0,
    xfwd: false,
    errorBody: (message) => ({
      contentType: "application/json",
      body: JSON.stringify({
        error: { message: `upstream request failed: ${message}` },
      }),
    }),
    onProxySocket: track,
  });
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

  if (env.MARS_TOKEN) {
    bridge
      .savePat({ token: env.MARS_TOKEN, label: "MARS_TOKEN" })
      .catch((error) => log("MARS_TOKEN was not accepted", error));
  }

  /**
   * Methods that differ from the bridge's own, called with the request
   * first; every other RPC_METHODS entry is `bridge[method](...args)`.
   */
  const overrides = {
    getAuthState() {
      // The web build takes personal access tokens only.
      return { ...bridge.getAuthState(), canUseOAuth: false };
    },
    async openTunnel(req, params) {
      const status = await bridge.openTunnel(params);
      return status?.host
        ? { ...status, host: sessionProxyBase(req, status.sessionId) }
        : status;
    },
  };

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
      const result = Object.hasOwn(overrides, method)
        ? await overrides[method](req, ...args)
        : await bridge[method](...args);
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

  /**
   * The session's tunnel listener, read from the bridge on every request so
   * a tunnel it dropped (closeTunnel, destroySession, sign-out) is never
   * proxied to after its port was freed.
   */
  function upstreamFor(sessionId) {
    const tunnel = bridge.getTunnel(sessionId);
    return tunnel?.status === "connected" && tunnel.localPort !== undefined
      ? buildTunnelHost(tunnel.localPort)
      : null;
  }

  function proxyHttp(req, res, { sessionId, path }) {
    const target = upstreamFor(sessionId);
    if (!target) {
      writeJson(res, 409, {
        error: { message: `session ${sessionId} is not connected` },
      });
      return;
    }
    const proxyBase = sessionProxyBase(req, sessionId);
    toUpstreamRequest(req, path);
    proxy.proxyHttp(req, res, target, {
      // Responses are read as UTF-8 for the URL rewrite; a compressed body
      // would have to be inflated first.
      headers: { "accept-encoding": "identity" },
      onResponse: rewriteJsonResponse(
        (body) => rewriteAgentServerUrls(body, proxyBase),
        { limit: JSON_REWRITE_LIMIT_BYTES },
      ),
    });
  }

  function proxyWebSocket(req, socket, head, { sessionId, path }) {
    const target = upstreamFor(sessionId);
    if (!target) {
      socket.end("HTTP/1.1 409 Conflict\r\n\r\n");
      return;
    }
    toUpstreamRequest(req, path);
    track(socket);
    proxy.proxyWebSocket(req, socket, head, target);
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
      for (const socket of sockets) socket.destroy();
      sockets.clear();
      await bridge.dispose();
    },
  };
}
