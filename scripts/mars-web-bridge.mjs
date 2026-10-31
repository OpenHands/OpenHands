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
 * sessions, so they want the server's session key on every request: the
 * renderer sends it as `X-Session-API-Key` on `/mars/rpc/*` (the same key it
 * already sends the agent-server), and a request that presents it gets an
 * `HttpOnly; SameSite=Strict; Path=/mars` cookie back, which the browser then
 * attaches on its own to the proxied REST calls and to the WebSocket upgrade
 * — the one request a browser cannot put a header on. An agent's SDK client
 * sends neither on the upgrade; it sends the key as its first message, which
 * the proxy checks before it dials the session. The key is
 * `MARS_WEB_KEY`, else the server's `--session-api-key`; with neither, the
 * bridge mounts only on a loopback bind (dev), where a `Host` check keeps a
 * DNS-rebound page from reaching it (cookies are host-scoped, so with a key in
 * play rebinding cannot carry one). Cross-origin requests are refused outright;
 * JSON POSTs preflight and this server sends no CORS headers.
 *
 * Credentials survive a restart. Their descriptions live under the Canvas
 * state directory (`MARS_STATE_DIR`, else `OH_CANVAS_SAFE_STATE_DIR`, else
 * `~/.openhands/agent-canvas`), and the tokens in the OS keychain, a file
 * encrypted with `OH_SECRET_KEY`, or memory (scripts/mars-secret-store.mjs).
 * The cookie is derived from the session key, so it outlives a restart too and
 * stops working when the key rotates. `MARS_TOKEN` seeds a connection at
 * startup.
 */

import {
  createHash,
  createHmac,
  randomBytes,
  timingSafeEqual,
} from "node:crypto";
import { homedir, networkInterfaces } from "node:os";
import path from "node:path";

import WebSocket, { WebSocketServer } from "ws";

import { DEFAULT_BIND_HOST, isLoopbackBind } from "./bind-host.mjs";
import { readRequestBody } from "./http-body.mjs";
import {
  buildTunnelHost,
  createMarsTunnelBridge,
} from "./mars-tunnel-bridge.mjs";
import { OAUTH_CALLBACK_PATH, renderReturnPage } from "./mars-oauth.mjs";
import { openSecretStore } from "./mars-secret-store.mjs";
import { createProxyHandlers, rewriteJsonResponse } from "./proxy-utils.mjs";

export const MARS_WEB_PREFIX = "/mars";
const HEALTH_PATH = `${MARS_WEB_PREFIX}/health`;
/** Where a new device redeems a pairing link; needs no key, only the link. */
const PAIR_PATH = `${MARS_WEB_PREFIX}/pair`;
/** The page a pairing link opens, under the base path; the secret rides in its fragment. */
const PAIR_PAGE_PATH = "/pair";
const PAIRING_TTL_MS = 5 * 60 * 1000;
const PAIR_BODY_LIMIT_BYTES = 4 * 1024;
const RPC_PREFIX = `${MARS_WEB_PREFIX}/rpc/`;
const SESSIONS_PREFIX = `${MARS_WEB_PREFIX}/sessions/`;

const RPC_BODY_LIMIT_BYTES = 64 * 1024;
/** Past this a JSON body streams through untouched rather than being rewritten. */
const JSON_REWRITE_LIMIT_BYTES = 8 * 1024 * 1024;
/** The session key on HTTP requests; same header the agent-server takes. */
const AUTH_HEADER = "x-session-api-key";
/** Issued to a request that presented the key; carries auth to WebSocket upgrades. */
const AUTH_COOKIE = "mars_web_auth";
/**
 * An upgrade with neither key nor cookie (an agent's SDK client) may send the
 * key as its first message, `{"type":"auth","session_api_key":…}`, the way
 * the Agent Server itself takes it. Nothing reaches the session before that.
 */
const FIRST_MESSAGE_AUTH_TIMEOUT_MS = 10_000;
/**
 * Anyone who can reach the server can open such a socket before the key is
 * checked, so bound what one may send until it logs in (the login frame, and
 * anything a client sends right behind it in the same burst) and how many
 * may wait to log in at once.
 */
export const FIRST_MESSAGE_AUTH_MAX_BYTES = 64 * 1024;
export const FIRST_MESSAGE_AUTH_MAX_PENDING = 32;
/**
 * Unsent bytes a slow reader may leave before the relay pauses its sender;
 * the sender resumes once that falls to a quarter.
 */
const RELAY_HIGH_WATER_BYTES = 1024 * 1024;
/** What the Agent Server closes a socket with when its key is wrong. */
const CLOSE_AUTH_FAILED = 4001;
const CLOSE_UNREACHABLE = 1011;
/** Close codes a peer may not send itself; relayed as their nearest cousin. */
const RESERVED_CLOSE_CODES = new Map([
  [1005, 1000],
  [1006, CLOSE_UNREACHABLE],
  [1015, CLOSE_UNREACHABLE],
]);

/** Domain separator for the cookie value derived from the session key. */
const AUTH_COOKIE_CONTEXT = "mars-web-cookie/v1";
/** Under the Canvas state directory: credential descriptions and secrets file. */
const STATE_SUBDIR = "digitalocean";
/** Response fields whose agent-server URL the renderer follows. */
const URL_FIELDS = new Set(["conversation_url"]);

/**
 * Bridge methods a browser may call — an explicit allow-list, so a method
 * added to the desktop IPC surface is not exposed here until listed. The
 * desktop's `signInWithOAuth` is left out: it opens a browser on the host.
 * Pages sign in with `beginSignIn` and the calls after it instead.
 */
export const RPC_METHODS = new Set([
  "getAuthState",
  "createPairingLink",
  "beginSignIn",
  "awaitSignIn",
  "completeSignIn",
  "cancelSignIn",
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

/** RPC methods the web server answers itself; the bridge has no such method. */
export const WEB_ONLY_RPC_METHODS = new Set(["createPairingLink"]);

/** Request headers that belong to the canvas origin, never to the session host. */
const CANVAS_ONLY_HEADERS = ["cookie", AUTH_HEADER, "origin", "referer"];

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
 * The Canvas state directory the bridge keeps its credentials under: the
 * launcher passes `MARS_STATE_DIR`; a stand-alone static server (Docker) uses
 * the same default the launcher does.
 */
export function resolveMarsStateDir(env = process.env) {
  const configured =
    env.MARS_STATE_DIR?.trim() || env.OH_CANVAS_SAFE_STATE_DIR?.trim();
  if (configured) return configured;
  try {
    return path.join(homedir(), ".openhands", "agent-canvas");
  } catch {
    // No home folder (HOME unset and no passwd entry, as with `docker run
    // --user <uid>`): sign-ins stay in memory rather than Canvas not starting.
    return null;
  }
}

/**
 * The cookie value a request earns by presenting the key. Derived from the
 * key rather than random per process, so a browser stays admitted across a
 * restart; a rotated key changes it, which signs every browser out.
 */
export function deriveAuthCookieValue(key) {
  return createHmac("sha256", key).update(AUTH_COOKIE_CONTEXT).digest("hex");
}

/**
 * The web integration is experimental: mount only with `MARS_WEB=1`.
 * This readiness flag is separate from authentication. An opted-in bridge
 * still refuses a non-loopback bind without a key. Launchers set `MARS_WEB=0`
 * on static children and under Electron so only the web ingress hosts it.
 * Off is silent unless the env suggests someone meant it on
 * (`marsWebOffWarning`).
 *
 * @param {object} [options]
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string} [options.host]
 * @param {string | null} [options.sessionApiKey]
 * @param {(message: string) => void} [options.log]
 * @param {(message: string) => void} [options.debug]
 */
export function mountMarsWebBridge({
  env = process.env,
  host = DEFAULT_BIND_HOST,
  sessionApiKey = null,
  basePath = "",
  log = (message) => console.warn(`[mars-web] ${message}`),
  debug = (message) => console.debug(`[mars-web] ${message}`),
} = {}) {
  if (env.MARS_WEB !== "1") {
    const warning = marsWebOffWarning(env);
    if (warning) log(warning);
    return null;
  }
  const key = resolveMarsWebKey({ env, sessionApiKey });
  if (!key && !isLoopbackBind(host)) {
    log(
      `no key and bind host ${host} is not loopback; not mounting /mars/* (set MARS_WEB_KEY or --session-api-key).`,
    );
    return null;
  }
  if (!key) debug("no key configured; /mars/* is open to local callers only");
  return createMarsWebBridge({
    env,
    host,
    key,
    sessionApiKey,
    basePath,
    stateDir: resolveMarsStateDir(env),
  });
}

/**
 * Why the bridge stays off, for an operator who seems to have wanted it on;
 * null otherwise, so a server nobody pointed at DigitalOcean starts quietly.
 * Any value but `1` or `0` (launchers set `0` on purpose) gets a line, since
 * it may be a mistyped opt-in (`true`, `"1 "`, a quoted `"1"` from
 * `docker run --env-file`), and so does `MARS_TOKEN` with no `MARS_WEB`.
 */
function marsWebOffWarning(env) {
  const value = env.MARS_WEB ?? "";
  if (value.trim() === "0") return null;
  if (value) {
    return `MARS_WEB=${JSON.stringify(value)} is not "1"; not mounting /mars/* (set MARS_WEB=1).`;
  }
  if (env.MARS_TOKEN?.trim()) {
    return `MARS_TOKEN is set but MARS_WEB is not "1"; not mounting /mars/* (set MARS_WEB=1).`;
  }
  return null;
}

function safeEqual(a, b) {
  const x = Buffer.from(a);
  const y = Buffer.from(b);
  return x.length === y.length && timingSafeEqual(x, y);
}

/** Whether a socket's first message is the Agent Server login with the key. */
function isAuthFrame(text, key) {
  let message;
  try {
    message = JSON.parse(text);
  } catch {
    return false;
  }
  return (
    message?.type === "auth" &&
    typeof message.session_api_key === "string" &&
    safeEqual(message.session_api_key, key)
  );
}

/** Whether a request brought the key header or a bridge cookie, right or not. */
function presentsCredentials(req) {
  return (
    req.headers[AUTH_HEADER] !== undefined ||
    readCookie(req, AUTH_COOKIE) !== null
  );
}

/**
 * Forward `from`'s frames to `to`. Frames that arrive while `to` is still
 * connecting wait in order, and `from` is paused while `to` has more than
 * RELAY_HIGH_WATER_BYTES unsent, so a slow reader cannot make this process
 * buffer without bound.
 */
function relayFrames(from, to) {
  const early = [];
  const forward = (data, isBinary) => {
    to.send(data, { binary: isBinary }, () => {
      if (from.isPaused && to.bufferedAmount < RELAY_HIGH_WATER_BYTES / 4) {
        from.resume();
      }
    });
    if (to.bufferedAmount > RELAY_HIGH_WATER_BYTES) from.pause();
  };
  from.on("message", (data, isBinary) => {
    if (to.readyState === WebSocket.CONNECTING) {
      early.push([data, isBinary]);
      from.pause();
    } else if (to.readyState === WebSocket.OPEN) {
      forward(data, isBinary);
    }
  });
  to.once("open", () => {
    for (const [data, isBinary] of early.splice(0)) forward(data, isBinary);
    if (from.isPaused && to.bufferedAmount < RELAY_HIGH_WATER_BYTES) {
      from.resume();
    }
  });
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

/** Hosts DigitalOcean accepts as loopback return addresses, on any port. */
const OAUTH_LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);

/**
 * Where DigitalOcean sends a browser back after sign-in. A page on a
 * loopback address comes back to its own origin. Any other address cannot
 * be registered with DigitalOcean, so the return goes to the same port on
 * 127.0.0.1: on the Canvas host that is this server; on another device it
 * fails to load, and the user pastes its address into the page instead.
 *
 * @returns {{ redirectUri: string, returnsHere: boolean }}
 */
function returnAddress(req) {
  const hostname = requestHostname(req);
  if (OAUTH_LOOPBACK_HOSTS.has(hostname)) {
    return {
      redirectUri: `${requestOrigin(req)}${OAUTH_CALLBACK_PATH}`,
      returnsHere: true,
    };
  }
  let port = "";
  try {
    port = new URL(`http://${req.headers.host ?? ""}`).port;
  } catch {
    // No usable Host header: the default port it is.
  }
  return {
    redirectUri: `http://127.0.0.1${port ? `:${port}` : ""}${OAUTH_CALLBACK_PATH}`,
    returnsHere: false,
  };
}

const ALL_INTERFACES = new Set(["", "0.0.0.0", "::"]);

/** This machine's network addresses, the likeliest home or tailnet ones first. */
function lanAddresses() {
  const rank = (ip) => {
    if (ip.startsWith("192.168.")) return 0;
    if (ip.startsWith("10.")) return 1;
    // Tailscale and other CGNAT tailnets.
    if (/^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\./.test(ip)) return 2;
    return 3;
  };
  return Object.values(networkInterfaces())
    .flat()
    .filter((entry) => entry && entry.family === "IPv4" && !entry.internal)
    .map((entry) => entry.address)
    .sort((a, b) => rank(a) - rank(b));
}

/**
 * Origins another device could open this Canvas on. A page already opened
 * from the network has one: its own. A page on loopback gets this
 * machine's network addresses when Canvas listens on them, and none when
 * Canvas listens only on this computer.
 */
function pairingOrigins(req, bindHost) {
  if (!isLoopbackBind(requestHostname(req))) return [requestOrigin(req)];
  if (isLoopbackBind(bindHost)) return [];
  let port = "";
  try {
    port = new URL(`http://${req.headers.host ?? ""}`).port;
  } catch {
    // No usable Host header: the default port it is.
  }
  const proto =
    req.headers["x-forwarded-proto"]?.split(",")[0].trim() || "http";
  const hosts = ALL_INTERFACES.has(String(bindHost).trim())
    ? lanAddresses()
    : [bindHost];
  return hosts.map((ip) => `${proto}://${ip}${port ? `:${port}` : ""}`);
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
  // Browsers never send a fragment, and the relay's own client refuses to
  // dial one, so it is a 400 here rather than an error later.
  if (url.includes("#")) throw new BadSessionPath("fragment in session path");
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
 * @param {string | null} [options.sessionApiKey] The agent-server's own key, when the server has one; pairing needs it to be `key`
 * @param {string} [options.basePath] Where the app is mounted (`/canvas` in Docker); pairing links open a page under it
 * @param {string | null} [options.stateDir] Canvas state directory; null keeps credentials in memory
 * @param {number} [options.authTimeoutMs] How long a socket may take to send the key
 * @param {(message: string, error?: unknown) => void} [options.log]
 */
export function createMarsWebBridge({
  bridge: bridgeOverride,
  env = process.env,
  host = DEFAULT_BIND_HOST,
  key = null,
  sessionApiKey = null,
  basePath = "",
  stateDir = null,
  authTimeoutMs = FIRST_MESSAGE_AUTH_TIMEOUT_MS,
  log = (message, error) => console.warn(`[mars-web] ${message}`, error ?? ""),
} = {}) {
  const pairPage = `${String(basePath).replace(/\/+$/, "")}${PAIR_PAGE_PATH}`;
  // A paired device keeps the key it gets as its backend's key. A separate
  // MARS_WEB_KEY would not open the agent-server, and handing out the
  // agent-server's key would reach past what the bridge's key allows.
  const backendKey = sessionApiKey?.trim() || null;
  const canPair = Boolean(key) && (!backendKey || backendKey === key);
  const dir = stateDir ? path.join(stateDir, STATE_SUBDIR) : null;
  const bridge =
    bridgeOverride ??
    createMarsTunnelBridge({
      userDataPath: dir,
      secrets: openSecretStore({ env, dir }),
      env,
    });

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
  /** Cookie value handed to requests that presented the key. */
  const cookieToken = key
    ? deriveAuthCookieValue(key)
    : randomBytes(32).toString("hex");

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
  /** Unused pairing secrets, by hash, with when each stops working. */
  const pairings = new Map();

  function prunePairings(now = Date.now()) {
    for (const [hash, expiresAt] of pairings) {
      if (expiresAt <= now) pairings.delete(hash);
    }
  }

  const hashSecret = (secret) =>
    createHash("sha256").update(secret).digest("hex");

  const overrides = {
    /**
     * A one-time link that lets another device in, with the same access as
     * typing the Canvas key. The secret goes in the fragment, which never
     * reaches a server or proxy log.
     */
    createPairingLink(req) {
      if (!canPair) {
        const error = new Error(
          key
            ? "Pairing is off because this Canvas has a separate MARS_WEB_KEY. Enter the session key on the new device instead."
            : "Pairing needs a Canvas session key.",
        );
        error.status = 400;
        throw error;
      }
      prunePairings();
      const secret = randomBytes(32).toString("base64url");
      const expiresAt = Date.now() + PAIRING_TTL_MS;
      pairings.set(hashSecret(secret), expiresAt);
      return {
        links: pairingOrigins(req, host).map(
          (origin) => `${origin}${pairPage}#${secret}`,
        ),
        expiresAt: new Date(expiresAt).toISOString(),
      };
    },
    beginSignIn(req) {
      const { redirectUri, returnsHere } = returnAddress(req);
      return { ...bridge.beginSignIn({ redirectUri }), returnsHere };
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
   * Redeem a pairing link: once, before it expires. The new device gets the
   * key (what typing it would have given) and the bridge cookie.
   */
  async function handlePair(req, res) {
    if (req.method !== "POST") {
      writeJson(res, 405, { error: { message: "use POST" } });
      return;
    }
    let secret = "";
    try {
      const body = await readJsonBody(req, PAIR_BODY_LIMIT_BYTES);
      if (typeof body?.secret === "string") secret = body.secret;
    } catch {
      // Treated as a link that does not work.
    }
    prunePairings();
    const hash = secret ? hashSecret(secret) : null;
    if (!canPair || !hash || !pairings.has(hash)) {
      writeJson(res, 400, {
        error: {
          message:
            "This link was already used or has expired. Make a new one on a device that is already in.",
        },
      });
      return;
    }
    pairings.delete(hash);
    res.setHeader("Set-Cookie", cookieHeader());
    writeJson(res, 200, { result: { sessionApiKey: key } });
  }

  /**
   * Where DigitalOcean sends the browser back. It needs no key: the browser
   * arrives from another site, so it carries no cookie (SameSite=Strict).
   * The `state` in the address must match a sign-in a keyed caller began,
   * once, and the PKCE proof never leaves the bridge, so a guessed or
   * replayed address does nothing.
   */
  async function handleOAuthReturn(req, res) {
    if (req.method !== "GET") {
      writeJson(res, 405, { error: { message: "use GET" } });
      return;
    }
    let page;
    let status = 200;
    try {
      const state = await bridge.completeSignIn(req.url ?? "");
      page = renderReturnPage({ ok: true, teamName: state?.active?.teamName });
    } catch (error) {
      status = 400;
      page = renderReturnPage({ ok: false, message: error?.message });
    }
    res.writeHead(status, {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
      "Referrer-Policy": "no-referrer",
    });
    res.end(page);
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

  /** Accepts upgrades that log in by their first message. */
  const firstMessageAuth = new WebSocketServer({ noServer: true });
  /** Sockets taken but not yet logged in. */
  let pendingLogins = 0;

  /**
   * Take the upgrade, wait for the key as the first message, and only then
   * dial the session and relay both ways. The login frame stays here: the
   * guest agent-server takes no key.
   */
  function acceptFirstMessageAuth(req, socket, head, session) {
    if (pendingLogins >= FIRST_MESSAGE_AUTH_MAX_PENDING) {
      socket.end("HTTP/1.1 503 Service Unavailable\r\n\r\n");
      return;
    }
    pendingLogins += 1;
    // ws buffers a whole frame before handing it over, so the bytes are
    // counted off the raw socket rather than left to its payload limit.
    let received = head.length;
    const countBytes = (chunk) => {
      received += chunk.length;
      if (received > FIRST_MESSAGE_AUTH_MAX_BYTES) socket.destroy();
    };
    let waiting = true;
    const loggedInOrGone = () => {
      if (!waiting) return;
      waiting = false;
      pendingLogins -= 1;
      socket.off("data", countBytes);
    };
    socket.on("data", countBytes);
    socket.once("close", loggedInOrGone);
    track(socket);
    firstMessageAuth.handleUpgrade(req, socket, head, (client) => {
      client.on("error", () => {});
      const timer = setTimeout(
        () => client.close(CLOSE_AUTH_FAILED, "Authentication failed"),
        authTimeoutMs,
      );
      client.once("message", (data, isBinary) => {
        clearTimeout(timer);
        // A key that arrives after the window closed is too late. A refused
        // socket keeps its place and byte limit until it is gone: the peer
        // may take its time answering the close.
        if (client.readyState !== WebSocket.OPEN) return;
        if (isBinary || !isAuthFrame(data.toString("utf8"), key)) {
          client.close(CLOSE_AUTH_FAILED, "Authentication failed");
          return;
        }
        loggedInOrGone();
        relayWebSocket(client, session);
      });
      client.once("close", () => clearTimeout(timer));
    });
  }

  function relayWebSocket(client, { sessionId, path }) {
    const target = upstreamFor(sessionId);
    if (!target) {
      // Fixed text: a close reason over 123 bytes makes ws throw, and a long
      // session id in the path would otherwise take the whole server down.
      client.close(CLOSE_UNREACHABLE, "session is not connected");
      return;
    }
    let upstream;
    try {
      // The guest's pings are passed to the agent and its pongs back, so a
      // dead agent connection is noticed instead of answered for here.
      upstream = new WebSocket(`${target.replace(/^http/, "ws")}${path}`, {
        autoPong: false,
      });
    } catch {
      // ws refuses some URLs synchronously; that must not escape the handler.
      client.close(CLOSE_UNREACHABLE, "session is not reachable");
      return;
    }
    track(upstream);
    upstream.on("error", () => {});
    relayFrames(client, upstream);
    relayFrames(upstream, client);
    upstream.on("ping", (data) => {
      if (client.readyState === WebSocket.OPEN) client.ping(data);
    });
    client.on("pong", (data) => {
      if (upstream.readyState === WebSocket.OPEN) upstream.pong(data);
    });
    upstream.on("close", (code, reason) => {
      if (client.readyState === WebSocket.OPEN) {
        client.close(RESERVED_CLOSE_CODES.get(code) ?? code, reason);
      }
    });
    client.on("close", () => {
      if (upstream.readyState === WebSocket.CONNECTING) upstream.terminate();
      else upstream.close();
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
      if (pathOnly === OAUTH_CALLBACK_PATH) {
        void handleOAuthReturn(req, res);
        return true;
      }
      if (pathOnly === PAIR_PATH) {
        void handlePair(req, res);
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
      const allowed = authorize(req) !== "deny";
      let session;
      try {
        session = parseSessionPath(url);
      } catch (error) {
        if (!(error instanceof BadSessionPath)) throw error;
        socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
        return true;
      }
      if (!session) {
        socket.end(
          allowed
            ? "HTTP/1.1 404 Not Found\r\n\r\n"
            : "HTTP/1.1 401 Unauthorized\r\n\r\n",
        );
        return true;
      }
      if (allowed) {
        proxyWebSocket(req, socket, head, session);
      } else if (presentsCredentials(req)) {
        // A wrong key or a stale cookie (the key was rotated) is refused at
        // once, as before; only a bare upgrade may log in by first message.
        socket.end("HTTP/1.1 401 Unauthorized\r\n\r\n");
      } else {
        acceptFirstMessageAuth(req, socket, head, session);
      }
      return true;
    },

    async dispose() {
      for (const socket of sockets) {
        if (socket instanceof WebSocket) socket.terminate();
        else socket.destroy();
      }
      sockets.clear();
      await bridge.dispose();
    },
  };
}
