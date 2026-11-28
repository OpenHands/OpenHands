/**
 * Bind-address helpers for the local Agent Canvas stack.
 *
 * Default is IPv4 loopback. Binding `::` / `0.0.0.0` exposes the UI (and any
 * session key injected into index.html) to every peer on the network.
 */

export const DEFAULT_BIND_HOST = "127.0.0.1";

const STATIC_LOOPBACK_HOSTS = new Set(["::1", "localhost"]);
const IPV4_LOOPBACK_RE = /^127(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}$/;
// `_` is not valid in DNS names but is common in hostnames that never reach DNS
// (Docker Compose service names, /etc/hosts aliases); WHATWG URL accepts it.
const LOCALHOST_SUBDOMAIN_RE =
  /^(?:[a-z\d_](?:[a-z\d_-]*[a-z\d_])?\.)+localhost$/i;
const HOST_AUTHORITY_RE = /^[a-z\d_.:[\]-]+$/i;
const SAFE_HTTP_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

export function isLoopbackBind(host) {
  if (host == null || host === "") return false;
  const normalized = String(host)
    .trim()
    .replace(/^\[|\]$/g, "")
    .toLowerCase();
  return (
    STATIC_LOOPBACK_HOSTS.has(normalized) || IPV4_LOOPBACK_RE.test(normalized)
  );
}

/**
 * Parse a `Host` header authority without permitting URL userinfo (`@`),
 * whitespace, commas, or path/query suffixes.
 *
 * @param {unknown} rawHost
 * @returns {{ host: string, hostname: string, port: string } | null}
 */
export function parseHttpHostHeader(rawHost) {
  if (typeof rawHost !== "string") return null;
  const trimmed = rawHost.trim();
  if (!trimmed || !HOST_AUTHORITY_RE.test(trimmed)) return null;
  try {
    const parsed = new URL(`http://${trimmed}`);
    if (
      parsed.username ||
      parsed.password ||
      parsed.pathname !== "/" ||
      parsed.search ||
      parsed.hash
    ) {
      return null;
    }
    const hostname = parsed.hostname.replace(/^\[|\]$/g, "").toLowerCase();
    if (!hostname) return null;
    return {
      host: parsed.host.toLowerCase(),
      hostname,
      port: parsed.port,
    };
  } catch {
    return null;
  }
}

/**
 * Parse an HTTP `Origin` header into a lowercase authority and unbracketed
 * hostname, rejecting `"null"`, userinfo, and non-HTTP(S) schemes.
 *
 * @param {unknown} rawOrigin
 * @returns {{ host: string, hostname: string } | null}
 */
export function parseHttpOriginHeader(rawOrigin) {
  if (typeof rawOrigin !== "string") return null;
  const trimmed = rawOrigin.trim();
  if (!trimmed || trimmed === "null") return null;
  try {
    const parsed = new URL(trimmed);
    if (
      (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
      parsed.username ||
      parsed.password
    ) {
      return null;
    }
    const hostname = parsed.hostname.replace(/^\[|\]$/g, "").toLowerCase();
    if (!hostname) return null;
    return {
      host: parsed.host.toLowerCase(),
      hostname,
    };
  } catch {
    return null;
  }
}

/**
 * Whether an unbracketed lowercase hostname refers to loopback (`127.0.0.0/8`,
 * `::1`, `localhost`, or `<label>.localhost` per RFC 6761).
 *
 * @param {string} hostname
 */
export function isLoopbackHostname(hostname) {
  if (!hostname) return false;
  return isLoopbackBind(hostname) || LOCALHOST_SUBDOMAIN_RE.test(hostname);
}

/**
 * Extra hostnames a loopback listener accepts in `Host` and `Origin`, from
 * comma-separated `OH_ALLOWED_HOSTS`: the public name of a tunnel or reverse
 * proxy (ngrok, Caddy, cloudflared) that forwards to it.
 *
 * @param {Record<string, string | undefined>} [env]
 * @returns {Set<string>}
 */
export function resolveAllowedHosts(env = process.env) {
  return new Set(
    String(env.OH_ALLOWED_HOSTS ?? "")
      .split(",")
      .map((host) => host.trim().toLowerCase())
      .filter(Boolean),
  );
}

/**
 * Check whether an incoming `Origin` header differs from `req.headers.host`.
 * Requests without `Origin` (plain navigations, non-browser clients) pass.
 *
 * @param {import("node:http").IncomingMessage} req
 */
export function isCrossOriginHttpRequest(req) {
  const rawOrigin = req.headers?.origin;
  if (rawOrigin === undefined) return false;
  const parsedOrigin = parseHttpOriginHeader(rawOrigin);
  const parsedHost = parseHttpHostHeader(req.headers?.host);
  if (!parsedOrigin || !parsedHost) return true;
  return parsedOrigin.host !== parsedHost.host;
}

/**
 * On a loopback listener, reject requests whose `Host` header is missing,
 * malformed, carries userinfo (`@`), or names a host that is neither loopback
 * nor in `OH_ALLOWED_HOSTS` (DNS rebinding), and reject WebSocket upgrades or
 * unsafe HTTP mutations carrying a foreign or `null` `Origin` (CSWSH / CSRF).
 *
 * @param {import("node:http").IncomingMessage} req
 * @param {string} [bindHost]
 * @param {{ upgrade?: boolean, env?: Record<string, string | undefined> }} [options]
 */
export function isForeignLoopbackRequest(
  req,
  bindHost,
  { upgrade = false, env = process.env } = {},
) {
  if (!isLoopbackBind(bindHost)) return false;
  const allowedHosts = resolveAllowedHosts(env);
  const isTrusted = (hostname) =>
    isLoopbackHostname(hostname) || allowedHosts.has(hostname);
  const parsedHost = parseHttpHostHeader(req.headers?.host);
  if (!parsedHost || !isTrusted(parsedHost.hostname)) {
    return true;
  }
  const rawOrigin = req.headers?.origin;
  if (rawOrigin === undefined) return false;
  const method = String(req.method ?? "GET").toUpperCase();
  if (!upgrade && SAFE_HTTP_METHODS.has(method)) return false;
  const parsedOrigin = parseHttpOriginHeader(rawOrigin);
  return !parsedOrigin || !isTrusted(parsedOrigin.hostname);
}

/**
 * Resolve the listen address.
 * CLI `--host` wins, then `OH_BIND_HOST`, then loopback.
 */
export function resolveBindHost({ flag, env } = {}) {
  const fromFlag =
    flag != null && String(flag).trim() !== "" ? String(flag).trim() : null;
  const fromEnv =
    env != null && String(env).trim() !== "" ? String(env).trim() : null;
  return fromFlag || fromEnv || DEFAULT_BIND_HOST;
}

export function bindHostArgs(host) {
  return ["--host", host || DEFAULT_BIND_HOST];
}

/**
 * Session keys must not be written into unauthenticated HTML when the
 * listener is reachable off-loopback. Convert to public-mode (`authRequired`)
 * unless the caller explicitly opts back in (container entrypoints).
 *
 * @param {object} [opts]
 * @param {string} [opts.host]
 * @param {string | null} [opts.sessionApiKey]
 * @param {boolean} [opts.authRequired]
 * @param {boolean} [opts.allowLanSessionKey]
 * @param {(msg: string) => void} [opts.warn]
 */
export function applySessionKeyPolicy(opts = {}) {
  const {
    host,
    sessionApiKey = null,
    authRequired = false,
    allowLanSessionKey = false,
    warn = console.warn,
  } = opts;
  const loopback = isLoopbackBind(host);
  if (loopback || allowLanSessionKey || !sessionApiKey) {
    return {
      sessionApiKey: sessionApiKey || null,
      authRequired: Boolean(authRequired),
      strippedSessionKey: false,
    };
  }

  warn(
    `WARNING: bind host ${host} is not loopback; refusing to inject the session API key into HTML.\n` +
      "  The UI will use the API key entry screen (same as --public).\n" +
      "  Pass --host 127.0.0.1 (default) for local mode, or --allow-lan-session-key only if you accept LAN exposure.",
  );

  return {
    sessionApiKey: null,
    authRequired: true,
    strippedSessionKey: true,
  };
}
