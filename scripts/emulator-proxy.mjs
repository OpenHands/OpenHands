/**
 * Android emulator (noVNC) reverse proxy for Agent Canvas.
 *
 * Mounted at `/api/emulator` on ingress / static-server.
 * - POST /api/emulator/start  — validate session key, probe upstream, set cookie
 * - GET  /api/emulator/status — readiness without forcing start
 * - /api/emulator/*           — authenticated HTTP proxy to engagement noVNC
 * - upgrade /api/emulator/*   — authenticated WebSocket proxy
 *
 * Upstream resolution (never expose internal URL to the browser):
 * 1. options.upstreamUrl / env EMULATOR_NOVNC_URL
 * 2. options.runtimeServicesInfo.services.android_emulator.url_from_agent
 * 3. unavailable empty state
 *
 * Auth: X-Session-API-Key or HttpOnly cookie Path=/api/emulator.
 */

import { createHash } from "node:crypto";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { connect as netConnect } from "node:net";
import { URL } from "node:url";
import { readSessionApiKey } from "./appwrite-proxy.mjs";
import { readCookie } from "./desktop-proxy.mjs";

export const EMULATOR_PROXY_PATH_PREFIX = "/api/emulator";
export const EMULATOR_AUTH_COOKIE = "agent-canvas-emulator-auth";
export const EMULATOR_IFRAME_PATH = `${EMULATOR_PROXY_PATH_PREFIX}/?autoconnect=1&reconnect=1&path=api/emulator/websockify`;

const AUTH_CACHE_TTL_MS = 30_000;
const DEFAULT_HEALTH_TIMEOUT_MS = 2_000;

const EMULATOR_STATIC_ASSET_RE =
  /\.(?:js|mjs|css|map|png|svg|jpe?g|gif|webp|ico|woff2?|ttf|otf|mp3|oga|wav|wasm)(?:$|\?)/i;

/**
 * @param {string} url
 */
export function isEmulatorProxyRequest(url) {
  const path = (url ?? "/").split("?")[0];
  return (
    path === EMULATOR_PROXY_PATH_PREFIX ||
    path.startsWith(`${EMULATOR_PROXY_PATH_PREFIX}/`)
  );
}

/**
 * @param {string} urlPath
 */
export function rewriteEmulatorProxyPath(urlPath) {
  const path = (urlPath ?? "/").split("?")[0];
  if (
    path === EMULATOR_PROXY_PATH_PREFIX ||
    path === `${EMULATOR_PROXY_PATH_PREFIX}/`
  ) {
    return "/";
  }
  const prefix = `${EMULATOR_PROXY_PATH_PREFIX}/`;
  if (!path.startsWith(prefix)) {
    throw new Error(`Not an Emulator proxy path: ${path}`);
  }
  const rest = path.slice(prefix.length);
  return rest ? `/${rest.replace(/^\/+/, "")}` : "/";
}

/**
 * @param {string} urlPath
 */
export function isEmulatorStaticAssetPath(urlPath) {
  const path = (urlPath ?? "/").split("?")[0];
  if (
    path === EMULATOR_PROXY_PATH_PREFIX ||
    path === `${EMULATOR_PROXY_PATH_PREFIX}/` ||
    path === `${EMULATOR_PROXY_PATH_PREFIX}/index.html`
  ) {
    return false;
  }
  return EMULATOR_STATIC_ASSET_RE.test(path);
}

/**
 * @param {string} sessionApiKey
 */
export function emulatorAuthToken(sessionApiKey) {
  return createHash("sha256")
    .update(`emulator:${sessionApiKey}`)
    .digest("hex");
}

/**
 * Resolve upstream noVNC base URL.
 *
 * Order: options.upstreamUrl → EMULATOR_NOVNC_URL →
 * runtime_services.services.android_emulator.url_from_agent.
 *
 * @param {{
 *   upstreamUrl?: string | null,
 *   runtimeServicesInfo?: unknown,
 * }} [options]
 * @returns {string | null}
 */
export function resolveEmulatorUpstreamUrl(options = {}) {
  const fromOption = String(options.upstreamUrl ?? "").trim();
  if (fromOption) return fromOption.replace(/\/+$/, "");

  const fromEnv = String(process.env.EMULATOR_NOVNC_URL ?? "").trim();
  if (fromEnv) return fromEnv.replace(/\/+$/, "");

  let runtime = options.runtimeServicesInfo;
  if (typeof runtime === "string" && runtime.trim()) {
    try {
      runtime = JSON.parse(runtime);
    } catch {
      runtime = null;
    }
  }
  const fromRuntime = String(
    runtime &&
      typeof runtime === "object" &&
      "services" in runtime &&
      runtime.services &&
      typeof runtime.services === "object" &&
      runtime.services.android_emulator &&
      typeof runtime.services.android_emulator === "object"
      ? runtime.services.android_emulator.url_from_agent ?? ""
      : "",
  ).trim();
  if (fromRuntime) return fromRuntime.replace(/\/+$/, "");

  return null;
}

/**
 * @param {import('node:http').ServerResponse} res
 * @param {number} status
 * @param {unknown} body
 * @param {Record<string, string>} [extraHeaders]
 */
function writeJson(res, status, body, extraHeaders = {}) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(payload),
    ...extraHeaders,
  });
  res.end(payload);
}

/**
 * @param {string} agentServerUrl
 * @param {string} sessionApiKey
 * @param {string} path
 * @param {typeof fetch} [fetchImpl]
 */
async function agentServerFetch(
  agentServerUrl,
  sessionApiKey,
  path,
  fetchImpl = fetch,
) {
  const base = agentServerUrl.replace(/\/+$/, "");
  const url = `${base}${path.startsWith("/") ? path : `/${path}`}`;
  return fetchImpl(url, {
    method: "GET",
    headers: {
      Accept: "application/json",
      "X-Session-API-Key": sessionApiKey,
    },
  });
}

/**
 * @param {{
 *   agentServerUrl: string,
 *   upstreamUrl?: string | null,
 *   runtimeServicesInfo?: unknown,
 *   fetchImpl?: typeof fetch,
 *   healthCheckImpl?: (upstream: string) => Promise<boolean>,
 *   upstreamAvailableImpl?: () => boolean,
 * }} options
 */
export function createEmulatorProxyHandler(options) {
  const agentServerUrl = options.agentServerUrl;
  const fetchImpl = options.fetchImpl ?? fetch;
  /** @type {Map<string, { expiresAt: number }>} */
  const authCache = new Map();

  function getUpstream() {
    return resolveEmulatorUpstreamUrl({
      upstreamUrl: options.upstreamUrl,
      runtimeServicesInfo: options.runtimeServicesInfo,
    });
  }

  function isEmulatorAvailable() {
    if (options.upstreamAvailableImpl) {
      return options.upstreamAvailableImpl();
    }
    return Boolean(getUpstream());
  }

  /**
   * @param {string} upstreamBase
   */
  function healthCheck(upstreamBase) {
    if (options.healthCheckImpl) {
      return options.healthCheckImpl(upstreamBase);
    }
    return new Promise((resolve) => {
      let parsed;
      try {
        parsed = new URL(upstreamBase);
      } catch {
        resolve(false);
        return;
      }
      const transport =
        parsed.protocol === "https:" ? httpsRequest : httpRequest;
      const req = transport(
        {
          hostname: parsed.hostname,
          port: parsed.port || (parsed.protocol === "https:" ? 443 : 80),
          path: parsed.pathname.endsWith("/")
            ? `${parsed.pathname}index.html`
            : `${parsed.pathname}/index.html`,
          method: "GET",
          timeout: DEFAULT_HEALTH_TIMEOUT_MS,
        },
        (res) => {
          res.resume();
          const code = res.statusCode ?? 0;
          resolve(code === 200 || code === 401 || code === 403);
        },
      );
      req.on("error", () => resolve(false));
      req.on("timeout", () => {
        req.destroy();
        resolve(false);
      });
      req.end();
    });
  }

  /**
   * @param {string} sessionApiKey
   */
  async function validateSession(sessionApiKey) {
    const cached = authCache.get(sessionApiKey);
    if (cached && cached.expiresAt > Date.now()) {
      return;
    }
    const response = await agentServerFetch(
      agentServerUrl,
      sessionApiKey,
      "/server_info",
      fetchImpl,
    );
    if (response.status === 401) {
      const err = new Error("Invalid X-Session-API-Key");
      // @ts-expect-error status
      err.status = 401;
      throw err;
    }
    if (!response.ok) {
      const err = new Error(
        `Failed to validate session against agent-server (${response.status})`,
      );
      // @ts-expect-error status
      err.status = 502;
      throw err;
    }
    authCache.set(sessionApiKey, {
      expiresAt: Date.now() + AUTH_CACHE_TTL_MS,
    });
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @returns {Promise<string | null>}
   */
  async function resolveAuthenticatedKey(req) {
    const headerKey = readSessionApiKey(req);
    if (headerKey) {
      await validateSession(headerKey);
      return headerKey;
    }
    const cookieToken = readCookie(req, EMULATOR_AUTH_COOKIE);
    if (!cookieToken) {
      return null;
    }
    try {
      await validateSession(cookieToken);
      return cookieToken;
    } catch {
      return null;
    }
  }

  /**
   * @param {import('node:http').ServerResponse} res
   * @param {string} sessionApiKey
   */
  function setAuthCookie(res, sessionApiKey) {
    const cookie = [
      `${EMULATOR_AUTH_COOKIE}=${encodeURIComponent(sessionApiKey)}`,
      "Path=/api/emulator",
      "HttpOnly",
      "SameSite=Lax",
    ].join("; ");
    res.setHeader("Set-Cookie", cookie);
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:http').ServerResponse} res
   * @param {string} upstreamBase
   */
  function proxyEmulatorHttp(req, res, upstreamBase) {
    let parsed;
    try {
      parsed = new URL(upstreamBase);
    } catch (err) {
      writeJson(res, 502, {
        detail: err instanceof Error ? err.message : String(err),
      });
      return;
    }
    const transport = parsed.protocol === "https:" ? httpsRequest : httpRequest;
    const headers = { ...req.headers, host: parsed.host };
    delete headers.cookie;
    delete headers.referer;
    delete headers.origin;
    delete headers["keep-alive"];
    delete headers.connection;
    delete headers["proxy-connection"];
    delete headers["transfer-encoding"];
    delete headers.upgrade;
    delete headers["sec-websocket-key"];
    delete headers["sec-websocket-version"];
    delete headers["sec-websocket-protocol"];
    delete headers["sec-websocket-extensions"];
    delete headers["sec-fetch-site"];
    delete headers["sec-fetch-mode"];
    delete headers["sec-fetch-dest"];
    delete headers["sec-fetch-user"];

    const proxyReq = transport(
      {
        protocol: parsed.protocol,
        hostname: parsed.hostname,
        port: parsed.port || undefined,
        path: req.url,
        method: req.method,
        headers,
      },
      (proxyRes) => {
        const outHeaders = { ...proxyRes.headers };
        delete outHeaders["cross-origin-embedder-policy"];
        delete outHeaders["cross-origin-opener-policy"];
        const reqOrigin =
          typeof req.headers.origin === "string" ? req.headers.origin : null;
        outHeaders["access-control-allow-origin"] = reqOrigin || "*";
        outHeaders["access-control-allow-methods"] = "GET, HEAD, OPTIONS";
        outHeaders["access-control-allow-headers"] =
          "Authorization, Content-Type";
        res.writeHead(proxyRes.statusCode ?? 502, outHeaders);
        proxyRes.pipe(res);
      },
    );
    proxyReq.on("error", (err) => {
      if (!res.headersSent) {
        writeJson(res, 502, {
          detail: err instanceof Error ? err.message : String(err),
        });
      } else {
        res.destroy();
      }
    });
    req.pipe(proxyReq);
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:stream').Duplex} socket
   * @param {Buffer} head
   * @param {string} upstreamBase
   */
  function proxyEmulatorWebSocket(req, socket, head, upstreamBase) {
    let parsed;
    try {
      parsed = new URL(upstreamBase);
    } catch {
      socket.write("HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    const port = Number(
      parsed.port || (parsed.protocol === "https:" ? 443 : 80),
    );
    const wsKey =
      typeof req.headers["sec-websocket-key"] === "string"
        ? req.headers["sec-websocket-key"]
        : "";
    if (!wsKey) {
      socket.write("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    const origin =
      (typeof req.headers.origin === "string" && req.headers.origin) ||
      `http://${parsed.host}`;
    const wsProtocol =
      typeof req.headers["sec-websocket-protocol"] === "string"
        ? req.headers["sec-websocket-protocol"]
        : "binary";
    const path = req.url || "/websockify";

    const upstream = netConnect({ host: parsed.hostname, port });
    let settled = false;
    /** @type {Buffer} */
    let buffer = Buffer.alloc(0);

    const fail = (statusLine = "HTTP/1.1 502 Bad Gateway") => {
      if (settled) return;
      settled = true;
      try {
        socket.write(`${statusLine}\r\nConnection: close\r\n\r\n`);
      } catch {
        // ignore
      }
      socket.destroy();
      upstream.destroy();
    };

    upstream.on("error", () => fail());
    socket.on("error", () => {
      settled = true;
      upstream.destroy();
    });

    upstream.on("connect", () => {
      const lines = [
        `GET ${path} HTTP/1.1`,
        `Host: ${parsed.host}`,
        "Upgrade: websocket",
        "Connection: Upgrade",
        `Sec-WebSocket-Key: ${wsKey}`,
        "Sec-WebSocket-Version: 13",
        `Sec-WebSocket-Protocol: ${wsProtocol}`,
        `Origin: ${origin}`,
        "",
        "",
      ];
      upstream.write(lines.join("\r\n"));
      if (head?.length) {
        upstream.write(head);
      }
    });

    upstream.on("data", (chunk) => {
      if (settled) return;
      buffer = Buffer.concat([buffer, chunk]);
      const headerEnd = buffer.indexOf("\r\n\r\n");
      if (headerEnd < 0) {
        if (buffer.length > 16_384) fail();
        return;
      }
      settled = true;
      const headerText = buffer.subarray(0, headerEnd).toString("latin1");
      const rest = buffer.subarray(headerEnd + 4);
      const statusLine = headerText.split("\r\n")[0] ?? "";
      if (!statusLine.includes("101")) {
        socket.write(`${headerText}\r\n\r\n`);
        if (rest.length) socket.write(rest);
        socket.destroy();
        upstream.destroy();
        return;
      }
      socket.write(buffer.subarray(0, headerEnd + 4));
      if (rest.length) socket.write(rest);
      upstream.pipe(socket);
      socket.pipe(upstream);
    });
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:http').ServerResponse} res
   */
  async function handleControl(req, res) {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname;
    const upstream = getUpstream();
    const available = isEmulatorAvailable();

    if (
      path === `${EMULATOR_PROXY_PATH_PREFIX}/status` &&
      req.method === "GET"
    ) {
      const ready = available && upstream ? await healthCheck(upstream) : false;
      const sessionApiKey = readSessionApiKey(req);
      if (sessionApiKey) {
        try {
          await validateSession(sessionApiKey);
          setAuthCookie(res, sessionApiKey);
        } catch {
          // Keep status probe usable without auth.
        }
      }
      writeJson(res, 200, {
        ready,
        starting: false,
        unavailable: !available,
        url: EMULATOR_IFRAME_PATH,
      });
      return true;
    }

    if (
      path === `${EMULATOR_PROXY_PATH_PREFIX}/start` &&
      req.method === "POST"
    ) {
      const sessionApiKey = readSessionApiKey(req);
      if (!sessionApiKey) {
        writeJson(res, 401, { detail: "Missing X-Session-API-Key" });
        return true;
      }
      try {
        await validateSession(sessionApiKey);
      } catch (err) {
        const status =
          err && typeof err === "object" && "status" in err
            ? Number(err.status) || 502
            : 502;
        writeJson(res, status, {
          detail: err instanceof Error ? err.message : "Unauthorized",
          ready: false,
          unavailable: false,
        });
        return true;
      }

      if (!available || !upstream) {
        writeJson(res, 503, {
          detail:
            "Emulator unavailable — set EMULATOR_NOVNC_URL or advertise android_emulator in runtime_services",
          ready: false,
          unavailable: true,
        });
        return true;
      }

      const ready = await healthCheck(upstream);
      if (!ready) {
        writeJson(res, 503, {
          detail: "Emulator upstream is not reachable",
          ready: false,
          unavailable: false,
        });
        return true;
      }

      setAuthCookie(res, sessionApiKey);
      writeJson(res, 200, {
        ready: true,
        unavailable: false,
        url: EMULATOR_IFRAME_PATH,
      });
      return true;
    }

    return false;
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:http').ServerResponse} res
   */
  async function handleHttp(req, res) {
    const rawUrl = req.url ?? "/";
    if (!isEmulatorProxyRequest(rawUrl)) {
      writeJson(res, 404, { detail: "Not found" });
      return;
    }

    if (await handleControl(req, res)) {
      return;
    }

    const url = new URL(rawUrl, "http://localhost");
    const allowAnonymousStatic =
      (req.method === "GET" || req.method === "HEAD") &&
      isEmulatorStaticAssetPath(url.pathname);

    if (!allowAnonymousStatic) {
      let sessionApiKey;
      try {
        sessionApiKey = await resolveAuthenticatedKey(req);
      } catch (err) {
        const status =
          err && typeof err === "object" && "status" in err
            ? Number(err.status) || 502
            : 502;
        writeJson(res, status, {
          detail: err instanceof Error ? err.message : "Unauthorized",
        });
        return;
      }
      if (!sessionApiKey) {
        writeJson(res, 401, {
          detail:
            "Missing emulator authentication (start the emulator first)",
        });
        return;
      }
    }

    const upstream = getUpstream();
    if (!upstream) {
      writeJson(res, 503, {
        detail: "Emulator is unavailable",
        unavailable: true,
      });
      return;
    }

    if (!(await healthCheck(upstream))) {
      writeJson(res, 503, {
        detail: "Emulator is not running — call POST /api/emulator/start",
        unavailable: false,
      });
      return;
    }

    const rewritten = rewriteEmulatorProxyPath(url.pathname);
    req.url = `${rewritten}${url.search}`;
    proxyEmulatorHttp(req, res, upstream);
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:stream').Duplex} socket
   * @param {Buffer} head
   * @returns {Promise<boolean>}
   */
  async function handleUpgrade(req, socket, head) {
    const rawUrl = req.url ?? "/";
    if (!isEmulatorProxyRequest(rawUrl)) {
      return false;
    }

    let sessionApiKey = null;
    try {
      sessionApiKey = await resolveAuthenticatedKey(req);
    } catch {
      socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return true;
    }
    if (!sessionApiKey) {
      socket.write("HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return true;
    }

    const upstream = getUpstream();
    if (!upstream) {
      socket.write("HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return true;
    }

    const url = new URL(rawUrl, "http://localhost");
    const rewritten = rewriteEmulatorProxyPath(url.pathname);
    req.url = `${rewritten}${url.search}`;
    proxyEmulatorWebSocket(req, socket, head, upstream);
    return true;
  }

  return Object.assign(handleHttp, { handleUpgrade });
}
