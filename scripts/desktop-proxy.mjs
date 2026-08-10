/**
 * Desktop (KasmVNC) reverse proxy for Agent Canvas.
 *
 * Mounted at `/api/desktop` on ingress / static-server.
 * - POST /api/desktop/start  — validate session key, ensure KasmVNC is up, set cookie
 * - GET  /api/desktop/status — readiness without side effects
 * - /api/desktop/*           — authenticated HTTP proxy to loopback KasmVNC
 * - upgrade /api/desktop/*   — authenticated WebSocket proxy
 *
 * The VNC port is never published; auth is enforced here via X-Session-API-Key
 * or the HttpOnly cookie issued by /start.
 */

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { connect as netConnect } from "node:net";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { URL } from "node:url";

const MODULE_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_DESKTOP_DIR = join(MODULE_DIR, "..", "docker", "desktop");
const IMAGE_DESKTOP_DIR = "/opt/agent-canvas/desktop";

/**
 * Prefer the Docker image path; fall back to the repo checkout for local
 * static-server/ingress (still requires `vncserver` on PATH to be usable).
 */
export function resolveDesktopDir() {
  if (process.env.AGENT_CANVAS_DESKTOP_DIR) {
    return process.env.AGENT_CANVAS_DESKTOP_DIR;
  }
  if (existsSync(join(IMAGE_DESKTOP_DIR, "start-desktop.sh"))) {
    return IMAGE_DESKTOP_DIR;
  }
  if (existsSync(join(REPO_DESKTOP_DIR, "start-desktop.sh"))) {
    return REPO_DESKTOP_DIR;
  }
  return IMAGE_DESKTOP_DIR;
}

/**
 * @returns {boolean}
 */
export function isVncServerInstalled() {
  try {
    const result = spawnSync("bash", ["-lc", "command -v vncserver"], {
      encoding: "utf8",
      timeout: 3000,
    });
    return result.status === 0 && Boolean(result.stdout?.trim());
  } catch {
    return false;
  }
}

import { readSessionApiKey } from "./appwrite-proxy.mjs";

export const DESKTOP_PROXY_PATH_PREFIX = "/api/desktop";
export const DESKTOP_AUTH_COOKIE = "agent-canvas-desktop-auth";
/** Prefer autoconnect so the iframe reaches XFCE without the Kasm control chrome.
 * `path` must be relative to the site root (noVNC builds `ws://host/<path>`),
 * otherwise the client dials `/websockify` and bypasses our `/api/desktop` proxy.
 */
export const DESKTOP_IFRAME_PATH = `${DESKTOP_PROXY_PATH_PREFIX}/index.html?autoconnect=1&reconnect=1&resize=remote&path=api/desktop/websockify`;

/**
 * KasmVNC's index.html loads JS/CSS with the `crossorigin` attribute, which
 * makes the browser omit cookies (CORS "anonymous"). Those subresources must
 * still be reachable after /start authenticated the document itself.
 */
const DESKTOP_STATIC_ASSET_RE =
  /\.(?:js|mjs|css|map|png|svg|jpe?g|gif|webp|ico|woff2?|ttf|otf|mp3|oga|wav|wasm)(?:$|\?)/i;

/**
 * @param {string} urlPath
 */
export function isDesktopStaticAssetPath(urlPath) {
  const path = (urlPath ?? "/").split("?")[0];
  if (
    path === DESKTOP_PROXY_PATH_PREFIX ||
    path === `${DESKTOP_PROXY_PATH_PREFIX}/` ||
    path === `${DESKTOP_PROXY_PATH_PREFIX}/index.html`
  ) {
    return false;
  }
  return DESKTOP_STATIC_ASSET_RE.test(path);
}

const DEFAULT_VNC_PORT = Number(process.env.DESKTOP_VNC_PORT || 6901);
const DEFAULT_START_TIMEOUT_MS = 30_000;
const AUTH_CACHE_TTL_MS = 30_000;

/**
 * @param {string} url
 */
export function isDesktopProxyRequest(url) {
  const path = (url ?? "/").split("?")[0];
  return (
    path === DESKTOP_PROXY_PATH_PREFIX ||
    path.startsWith(`${DESKTOP_PROXY_PATH_PREFIX}/`)
  );
}

/**
 * @param {string} urlPath
 */
export function rewriteDesktopProxyPath(urlPath) {
  const path = (urlPath ?? "/").split("?")[0];
  if (
    path === DESKTOP_PROXY_PATH_PREFIX ||
    path === `${DESKTOP_PROXY_PATH_PREFIX}/`
  ) {
    return "/";
  }
  const prefix = `${DESKTOP_PROXY_PATH_PREFIX}/`;
  if (!path.startsWith(prefix)) {
    throw new Error(`Not a Desktop proxy path: ${path}`);
  }
  const rest = path.slice(prefix.length);
  return rest ? `/${rest.replace(/^\/+/, "")}` : "/";
}

/**
 * @param {import('node:http').IncomingMessage} req
 * @param {string} name
 */
export function readCookie(req, name) {
  const raw = req.headers.cookie;
  if (!raw || typeof raw !== "string") return null;
  for (const part of raw.split(";")) {
    const [k, ...rest] = part.trim().split("=");
    if (k === name) {
      try {
        return decodeURIComponent(rest.join("="));
      } catch {
        return rest.join("=");
      }
    }
  }
  return null;
}

/**
 * @param {string} sessionApiKey
 */
export function desktopAuthToken(sessionApiKey) {
  return createHash("sha256").update(`desktop:${sessionApiKey}`).digest("hex");
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
  const response = await fetchImpl(url, {
    method: "GET",
    headers: {
      Accept: "application/json",
      "X-Session-API-Key": sessionApiKey,
    },
  });
  return response;
}

/**
 * @param {{
 *   agentServerUrl: string,
 *   vncPort?: number,
 *   startScriptPath?: string,
 *   stopScriptPath?: string,
 *   startTimeoutMs?: number,
 *   fetchImpl?: typeof fetch,
 *   spawnImpl?: typeof spawn,
 *   healthCheckImpl?: () => Promise<boolean>,
 *   desktopAvailableImpl?: () => boolean,
 * }} options
 */
export function createDesktopProxyHandler(options) {
  const agentServerUrl = options.agentServerUrl;
  const vncPort = options.vncPort ?? DEFAULT_VNC_PORT;
  const startTimeoutMs = options.startTimeoutMs ?? DEFAULT_START_TIMEOUT_MS;
  const fetchImpl = options.fetchImpl ?? fetch;
  const spawnImpl = options.spawnImpl ?? spawn;
  const desktopDir = resolveDesktopDir();
  const startScriptPath =
    options.startScriptPath ?? join(desktopDir, "start-desktop.sh");
  const stopScriptPath =
    options.stopScriptPath ?? join(desktopDir, "stop-desktop.sh");

  const kasmUser = process.env.DESKTOP_VNC_USER || "openhands";
  const kasmPassword = process.env.DESKTOP_VNC_PASSWORD || "canvas";
  const kasmBasicAuth = Buffer.from(`${kasmUser}:${kasmPassword}`).toString(
    "base64",
  );

  /**
   * Native HTTP reverse-proxy (avoids httpxy/undici + KasmVNC Connection:close).
   * Strips COEP/COOP so the Canvas iframe can embed the UI.
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:http').ServerResponse} res
   */
  function proxyDesktopHttp(req, res) {
    const headers = { ...req.headers, host: `127.0.0.1:${vncPort}` };
    headers.authorization = `Basic ${kasmBasicAuth}`;
    // Never forward Canvas cookies / hop-by-hop headers to KasmVNC — they can
    // make the upstream treat static GETs as failed websocket upgrades (401).
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
    delete headers["sec-websocket-origin"];
    // KasmVNC returns 401 for static assets when Referer points at our
    // /api/desktop URL (outside its own httpd root). Drop Sec-Fetch-* too.
    delete headers["sec-fetch-site"];
    delete headers["sec-fetch-mode"];
    delete headers["sec-fetch-dest"];
    delete headers["sec-fetch-user"];
    delete headers["sec-fetch-storage-access"];
    const proxyReq = httpRequest(
      {
        hostname: "127.0.0.1",
        port: vncPort,
        path: req.url,
        method: req.method,
        headers,
      },
      (proxyRes) => {
        const outHeaders = { ...proxyRes.headers };
        delete outHeaders["cross-origin-embedder-policy"];
        delete outHeaders["cross-origin-opener-policy"];
        // KasmVNC's index.html loads module scripts/CSS with `crossorigin`
        // (anonymous). Without ACAO the browser refuses to execute/apply them
        // under /api/desktop, leaving the UI in the broken fallback state.
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
   * Native WebSocket upgrade to KasmVNC with the headers it requires.
   *
   * Important: KasmVNC's websocket checks are case-sensitive on header names
   * (`Host`, `Sec-WebSocket-Origin`, `Sec-WebSocket-Protocol`). Node's
   * `http.request` lowercases header names, which Kasm treats as "missing".
   * We therefore write the upgrade request on a raw TCP socket.
   *
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:stream').Duplex} socket
   * @param {Buffer} head
   */
  function proxyDesktopWebSocket(req, socket, head) {
    const origin =
      (typeof req.headers.origin === "string" && req.headers.origin) ||
      (typeof req.headers["sec-websocket-origin"] === "string" &&
        req.headers["sec-websocket-origin"]) ||
      `http://127.0.0.1:${vncPort}`;
    const wsKey =
      typeof req.headers["sec-websocket-key"] === "string"
        ? req.headers["sec-websocket-key"]
        : "";
    if (!wsKey) {
      socket.write("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n");
      socket.destroy();
      return;
    }
    const wsProtocol =
      typeof req.headers["sec-websocket-protocol"] === "string"
        ? req.headers["sec-websocket-protocol"]
        : "binary";
    const path = req.url || "/websockify";

    const upstream = netConnect({ host: "127.0.0.1", port: vncPort });
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
        `Host: 127.0.0.1:${vncPort}`,
        "Upgrade: websocket",
        "Connection: Upgrade",
        `Sec-WebSocket-Key: ${wsKey}`,
        "Sec-WebSocket-Version: 13",
        `Sec-WebSocket-Protocol: ${wsProtocol}`,
        `Origin: ${origin}`,
        `Sec-WebSocket-Origin: ${origin}`,
        `Authorization: Basic ${kasmBasicAuth}`,
        "",
        "",
      ];
      upstream.write(lines.join("\r\n"));
      if (head?.length) {
        upstream.write(head);
      }
    });

    upstream.on("data", (chunk) => {
      if (settled) {
        return;
      }
      buffer = Buffer.concat([buffer, chunk]);
      const headerEnd = buffer.indexOf("\r\n\r\n");
      if (headerEnd < 0) {
        if (buffer.length > 16_384) {
          fail();
        }
        return;
      }
      settled = true;
      const headerText = buffer.subarray(0, headerEnd).toString("latin1");
      const rest = buffer.subarray(headerEnd + 4);
      const statusLine = headerText.split("\r\n")[0] ?? "";
      if (!statusLine.includes("101")) {
        socket.write(`${headerText}\r\n\r\n`);
        if (rest.length) {
          socket.write(rest);
        }
        socket.destroy();
        upstream.destroy();
        return;
      }
      // Forward the upstream 101 response as-is (already correctly cased).
      socket.write(buffer.subarray(0, headerEnd + 4));
      if (rest.length) {
        socket.write(rest);
      }
      upstream.pipe(socket);
      socket.pipe(upstream);
    });
  }

  /** @type {Map<string, { expiresAt: number }>} */
  const authCache = new Map();
  /** @type {Promise<void> | null} */
  let startInFlight = null;

  function isDesktopAvailable() {
    if (options.desktopAvailableImpl) {
      return options.desktopAvailableImpl();
    }
    // Scripts alone are not enough — without KasmVNC (`vncserver`) the start
    // script always fails (typical on Windows npm/Electron hosts).
    return existsSync(startScriptPath) && isVncServerInstalled();
  }

  /**
   * Probe KasmVNC without undici `fetch`.
   *
   * KasmVNC answers `/index.html` with `Connection: close` and a large body.
   * Leaving an undici response paused when the socket closes triggers
   * `AssertionError: assert(!this.paused)` and kills the whole Node process
   * (ingress/static-server), which is what made the Desktop tab look like a
   * start failure after each click.
   */
  function healthCheck() {
    if (options.healthCheckImpl) {
      return options.healthCheckImpl();
    }
    return new Promise((resolve) => {
      const req = httpRequest(
        {
          hostname: "127.0.0.1",
          port: vncPort,
          path: "/index.html",
          method: "GET",
          headers: { Authorization: `Basic ${kasmBasicAuth}` },
          timeout: 1500,
        },
        (res) => {
          // Drain so the socket can close cleanly; we only need the status.
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
    authCache.set(sessionApiKey, { expiresAt: Date.now() + AUTH_CACHE_TTL_MS });
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
    const cookieToken = readCookie(req, DESKTOP_AUTH_COOKIE);
    if (!cookieToken) {
      return null;
    }
    // Cookie stores the raw session key (HttpOnly, Path=/api/desktop).
    // Prefer validating as session key; also accept legacy token form.
    try {
      await validateSession(cookieToken);
      return cookieToken;
    } catch {
      return null;
    }
  }

  function runScript(scriptPath) {
    return new Promise((resolve, reject) => {
      if (!existsSync(scriptPath)) {
        const err = new Error(`Desktop script missing: ${scriptPath}`);
        // @ts-expect-error status
        err.status = 503;
        reject(err);
        return;
      }
      const child = spawnImpl("bash", [scriptPath], {
        env: {
          ...process.env,
          DESKTOP_VNC_PORT: String(vncPort),
          HOME: process.env.HOME || homedir(),
        },
        stdio: ["ignore", "pipe", "pipe"],
      });
      let stderr = "";
      child.stderr?.on("data", (chunk) => {
        stderr += String(chunk);
      });
      const timer = setTimeout(() => {
        child.kill("SIGTERM");
        const err = new Error("Desktop start timed out");
        // @ts-expect-error status
        err.status = 504;
        reject(err);
      }, startTimeoutMs);
      child.on("error", (err) => {
        clearTimeout(timer);
        reject(err);
      });
      child.on("exit", (code) => {
        clearTimeout(timer);
        if (code === 0 || code === null) {
          resolve();
          return;
        }
        if (code === 2) {
          const err = new Error("Desktop unavailable (KasmVNC not installed)");
          // @ts-expect-error status
          err.status = 503;
          reject(err);
          return;
        }
        const err = new Error(
          stderr.trim() || `Desktop start failed (exit ${code})`,
        );
        // @ts-expect-error status
        err.status = 500;
        reject(err);
      });
    });
  }

  async function ensureDesktopRunning() {
    if (await healthCheck()) {
      return;
    }
    if (!isDesktopAvailable()) {
      const err = new Error("Desktop unavailable outside the Docker image");
      // @ts-expect-error status
      err.status = 503;
      throw err;
    }
    if (!startInFlight) {
      startInFlight = runScript(startScriptPath).finally(() => {
        startInFlight = null;
      });
    }
    await startInFlight;
    const ready = await healthCheck();
    if (!ready) {
      const err = new Error("Desktop started but health check failed");
      // @ts-expect-error status
      err.status = 504;
      throw err;
    }
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:http').ServerResponse} res
   * @param {string} sessionApiKey
   */
  function setAuthCookie(res, sessionApiKey) {
    const cookie = [
      `${DESKTOP_AUTH_COOKIE}=${encodeURIComponent(sessionApiKey)}`,
      "Path=/api/desktop",
      "HttpOnly",
      "SameSite=Lax",
    ].join("; ");
    res.setHeader("Set-Cookie", cookie);
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:http').ServerResponse} res
   */
  async function handleControl(req, res) {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname;

    if (path === `${DESKTOP_PROXY_PATH_PREFIX}/status` && req.method === "GET") {
      const available = isDesktopAvailable();
      const ready = available ? await healthCheck() : false;
      // Mint the auth cookie whenever the client already has a valid session
      // key. The UI may open the iframe as soon as status.ready is true and
      // skip POST /start — without this cookie the iframe gets 401.
      const sessionApiKey = readSessionApiKey(req);
      if (sessionApiKey) {
        try {
          await validateSession(sessionApiKey);
          setAuthCookie(res, sessionApiKey);
        } catch {
          // Ignore invalid keys on status — keep the probe unauthenticated.
        }
      }
      writeJson(res, 200, {
        ready,
        starting: Boolean(startInFlight),
        unavailable: !available,
        url: DESKTOP_IFRAME_PATH,
      });
      return true;
    }

    if (path === `${DESKTOP_PROXY_PATH_PREFIX}/start` && req.method === "POST") {
      const sessionApiKey = readSessionApiKey(req);
      if (!sessionApiKey) {
        writeJson(res, 401, { detail: "Missing X-Session-API-Key" });
        return true;
      }
      try {
        await validateSession(sessionApiKey);
        await ensureDesktopRunning();
      } catch (err) {
        const status =
          err && typeof err === "object" && "status" in err
            ? Number(err.status) || 502
            : 502;
        writeJson(res, status, {
          detail:
            err instanceof Error ? err.message : "Failed to start desktop",
          ready: false,
          unavailable: status === 503,
        });
        return true;
      }
      setAuthCookie(res, sessionApiKey);
      writeJson(res, 200, {
        ready: true,
        unavailable: false,
        url: DESKTOP_IFRAME_PATH,
      });
      return true;
    }

    if (path === `${DESKTOP_PROXY_PATH_PREFIX}/stop` && req.method === "POST") {
      const sessionApiKey = readSessionApiKey(req);
      if (!sessionApiKey) {
        writeJson(res, 401, { detail: "Missing X-Session-API-Key" });
        return true;
      }
      try {
        await validateSession(sessionApiKey);
        if (existsSync(stopScriptPath)) {
          await runScript(stopScriptPath);
        }
      } catch (err) {
        const status =
          err && typeof err === "object" && "status" in err
            ? Number(err.status) || 502
            : 502;
        writeJson(res, status, {
          detail: err instanceof Error ? err.message : "Failed to stop desktop",
        });
        return true;
      }
      writeJson(res, 200, { ready: false });
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
    if (!isDesktopProxyRequest(rawUrl)) {
      writeJson(res, 404, { detail: "Not found" });
      return;
    }

    if (await handleControl(req, res)) {
      return;
    }

    const url = new URL(rawUrl, "http://localhost");
    const allowAnonymousStatic =
      (req.method === "GET" || req.method === "HEAD") &&
      isDesktopStaticAssetPath(url.pathname);

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
          detail: "Missing desktop authentication (start the desktop first)",
        });
        return;
      }
    }

    if (!(await healthCheck())) {
      writeJson(res, 503, {
        detail: "Desktop is not running — call POST /api/desktop/start",
        unavailable: !isDesktopAvailable(),
      });
      return;
    }

    const rewritten = rewriteDesktopProxyPath(url.pathname);
    req.url = `${rewritten}${url.search}`;
    proxyDesktopHttp(req, res);
  }

  /**
   * Authenticated WebSocket upgrade handler.
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:stream').Duplex} socket
   * @param {Buffer} head
   * @returns {Promise<boolean>} true if this request was handled
   */
  async function handleUpgrade(req, socket, head) {
    const rawUrl = req.url ?? "/";
    if (!isDesktopProxyRequest(rawUrl)) {
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

    const url = new URL(rawUrl, "http://localhost");
    const rewritten = rewriteDesktopProxyPath(url.pathname);
    req.url = `${rewritten}${url.search}`;
    proxyDesktopWebSocket(req, socket, head);
    return true;
  }

  return Object.assign(handleHttp, { handleUpgrade });
}
