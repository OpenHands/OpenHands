import { createProxyServer } from "httpxy";

const DEFAULT_PROXY_TIMEOUT_MS = 120_000;
const BENIGN_SOCKET_ERRORS = new Set([
  "ECONNRESET",
  "EPIPE",
  "ECONNABORTED",
  "ERR_STREAM_PREMATURE_CLOSE",
]);

export function matchesPathPrefix(url, prefix) {
  return (
    url === prefix ||
    url.startsWith(prefix + "/") ||
    url.startsWith(prefix + "?")
  );
}

export function createRouter(routes, defaultBackend = null) {
  const sortedRoutes = Object.entries(routes).sort(
    ([a], [b]) => b.length - a.length,
  );

  return function route(url) {
    for (const [prefix, backend] of sortedRoutes) {
      if (matchesPathPrefix(url, prefix)) {
        return backend;
      }
    }
    return defaultBackend;
  };
}

export function isBenignSocketError(err) {
  return Boolean(err && BENIGN_SOCKET_ERRORS.has(err.code));
}

/**
 * Rewrite a single `Set-Cookie` value so it is usable from a plain-HTTP
 * origin. Browsers refuse to store a cookie carrying the `Secure` flag over
 * HTTP, and `SameSite=none` is only valid with `Secure`, so strip both and
 * drop the `Partitioned` attribute (which also requires `Secure`). Returns
 * the rewritten cookie string, or the original string when it carries no
 * HTTP-incompatible attributes.
 *
 * This is what makes the agent-server's workspace-session cookie
 * (`oh_workspace_session_key`, minted with `Secure; SameSite=none;
 * Partitioned`) actually reach the browser jar in a local dev / static stack
 * served over plain HTTP — without it the file viewer's iframe / <img>
 * requests to the static workspace fileserver have no credential and 401.
 *
 * @param {string} cookie
 * @returns {string}
 */
export function downgradeSecureCookie(cookie) {
  if (!cookie) return cookie;
  const parts = cookie.split(/;\s*/).filter(Boolean);
  if (
    !parts.some(
      (p) =>
        p.toLowerCase() === "secure" ||
        p.toLowerCase() === "partitioned" ||
        p.toLowerCase().startsWith("samesite=none"),
    )
  ) {
    return cookie;
  }

  // Drop HTTP-incompatible attributes. A non-none `SameSite` attribute (e.g.
  // `SameSite=Lax`) survives the filter and must not be duplicated by the
  // `SameSite=none` → `SameSite=Lax` replacement below, so we remember
  // whether the source already had one.
  let hasSameSite = false;
  const rewritten = [];
  for (const part of parts) {
    const lower = part.toLowerCase();
    if (
      lower === "secure" ||
      lower === "partitioned" ||
      lower === "samesite=none"
    ) {
      continue;
    }
    if (lower.startsWith("samesite=")) {
      hasSameSite = true;
    }
    rewritten.push(part);
  }
  if (!hasSameSite) rewritten.push("SameSite=Lax");
  return rewritten.join("; ");
}

/**
 * Install a `proxyRes` listener on an httpxy `ProxyServer` that downgrades
 * every `Set-Cookie` response header to an HTTP-safe form when the incoming
 * client request was made over plain HTTP (not TLS). Over HTTPS the cookies
 * pass through untouched, preserving the server's `Secure` posture for
 * production deployments.
 *
 * httpxy emits `proxyRes` *before* its outgoing middleware copies headers
 * onto the downstream response, so mutating `proxyRes.headers["set-cookie"]`
 * here is observed when the headers are written.
 *
 * @param {import("httpxy").ProxyServer} proxy
 */
export function installSecureCookieDowngrade(proxy) {
  proxy.on("proxyRes", (proxyRes, req) => {
    if (req.socket?.encrypted || req.connection?.encrypted) return;
    const raw = proxyRes.headers["set-cookie"];
    if (!raw) return;
    const list = Array.isArray(raw) ? raw : [raw];
    proxyRes.headers["set-cookie"] = list.map(downgradeSecureCookie);
  });
}

function once(fn) {
  let called = false;
  return (...args) => {
    if (called) return;
    called = true;
    fn(...args);
  };
}

function writeProxyError(res, message) {
  if (res.destroyed) return;
  if (!res.headersSent) {
    res.writeHead(502, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(`Bad Gateway: ${message}`);
    return;
  }
  res.destroy();
}

export function createProxyHandlers({
  label = "proxy",
  timeout = DEFAULT_PROXY_TIMEOUT_MS,
  proxyTimeout = DEFAULT_PROXY_TIMEOUT_MS,
  downgradeSecureCookieOverHttp = true,
} = {}) {
  const proxy = createProxyServer({
    ws: true,
    changeOrigin: true,
    xfwd: true,
    timeout,
    proxyTimeout,
  });

  // Downgrade the agent-server's `Secure; SameSite=none; Partitioned`
  // workspace-session cookie to an HTTP-safe form when the client connects
  // over plain HTTP. Without this the browser drops the cookie (it carries
  // `Secure`), and iframe / <img> requests to the static workspace fileserver
  // have no credential and 401. Disabled (no-op) over TLS.
  if (downgradeSecureCookieOverHttp) {
    installSecureCookieDowngrade(proxy);
  }
  const metrics = {
    activeHttpRequests: 0,
    activeWebSockets: 0,
    totalHttpRequests: 0,
    totalWebSockets: 0,
    totalErrors: 0,
  };

  proxy.on("error", (err, _req, resOrSocket, target) => {
    metrics.totalErrors += 1;
    const targetText = target ? ` -> ${target}` : "";
    if (!isBenignSocketError(err)) {
      console.error(`[${label}] Proxy error${targetText}: ${err.message}`);
    }
    if (resOrSocket && typeof resOrSocket.writeHead === "function") {
      writeProxyError(resOrSocket, err.message);
    } else if (resOrSocket && typeof resOrSocket.destroy === "function") {
      resOrSocket.destroy();
    }
  });

  function proxyHttp(req, res, target) {
    metrics.activeHttpRequests += 1;
    metrics.totalHttpRequests += 1;
    const finish = once(() => {
      metrics.activeHttpRequests = Math.max(0, metrics.activeHttpRequests - 1);
    });
    res.on("close", finish);
    res.on("finish", finish);
    res.on("error", finish);

    proxy.web(req, res, { target }).catch((err) => {
      metrics.totalErrors += 1;
      if (!isBenignSocketError(err)) {
        console.error(
          `[${label}] Proxy error for ${req.url} -> ${target}:`,
          err,
        );
      }
      writeProxyError(res, err instanceof Error ? err.message : String(err));
      finish();
    });
  }

  function proxyWebSocket(req, socket, head, target) {
    metrics.activeWebSockets += 1;
    metrics.totalWebSockets += 1;
    const finish = once(() => {
      metrics.activeWebSockets = Math.max(0, metrics.activeWebSockets - 1);
    });
    socket.on("close", finish);
    socket.on("error", finish);

    try {
      proxy.ws(req, socket, { target }, head).catch((err) => {
        metrics.totalErrors += 1;
        if (!isBenignSocketError(err)) {
          console.error(
            `[${label}] WebSocket proxy error for ${req.url} -> ${target}:`,
            err,
          );
        }
        socket.destroy();
        finish();
      });
    } catch (err) {
      metrics.totalErrors += 1;
      if (!isBenignSocketError(err)) {
        console.error(
          `[${label}] WebSocket proxy error for ${req.url} -> ${target}:`,
          err,
        );
      }
      socket.destroy();
      finish();
    }
  }

  function dumpMetrics() {
    console.log(
      `[${label}] active_http=${metrics.activeHttpRequests} ` +
        `active_ws=${metrics.activeWebSockets} ` +
        `total_http=${metrics.totalHttpRequests} ` +
        `total_ws=${metrics.totalWebSockets} ` +
        `total_errors=${metrics.totalErrors}`,
    );
  }

  function installDiagnostics(signal = "SIGUSR1") {
    process.on(signal, dumpMetrics);
    return () => {
      process.off(signal, dumpMetrics);
    };
  }

  return {
    proxyHttp,
    proxyWebSocket,
    dumpMetrics,
    installDiagnostics,
    metrics,
  };
}
