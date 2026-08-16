import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { createProxyServer } from "httpxy";

const DEFAULT_PROXY_TIMEOUT_MS = 120_000;
const SERVER_INFO_PATH = "/server_info";
const WORKSPACE_SESSION_COOKIE_NAME = "oh_workspace_session_key";
const BENIGN_SOCKET_ERRORS = new Set([
  "ECONNRESET",
  "EPIPE",
  "ECONNABORTED",
  "ERR_STREAM_PREMATURE_CLOSE",
]);

/**
 * Rewrite a single `Set-Cookie` value for the workspace-session cookie so a
 * browser will store it when the GUI is served over plain `http://` on a
 * non-loopback host (LAN / VM / container IP).
 *
 * The agent-server mints `oh_workspace_session_key` with `Secure`,
 * `SameSite=None` and `Partitioned`; browsers reject that combination over
 * plain HTTP. We drop `Secure` and `Partitioned` (the latter requires
 * `Secure`) and downgrade `SameSite=None` to `SameSite=Lax`.  Any other
 * cookie is returned unchanged.
 */
export function downgradeWorkspaceSessionCookie(setCookieValue) {
  const parts = setCookieValue.split(";").map((part) => part.trim());
  const name = parts[0].split("=")[0].trim();
  if (name !== WORKSPACE_SESSION_COOKIE_NAME) return setCookieValue;

  let changed = false;
  const rewritten = parts
    .filter((part, index) => {
      if (index === 0) return true;
      const attrName = part.split("=")[0].trim().toLowerCase();
      if (attrName === "secure" || attrName === "partitioned") {
        changed = true;
        return false;
      }
      return true;
    })
    .map((part, index) => {
      if (index === 0) return part;
      const [attrName, ...rest] = part.split("=");
      const lowerName = attrName.trim().toLowerCase();
      if (lowerName === "samesite") {
        const value = rest.join("=").trim().toLowerCase();
        if (value === "none") {
          changed = true;
          return "SameSite=Lax";
        }
      }
      return part;
    });

  return changed ? rewritten.join("; ") : setCookieValue;
}

/**
 * Rewrite the workspace-session cookie(s) in a proxied response's
 * `Set-Cookie` header(s) so they can be stored by a browser over plain HTTP.
 * Only invoked when the caller opts in via `stripSecureCookie`.
 */
export function rewriteWorkspaceSessionCookies(proxyRes) {
  const raw = proxyRes.headers["set-cookie"];
  if (raw === undefined) return;
  if (Array.isArray(raw)) {
    proxyRes.headers["set-cookie"] = raw.map(downgradeWorkspaceSessionCookie);
  } else {
    proxyRes.headers["set-cookie"] = downgradeWorkspaceSessionCookie(raw);
  }
}

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

function parseBackendUrl(backendUrl) {
  const url = new URL(backendUrl);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Invalid backend URL");
  }
  return {
    hostname: url.hostname,
    port: Number.parseInt(url.port, 10) || (url.protocol === "https:" ? 443 : 80),
    protocol: url.protocol,
  };
}

function writeInvalidBackendUrlResponse(req, res) {
  const message = "Invalid backend URL";
  console.error(`Proxy error for ${req.url}: ${message}`);
  if (!res.headersSent) {
    res.writeHead(502, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(`Bad Gateway: ${message}`);
  } else {
    res.destroy();
  }
}

export function isServerInfoRequest(req) {
  const pathname = new URL(req.url ?? "/", "http://localhost").pathname;
  return pathname === SERVER_INFO_PATH;
}

export function proxyServerInfoRequest(
  req,
  res,
  backendUrl,
  runtimeServicesInfo,
) {
  let backend;
  try {
    backend = parseBackendUrl(backendUrl);
  } catch {
    writeInvalidBackendUrlResponse(req, res);
    return;
  }

  const request = backend.protocol === "https:" ? httpsRequest : httpRequest;
  const proxyReq = request(
    {
      hostname: backend.hostname,
      port: backend.port,
      path: req.url,
      method: req.method,
      headers: {
        ...req.headers,
        host: `${backend.hostname}:${backend.port}`,
      },
    },
    (proxyRes) => {
      const chunks = [];

      proxyRes.on("data", (chunk) => {
        chunks.push(Buffer.from(chunk));
      });

      proxyRes.on("error", (err) => {
        if (!isBenignSocketError(err)) {
          console.error(`Upstream response error for ${req.url}:`, err.message);
        }
        if (!res.headersSent) {
          res.writeHead(502);
          res.end(`Bad Gateway: ${err.message}`);
        } else {
          res.destroy();
        }
      });

      proxyRes.on("end", () => {
        const statusCode = proxyRes.statusCode ?? 502;
        const headers = { ...proxyRes.headers };
        const originalBody = Buffer.concat(chunks);

        if (statusCode < 200 || statusCode >= 300 || req.method === "HEAD") {
          res.writeHead(statusCode, headers);
          res.end(req.method === "HEAD" ? "" : originalBody);
          return;
        }

        try {
          const serverInfo = JSON.parse(originalBody.toString("utf8"));
          const runtimeServices =
            typeof runtimeServicesInfo === "string"
              ? JSON.parse(runtimeServicesInfo)
              : runtimeServicesInfo;
          const body = Buffer.from(
            JSON.stringify({
              ...serverInfo,
              runtime_services: runtimeServices,
            }),
            "utf8",
          );

          delete headers["content-length"];
          delete headers["transfer-encoding"];
          headers["content-type"] = "application/json; charset=utf-8";
          headers["cache-control"] = "no-store";
          res.writeHead(statusCode, headers);
          res.end(body);
        } catch (err) {
          console.warn(
            `Could not append runtime_services to ${SERVER_INFO_PATH}: ${
              err instanceof Error ? err.message : String(err)
            }`,
          );
          res.writeHead(statusCode, headers);
          res.end(originalBody);
        }
      });
    },
  );

  proxyReq.on("error", (err) => {
    if (!isBenignSocketError(err)) {
      console.error(`Proxy error for ${req.url}:`, err.message);
    }
    if (!res.headersSent) {
      res.writeHead(502);
      res.end(`Bad Gateway: ${err.message}`);
    } else {
      res.destroy();
    }
  });

  req.on("error", (err) => {
    if (!isBenignSocketError(err)) {
      console.error(`Client request error for ${req.url}:`, err.message);
    }
    proxyReq.destroy();
  });

  res.on("error", (err) => {
    if (!isBenignSocketError(err)) {
      console.error(`Client response error for ${req.url}:`, err.message);
    }
    proxyReq.destroy();
  });

  req.pipe(proxyReq, { end: true });
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
  stripSecureCookie = false,
} = {}) {
  const proxy = createProxyServer({
    ws: true,
    changeOrigin: true,
    xfwd: true,
    timeout,
    proxyTimeout,
  });

  // When the operator opts in with `--disable-secure` (a trusted plain-HTTP
  // deployment such as a LAN / VM / container IP), rewrite the
  // workspace-session cookie so the browser will store it over plain HTTP.
  // This fixes workspace file-preview 401s in those setups.
  if (stripSecureCookie) {
    proxy.on("proxyRes", (proxyRes) => {
      rewriteWorkspaceSessionCookies(proxyRes);
    });
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

    const handleProxyError = (err) => {
      metrics.totalErrors += 1;
      if (!isBenignSocketError(err)) {
        console.error(
          `[${label}] Proxy error for ${req.url} -> ${target}:`,
          err,
        );
      }
      writeProxyError(res, err instanceof Error ? err.message : String(err));
      finish();
    };

    try {
      proxy.web(req, res, { target }).catch(handleProxyError);
    } catch (err) {
      handleProxyError(err);
    }
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
