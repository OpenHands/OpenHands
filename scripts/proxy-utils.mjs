import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { isIP } from "node:net";
import { createProxyServer } from "httpxy";

const DEFAULT_PROXY_TIMEOUT_MS = 120_000;
const SERVER_INFO_PATH = "/server_info";
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

function firstForwardedEntry(value) {
  return typeof value === "string" ? value.split(",")[0].trim() : "";
}

/**
 * The client-facing host/port/protocol of a request: what the browser
 * connected to, taken from the first `X-Forwarded-*` entry when a reverse
 * proxy is in front, else the Host header and the connection itself. These
 * values are only as trustworthy as whatever set them — with direct
 * exposure the client controls them exactly as it controls Host — so the
 * upstream's own checks remain the authority. Returns `null` when there
 * is no usable host entry to forward.
 */
function clientFacingIdentity(req) {
  const forwardedHost = req.headers["x-forwarded-host"];
  const host = forwardedHost
    ? firstForwardedEntry(forwardedHost)
    : typeof req.headers.host === "string"
      ? req.headers.host
      : "";
  // Don't turn a nonempty forwarded list with a blank first entry into Host:
  // the editor proxy would reject that list. Leave the original header alone.
  if (!host) return null;
  const proto =
    firstForwardedEntry(req.headers["x-forwarded-proto"]) ||
    (req.socket && req.socket.encrypted ? "https" : "http");
  let port = firstForwardedEntry(req.headers["x-forwarded-port"]);
  if (!port) {
    try {
      const parsed = new URL(`${proto}://${host}`);
      if (parsed.port) port = parsed.port;
    } catch {
      // A malformed Host: fall through to the protocol's default port.
    }
  }
  if (!port) port = proto === "https" ? "443" : "80";
  return { host, port, proto };
}

/**
 * `X-Forwarded-*` headers describing that client-facing identity, for
 * proxies whose upstream validates the browser's Origin against it — the
 * agent-server editor proxy checks `x-forwarded-host || Host`, and
 * code-server can also use the forwarded port. On the direct Docker path,
 * rewriting Host without forwarding it broke the Origin check (#19945).
 */
function clientForwardedHeaders(req) {
  const identity = clientFacingIdentity(req);
  if (!identity) return {};
  const headers = {
    "x-forwarded-host": identity.host,
    "x-forwarded-port": identity.port,
    "x-forwarded-proto": identity.proto,
  };
  const forwardedFor =
    typeof req.headers["x-forwarded-for"] === "string"
      ? req.headers["x-forwarded-for"]
      : "";
  const remoteAddress = req.socket && req.socket.remoteAddress;
  if (remoteAddress) {
    headers["x-forwarded-for"] = forwardedFor
      ? `${forwardedFor},${remoteAddress}`
      : remoteAddress;
  } else if (forwardedFor) {
    headers["x-forwarded-for"] = forwardedFor;
  }
  return headers;
}

function clientIdentityProxyOptions(req, target, headers) {
  const url = new URL(target);
  const options = {
    target: url,
    headers: { ...clientForwardedHeaders(req), ...headers },
  };
  if (url.protocol === "https:" || url.protocol === "wss:") {
    const hostname = url.hostname.replace(/^\[|\]$/g, "");
    // httpxy seeds its outgoing request options from ssl. Pin the TLS name
    // to the target, not the preserved HTTP Host; omit SNI for IP literals.
    options.ssl = { servername: isIP(hostname) ? "" : hostname };
  }
  return options;
}

function parseBackendUrl(backendUrl) {
  const url = new URL(backendUrl);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Invalid backend URL");
  }
  return {
    hostname: url.hostname,
    port:
      Number.parseInt(url.port, 10) || (url.protocol === "https:" ? 443 : 80),
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

/**
 * An `onResponse` for createProxyHandlers' proxyHttp that rewrites a JSON
 * body through `transform` — buffered, re-serialized, and sent with a fixed
 * Content-Length, as proxyServerInfoRequest does. Any other response, an
 * encoded one, and a JSON body past `limit` bytes stream through untouched
 * (what was buffered first, then the rest), never truncated into broken
 * JSON.
 *
 * @param {(body: unknown) => unknown} transform
 * @param {{ limit: number }} options
 */
export function rewriteJsonResponse(transform, { limit }) {
  return (proxyRes, req, res) => {
    const status = proxyRes.statusCode ?? 502;
    const headers = { ...proxyRes.headers };
    const encoding = headers["content-encoding"];
    const isJson = /\bapplication\/json\b/i.test(headers["content-type"] ?? "");
    if (
      !isJson ||
      req.method === "HEAD" ||
      (encoding && encoding !== "identity")
    ) {
      res.writeHead(status, headers);
      proxyRes.pipe(res);
      return;
    }
    const chunks = [];
    let size = 0;
    let streaming = false;
    proxyRes.on("data", (chunk) => {
      if (streaming) return;
      size += chunk.length;
      if (size <= limit) {
        chunks.push(chunk);
        return;
      }
      streaming = true;
      res.writeHead(status, headers);
      for (const buffered of chunks.splice(0)) res.write(buffered);
      res.write(chunk);
      proxyRes.pipe(res);
    });
    proxyRes.on("end", () => {
      if (streaming) return;
      let body = Buffer.concat(chunks);
      try {
        body = Buffer.from(
          JSON.stringify(transform(JSON.parse(body.toString("utf8")))),
        );
      } catch {
        // Not JSON after all; pass it through as received.
      }
      delete headers["transfer-encoding"];
      headers["content-length"] = String(body.length);
      res.writeHead(status, headers);
      res.end(body);
    });
  };
}

function once(fn) {
  let called = false;
  return (...args) => {
    if (called) return;
    called = true;
    fn(...args);
  };
}

function textErrorBody(message) {
  return {
    contentType: "text/plain; charset=utf-8",
    body: `Bad Gateway: ${message}`,
  };
}

function writeProxyError(res, message, errorBody) {
  if (res.destroyed) return;
  if (!res.headersSent) {
    const { contentType, body } = errorBody(message);
    res.writeHead(502, { "Content-Type": contentType });
    res.end(body);
    return;
  }
  res.destroy();
}

function emptyProxyMetrics() {
  return {
    activeHttpRequests: 0,
    activeWebSockets: 0,
    totalHttpRequests: 0,
    totalWebSockets: 0,
    totalErrors: 0,
  };
}

/**
 * @param {object} [options]
 * @param {string} [options.label] Prefix for log lines and metrics
 * @param {ReturnType<typeof emptyProxyMetrics>} [options.metrics] Share counters
 *   between handlers exposed through one diagnostics listener.
 * @param {number} [options.timeout] Client socket idle timeout in ms; 0 = none
 * @param {number} [options.proxyTimeout] Upstream response timeout in ms; 0 = none
 * @param {boolean} [options.xfwd] Add X-Forwarded-* headers upstream
 * @param {boolean} [options.forwardClientIdentity] Forward the client-facing
 *   Host and X-Forwarded-* identity (host/port/proto/for) to the upstream on
 *   both proxied HTTP requests and WebSocket upgrades, instead of rewriting
 *   Host to the local hop. For upstreams that validate the browser's Origin
 *   against that identity — the agent-server editor proxy and code-server
 *   behind it (#19945). Replaces httpxy's own xfwd pass, which would fill
 *   x-forwarded-port/proto from the local hop.
 * @param {(message: string) => { contentType: string, body: string }} [options.errorBody]
 *   The 502 written when the upstream fails before a response started
 * @param {(proxySocket: import("node:net").Socket) => void} [options.onProxySocket]
 *   Called with each upstream socket of an established WebSocket proxy
 */
export function createProxyHandlers({
  label = "proxy",
  timeout = DEFAULT_PROXY_TIMEOUT_MS,
  proxyTimeout = DEFAULT_PROXY_TIMEOUT_MS,
  xfwd = true,
  forwardClientIdentity = false,
  metrics = emptyProxyMetrics(),
  errorBody = textErrorBody,
  onProxySocket,
} = {}) {
  const proxy = createProxyServer({
    ws: true,
    changeOrigin: !forwardClientIdentity,
    xfwd: forwardClientIdentity ? false : xfwd,
    timeout,
    proxyTimeout,
  });
  if (onProxySocket) proxy.on("proxySocket", onProxySocket);
  if (forwardClientIdentity) {
    // The upstream's non-101 answer to an upgrade (e.g. the editor proxy's
    // 401 for a mismatched Origin) is piped to the client, but nothing logs
    // it; without a line here the rejection is invisible server-side. Keep
    // both the request query and the configured target out of the log.
    proxy.on("proxyReqWs", (proxyReq, req) => {
      proxyReq.once("response", (proxyRes) => {
        const path = (req.url ?? "/").split("?", 1)[0];
        console.error(
          `[${label}:editor] WebSocket upstream rejected ${path}: HTTP ${proxyRes.statusCode}`,
        );
      });
    });
  }

  /** Per-request `onResponse` handlers; see proxyHttp. */
  const responseHandlers = new WeakMap();
  proxy.on("proxyRes", (proxyRes, req, res) => {
    responseHandlers.get(req)?.(proxyRes, req, res);
  });

  proxy.on("error", (err, _req, resOrSocket, target) => {
    metrics.totalErrors += 1;
    const targetText = target ? ` -> ${target}` : "";
    if (!isBenignSocketError(err)) {
      console.error(`[${label}] Proxy error${targetText}: ${err.message}`);
    }
    if (resOrSocket && typeof resOrSocket.writeHead === "function") {
      writeProxyError(resOrSocket, err.message, errorBody);
    } else if (resOrSocket && typeof resOrSocket.destroy === "function") {
      resOrSocket.destroy();
    }
  });

  /**
   * @param {import("node:http").IncomingMessage} req
   * @param {import("node:http").ServerResponse} res
   * @param {string} target
   * @param {object} [options]
   * @param {Record<string, string>} [options.headers] Set on the upstream request
   * @param {(proxyRes: import("node:http").IncomingMessage, req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => void} [options.onResponse]
   *   Writes the response itself instead of it being piped through (e.g.
   *   rewriteJsonResponse).
   */
  function proxyHttp(req, res, target, { headers, onResponse } = {}) {
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
      writeProxyError(
        res,
        err instanceof Error ? err.message : String(err),
        errorBody,
      );
      finish();
    };

    try {
      const options = forwardClientIdentity
        ? clientIdentityProxyOptions(req, target, headers)
        : { target, headers };
      if (onResponse) {
        responseHandlers.set(req, onResponse);
        options.selfHandleResponse = true;
      }
      proxy.web(req, res, options).catch(handleProxyError);
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
      proxy
        .ws(
          req,
          socket,
          forwardClientIdentity
            ? clientIdentityProxyOptions(req, target)
            : { target },
          head,
        )
        .catch((err) => {
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
