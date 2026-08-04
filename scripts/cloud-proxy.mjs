/**
 * Local `/api/cloud-proxy` handler for Cloud runtime sandbox hops.
 *
 * Cloud sandboxes (`*.prod-runtime.all-hands.dev`) reject browser CORS from
 * localhost / Electron / self-hosted Canvas origins. The TypeScript client's
 * `hostOverride` path POSTs an envelope here; this module forwards the
 * upstream call server-side so CORS does not apply.
 *
 * The agent-server no longer ships this endpoint; Canvas owns it in the
 * ingress / static-server / Vite edge so Cloud bash/git/file runtime calls
 * keep working.
 */

import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";

export const CLOUD_PROXY_PATH = "/api/cloud-proxy";

const DEFAULT_TIMEOUT_SECONDS = 30;
const MAX_TIMEOUT_SECONDS = 120;

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

/**
 * @param {string | null | undefined} hostname
 * @returns {boolean}
 */
export function isAllowedCloudProxyHost(hostname) {
  if (!hostname) return false;
  const host = hostname.trim().toLowerCase();
  if (!host) return false;
  if (LOOPBACK_HOSTS.has(host)) return true;
  // Production / staging conversation runtimes.
  if (host.endsWith(".prod-runtime.all-hands.dev")) return true;
  if (host.endsWith(".staging-runtime.all-hands.dev")) return true;
  if (host.endsWith(".runtime.all-hands.dev")) return true;
  return false;
}

/**
 * @param {import("node:http").IncomingMessage} req
 * @returns {boolean}
 */
export function isCloudProxyRequest(req) {
  if ((req.method ?? "GET").toUpperCase() !== "POST") return false;
  const pathname = new URL(req.url ?? "/", "http://localhost").pathname;
  return pathname === CLOUD_PROXY_PATH;
}

/**
 * @param {import("node:http").IncomingMessage} req
 * @returns {Promise<unknown>}
 */
async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  if (!raw) {
    throw Object.assign(new Error("Request body is required"), { status: 400 });
  }
  try {
    return JSON.parse(raw);
  } catch {
    throw Object.assign(new Error("Request body must be JSON"), { status: 400 });
  }
}

/**
 * @param {unknown} envelope
 * @returns {{
 *   upstreamUrl: URL,
 *   method: string,
 *   headers: Record<string, string>,
 *   body: unknown,
 *   timeoutMs: number,
 * }}
 */
export function parseCloudProxyEnvelope(envelope) {
  if (!envelope || typeof envelope !== "object" || Array.isArray(envelope)) {
    throw Object.assign(new Error("Invalid proxy envelope"), { status: 400 });
  }

  const host = typeof envelope.host === "string" ? envelope.host.trim() : "";
  const path = typeof envelope.path === "string" ? envelope.path.trim() : "";
  const method =
    typeof envelope.method === "string" && envelope.method.trim()
      ? envelope.method.trim().toUpperCase()
      : "GET";

  if (!host) {
    throw Object.assign(new Error("Envelope host is required"), { status: 400 });
  }
  if (!path || !path.startsWith("/")) {
    throw Object.assign(new Error("Envelope path must start with /"), {
      status: 400,
    });
  }

  let upstreamUrl;
  try {
    upstreamUrl = new URL(path, host.endsWith("/") ? host : `${host}/`);
  } catch {
    throw Object.assign(new Error("Envelope host is not a valid URL"), {
      status: 400,
    });
  }

  if (upstreamUrl.protocol !== "https:" && upstreamUrl.protocol !== "http:") {
    throw Object.assign(new Error("Envelope host must be http(s)"), {
      status: 400,
    });
  }
  if (
    upstreamUrl.protocol !== "https:" &&
    !LOOPBACK_HOSTS.has(upstreamUrl.hostname.toLowerCase())
  ) {
    throw Object.assign(new Error("Non-loopback upstream hosts require https"), {
      status: 400,
    });
  }
  if (!isAllowedCloudProxyHost(upstreamUrl.hostname)) {
    throw Object.assign(
      new Error(`Upstream host is not allowlisted: ${upstreamUrl.hostname}`),
      { status: 403 },
    );
  }

  const headers =
    envelope.headers &&
    typeof envelope.headers === "object" &&
    !Array.isArray(envelope.headers)
      ? Object.fromEntries(
          Object.entries(envelope.headers).filter(
            ([key, value]) =>
              typeof key === "string" &&
              typeof value === "string" &&
              key.trim() !== "",
          ),
        )
      : {};

  const timeoutSecondsRaw = envelope.timeout_seconds ?? envelope.timeoutSeconds;
  const timeoutSeconds = Number(timeoutSecondsRaw);
  const timeoutMs =
    Number.isFinite(timeoutSeconds) && timeoutSeconds > 0
      ? Math.min(timeoutSeconds, MAX_TIMEOUT_SECONDS) * 1000
      : DEFAULT_TIMEOUT_SECONDS * 1000;

  return {
    upstreamUrl,
    method,
    headers,
    body: "body" in envelope ? envelope.body : undefined,
    timeoutMs,
  };
}

/**
 * @param {URL} upstreamUrl
 * @param {{
 *   method: string,
 *   headers: Record<string, string>,
 *   body?: string,
 *   timeoutMs: number,
 * }} options
 * @returns {Promise<{ statusCode: number, headers: Record<string, string>, body: Buffer }>}
 */
function requestUpstream(upstreamUrl, options) {
  const transport =
    upstreamUrl.protocol === "https:" ? httpsRequest : httpRequest;
  const headers = { ...options.headers };
  if (options.body !== undefined) {
    headers["Content-Length"] = String(Buffer.byteLength(options.body));
  }

  return new Promise((resolve, reject) => {
    const req = transport(
      {
        protocol: upstreamUrl.protocol,
        hostname: upstreamUrl.hostname,
        port: upstreamUrl.port || undefined,
        path: `${upstreamUrl.pathname}${upstreamUrl.search}`,
        method: options.method,
        headers,
      },
      (upstreamRes) => {
        const chunks = [];
        upstreamRes.on("data", (chunk) => chunks.push(chunk));
        upstreamRes.on("end", () => {
          const responseHeaders = {};
          const contentType = upstreamRes.headers["content-type"];
          if (typeof contentType === "string") {
            responseHeaders["Content-Type"] = contentType;
          }
          resolve({
            statusCode: upstreamRes.statusCode ?? 502,
            headers: responseHeaders,
            body: Buffer.concat(chunks),
          });
        });
      },
    );

    req.setTimeout(options.timeoutMs, () => {
      req.destroy(
        Object.assign(new Error(`Upstream timeout after ${options.timeoutMs}ms`), {
          status: 504,
        }),
      );
    });
    req.on("error", reject);

    if (options.body !== undefined) {
      req.end(options.body);
    } else {
      req.end();
    }
  });
}

/**
 * @param {import("node:http").IncomingMessage} req
 * @param {import("node:http").ServerResponse} res
 * @returns {Promise<boolean>} true when the request was handled
 */
export async function handleCloudProxy(req, res) {
  if (!isCloudProxyRequest(req)) {
    return false;
  }

  try {
    const envelope = await readJsonBody(req);
    const { upstreamUrl, method, headers, body, timeoutMs } =
      parseCloudProxyEnvelope(envelope);

    const outboundHeaders = { ...headers };
    let outboundBody;
    if (body !== undefined && body !== null && method !== "GET" && method !== "HEAD") {
      if (!outboundHeaders["Content-Type"] && !outboundHeaders["content-type"]) {
        outboundHeaders["Content-Type"] = "application/json";
      }
      outboundBody = typeof body === "string" ? body : JSON.stringify(body);
    }

    const upstream = await requestUpstream(upstreamUrl, {
      method,
      headers: outboundHeaders,
      body: outboundBody,
      timeoutMs,
    });
    res.writeHead(upstream.statusCode, upstream.headers);
    res.end(upstream.body);
  } catch (error) {
    if (res.headersSent) {
      res.destroy();
      return true;
    }

    const status =
      typeof error?.status === "number"
        ? error.status
        : error?.code === "ABORT_ERR" || error?.code === "ECONNRESET"
          ? 504
          : 502;
    const message =
      error instanceof Error ? error.message : "Cloud proxy request failed";
    res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ detail: message }));
  }

  return true;
}
