/**
 * Dependency-Track integration reverse proxy for Agent Canvas.
 *
 * Mounted at `/api/integrations/dependency-track` on ingress / static-server.
 * Resolves base URL / project UUID / API key from agent-server settings +
 * secrets for the workspace in `X-OpenHands-Workspace-Id`.
 */

import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { URL } from "node:url";
import {
  APPWRITE_WORKSPACE_ID_HEADER,
  readSessionApiKey,
} from "./appwrite-proxy.mjs";

export const DEPENDENCY_TRACK_PROXY_PATH_PREFIX =
  "/api/integrations/dependency-track";
export const CONFIG_CACHE_TTL_MS = 30_000;

/**
 * @param {string} workspaceId
 */
export function defaultDependencyTrackSecretName(workspaceId) {
  const cleaned = String(workspaceId ?? "")
    .trim()
    .replace(/[^A-Za-z0-9_]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return `INTEGRATION_DEPENDENCY_TRACK_API_KEY_${cleaned || "workspace"}`;
}

/**
 * @param {string} url
 */
export function isDependencyTrackProxyRequest(url) {
  const path = (url ?? "/").split("?")[0];
  return (
    path === DEPENDENCY_TRACK_PROXY_PATH_PREFIX ||
    path.startsWith(`${DEPENDENCY_TRACK_PROXY_PATH_PREFIX}/`)
  );
}

/**
 * @param {string} urlPath pathname only (no query)
 * @returns {string} path to append to the Dependency-Track base URL
 */
export function rewriteDependencyTrackProxyPath(urlPath) {
  const path = (urlPath ?? "/").split("?")[0];
  if (path === DEPENDENCY_TRACK_PROXY_PATH_PREFIX) {
    return "/api/v1/version";
  }
  const prefix = `${DEPENDENCY_TRACK_PROXY_PATH_PREFIX}/`;
  if (!path.startsWith(prefix)) {
    throw new Error(`Not a Dependency-Track proxy path: ${path}`);
  }
  const rest = path.slice(prefix.length);
  return rest.startsWith("/") ? rest : `/${rest}`;
}

/**
 * @param {string} baseUrl
 * @param {string} relativePath
 */
export function buildDependencyTrackTargetUrl(baseUrl, relativePath) {
  const base = String(baseUrl ?? "").trim().replace(/\/+$/, "");
  if (!base) {
    throw new Error("Dependency-Track base URL is empty");
  }
  let rel = String(relativePath ?? "").trim();
  if (!rel || rel === "/") {
    return base;
  }
  if (!rel.startsWith("/")) {
    rel = `/${rel}`;
  }
  return `${base}${rel}`;
}

/**
 * @param {import('node:http').IncomingMessage} req
 * @returns {string | null}
 */
export function readWorkspaceId(req) {
  const header = req.headers[APPWRITE_WORKSPACE_ID_HEADER.toLowerCase()];
  if (typeof header === "string" && header.trim()) {
    return header.trim();
  }
  if (Array.isArray(header) && header[0]?.trim()) {
    return header[0].trim();
  }
  return null;
}

/**
 * @param {import('node:http').IncomingMessage} req
 * @returns {Promise<Buffer>}
 */
function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    /** @type {Buffer[]} */
    const chunks = [];
    req.on("data", (chunk) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

/**
 * @param {import('node:http').ServerResponse} res
 * @param {number} status
 * @param {unknown} body
 */
function writeJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

/**
 * @param {string} agentServerUrl
 * @param {string} sessionApiKey
 * @param {string} path
 * @param {{ responseType?: 'json' | 'text' }} [opts]
 */
async function agentServerFetch(agentServerUrl, sessionApiKey, path, opts = {}) {
  const target = new URL(path, agentServerUrl.endsWith("/")
    ? agentServerUrl
    : `${agentServerUrl}/`);
  const isHttps = target.protocol === "https:";
  const transport = isHttps ? httpsRequest : httpRequest;

  return new Promise((resolve, reject) => {
    const req = transport(
      {
        protocol: target.protocol,
        hostname: target.hostname,
        port: target.port || (isHttps ? 443 : 80),
        path: `${target.pathname}${target.search}`,
        method: "GET",
        headers: {
          Accept: opts.responseType === "text" ? "text/plain" : "application/json",
          "X-Session-API-Key": sessionApiKey,
        },
      },
      (res) => {
        /** @type {Buffer[]} */
        const chunks = [];
        res.on("data", (c) => chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c)));
        res.on("end", () => {
          const buf = Buffer.concat(chunks);
          const text = buf.toString("utf8");
          resolve({
            status: res.statusCode ?? 500,
            text,
            json: () => {
              try {
                return JSON.parse(text);
              } catch {
                return null;
              }
            },
          });
        });
      },
    );
    req.on("error", reject);
    req.end();
  });
}

/**
 * @typedef {{
 *   enabled: boolean,
 *   baseUrl: string,
 *   projectUuid: string,
 *   apiKey: string,
 *   secretName: string,
 * }} ResolvedDependencyTrackConfig
 */

/**
 * @param {{
 *   agentServerUrl: string,
 *   cacheTtlMs?: number,
 *   fetchImpl?: typeof agentServerFetch,
 * }} options
 */
export function createDependencyTrackProxyHandler(options) {
  const agentServerUrl = options.agentServerUrl;
  const cacheTtlMs = options.cacheTtlMs ?? CONFIG_CACHE_TTL_MS;
  const fetchImpl = options.fetchImpl ?? agentServerFetch;

  /** @type {Map<string, { expiresAt: number, config: ResolvedDependencyTrackConfig }>} */
  const cache = new Map();

  /**
   * @param {string} sessionApiKey
   * @param {string} workspaceId
   * @returns {Promise<ResolvedDependencyTrackConfig>}
   */
  async function resolveConfig(sessionApiKey, workspaceId) {
    const cacheKey = `${sessionApiKey}::${workspaceId}`;
    const cached = cache.get(cacheKey);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.config;
    }

    const settingsRes = await fetchImpl(
      agentServerUrl,
      sessionApiKey,
      "/api/settings",
    );
    if (settingsRes.status !== 200) {
      const err = new Error(
        `Failed to load settings from agent-server (${settingsRes.status})`,
      );
      // @ts-expect-error attach status
      err.status = settingsRes.status === 401 ? 401 : 502;
      throw err;
    }
    const settings = settingsRes.json();
    const byWorkspace =
      settings?.misc_settings?.integrations?.dependencyTrack?.byWorkspace ??
      {};
    const dt = byWorkspace[workspaceId] ?? {};
    const enabled = Boolean(dt.enabled);
    const baseUrl = String(dt.baseUrl ?? "").trim();
    const projectUuid = String(dt.projectUuid ?? "").trim();
    const secretName =
      String(dt.apiKeySecretName ?? "").trim() ||
      defaultDependencyTrackSecretName(workspaceId);

    if (!enabled) {
      const err = new Error(
        `Dependency-Track integration is disabled for workspace '${workspaceId}'`,
      );
      // @ts-expect-error attach status
      err.status = 503;
      throw err;
    }
    if (!baseUrl || !projectUuid) {
      const err = new Error(
        `Dependency-Track integration for workspace '${workspaceId}' is missing baseUrl or projectUuid`,
      );
      // @ts-expect-error attach status
      err.status = 400;
      throw err;
    }

    const secretRes = await fetchImpl(
      agentServerUrl,
      sessionApiKey,
      `/api/settings/secrets/${encodeURIComponent(secretName)}`,
      { responseType: "text" },
    );
    if (secretRes.status !== 200) {
      const err = new Error(
        secretRes.status === 404
          ? `Dependency-Track API key secret '${secretName}' is not configured`
          : `Failed to load Dependency-Track API key (${secretRes.status})`,
      );
      // @ts-expect-error attach status
      err.status = secretRes.status === 404 ? 400 : 502;
      throw err;
    }
    const apiKey = String(secretRes.text ?? "").trim();
    if (!apiKey) {
      const err = new Error("Dependency-Track API key secret is empty");
      // @ts-expect-error attach status
      err.status = 400;
      throw err;
    }

    const config = { enabled, baseUrl, projectUuid, apiKey, secretName };
    cache.set(cacheKey, {
      expiresAt: Date.now() + cacheTtlMs,
      config,
    });
    return config;
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:http').ServerResponse} res
   */
  return async function handleDependencyTrackProxy(req, res) {
    const rawUrl = req.url ?? "/";
    if (!isDependencyTrackProxyRequest(rawUrl)) {
      writeJson(res, 404, { detail: "Not found" });
      return;
    }

    const sessionApiKey = readSessionApiKey(req);
    if (!sessionApiKey) {
      writeJson(res, 401, { detail: "Missing X-Session-API-Key" });
      return;
    }

    const workspaceId = readWorkspaceId(req);
    if (!workspaceId) {
      writeJson(res, 400, {
        detail: "Missing X-OpenHands-Workspace-Id",
      });
      return;
    }

    let config;
    try {
      config = await resolveConfig(sessionApiKey, workspaceId);
    } catch (err) {
      const status =
        err && typeof err === "object" && "status" in err
          ? Number(err.status) || 502
          : 502;
      writeJson(res, status, {
        detail: err instanceof Error ? err.message : "Proxy configuration error",
      });
      return;
    }

    const url = new URL(rawUrl, "http://localhost");
    let relativePath;
    try {
      relativePath = rewriteDependencyTrackProxyPath(url.pathname);
    } catch (err) {
      writeJson(res, 400, {
        detail: err instanceof Error ? err.message : "Bad path",
      });
      return;
    }

    const targetUrl = buildDependencyTrackTargetUrl(config.baseUrl, relativePath);
    const target = new URL(targetUrl);
    if (url.search) {
      target.search = url.search;
    }

    const method = (req.method ?? "GET").toUpperCase();
    const body =
      method === "GET" || method === "HEAD" ? null : await readRequestBody(req);

    const isHttps = target.protocol === "https:";
    const transport = isHttps ? httpsRequest : httpRequest;

    await new Promise((resolve) => {
      const upstream = transport(
        {
          protocol: target.protocol,
          hostname: target.hostname,
          port: target.port || (isHttps ? 443 : 80),
          path: `${target.pathname}${target.search}`,
          method,
          headers: {
            Accept: req.headers.accept ?? "application/json",
            "Content-Type":
              req.headers["content-type"] ?? "application/json",
            "X-Api-Key": config.apiKey,
            ...(body && body.length
              ? { "Content-Length": body.length }
              : {}),
          },
        },
        (upstreamRes) => {
          const hopByHop = new Set([
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
          ]);
          /** @type {Record<string, string | string[] | undefined>} */
          const headers = {};
          for (const [key, value] of Object.entries(upstreamRes.headers)) {
            if (hopByHop.has(key.toLowerCase())) continue;
            headers[key] = value;
          }
          res.writeHead(upstreamRes.statusCode ?? 502, headers);
          upstreamRes.pipe(res);
          upstreamRes.on("end", resolve);
          upstreamRes.on("error", resolve);
        },
      );
      upstream.on("error", (err) => {
        if (!res.headersSent) {
          writeJson(res, 502, {
            detail: `Dependency-Track upstream error: ${err.message}`,
          });
        }
        resolve();
      });
      if (body && body.length) {
        upstream.write(body);
      }
      upstream.end();
    });
  };
}
