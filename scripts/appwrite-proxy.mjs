/**
 * AppWrite integration reverse proxy for Agent Canvas.
 *
 * Mounted at `/api/integrations/appwrite` on ingress / static-server.
 * Resolves endpoint/project/API key from the agent-server settings + secrets
 * store using the caller's session key, then forwards to AppWrite. The API
 * key never leaves the proxy process.
 */

import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { URL } from "node:url";

export const APPWRITE_PROXY_PATH_PREFIX = "/api/integrations/appwrite";
export const APPWRITE_WORKSPACE_ID_HEADER = "x-openhands-workspace-id";
export const CONFIG_CACHE_TTL_MS = 30_000;

/**
 * @param {string} workspaceId
 */
export function defaultAppwriteSecretName(workspaceId) {
  const cleaned = String(workspaceId ?? "")
    .trim()
    .replace(/[^A-Za-z0-9_]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return `INTEGRATION_APPWRITE_API_KEY_${cleaned || "workspace"}`;
}

/**
 * @param {string} url
 */
export function isAppwriteProxyRequest(url) {
  const path = (url ?? "/").split("?")[0];
  return (
    path === APPWRITE_PROXY_PATH_PREFIX ||
    path.startsWith(`${APPWRITE_PROXY_PATH_PREFIX}/`)
  );
}

/**
 * Strip the proxy prefix and ensure the remaining path starts with `/v1`
 * or is empty (health ping).
 *
 * @param {string} urlPath pathname only (no query)
 * @returns {string} path to append to the AppWrite endpoint
 */
export function rewriteAppwriteProxyPath(urlPath) {
  const path = (urlPath ?? "/").split("?")[0];
  if (path === APPWRITE_PROXY_PATH_PREFIX) {
    return "";
  }
  const prefix = `${APPWRITE_PROXY_PATH_PREFIX}/`;
  if (!path.startsWith(prefix)) {
    throw new Error(`Not an AppWrite proxy path: ${path}`);
  }
  const rest = path.slice(prefix.length);
  return rest ? `/${rest.replace(/^\/+/, "")}` : "";
}

/**
 * Join AppWrite endpoint (may already end with `/v1`) with a relative path
 * that typically also starts with `/v1/...`.
 *
 * @param {string} endpoint
 * @param {string} relativePath
 */
export function buildAppwriteTargetUrl(endpoint, relativePath) {
  const base = String(endpoint ?? "").trim().replace(/\/+$/, "");
  if (!base) {
    throw new Error("AppWrite endpoint is empty");
  }
  let rel = String(relativePath ?? "").trim();
  if (!rel || rel === "/") {
    return base;
  }
  if (!rel.startsWith("/")) {
    rel = `/${rel}`;
  }
  // Avoid `/v1/v1/...` when both endpoint and path include the version prefix.
  if (base.endsWith("/v1") && (rel === "/v1" || rel.startsWith("/v1/"))) {
    rel = rel.slice("/v1".length) || "";
  }
  return `${base}${rel}`;
}

/**
 * @param {import('node:http').IncomingMessage} req
 * @returns {string | null}
 */
export function readSessionApiKey(req) {
  const header = req.headers["x-session-api-key"];
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
 * @returns {string | null}
 */
export function readWorkspaceId(req) {
  const header = req.headers[APPWRITE_WORKSPACE_ID_HEADER];
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
 *   endpoint: string,
 *   projectId: string,
 *   apiKey: string,
 *   secretName: string,
 * }} ResolvedAppwriteConfig
 */

/**
 * @param {{
 *   agentServerUrl: string,
 *   cacheTtlMs?: number,
 *   fetchImpl?: typeof agentServerFetch,
 * }} options
 */
export function createAppwriteProxyHandler(options) {
  const agentServerUrl = options.agentServerUrl;
  const cacheTtlMs = options.cacheTtlMs ?? CONFIG_CACHE_TTL_MS;
  const fetchImpl = options.fetchImpl ?? agentServerFetch;

  /** @type {Map<string, { expiresAt: number, config: ResolvedAppwriteConfig }>} */
  const cache = new Map();

  /**
   * @param {string} sessionApiKey
   * @param {string} workspaceId
   * @returns {Promise<ResolvedAppwriteConfig>}
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
      settings?.misc_settings?.integrations?.appwrite?.byWorkspace ?? {};
    const appwrite = byWorkspace[workspaceId] ?? {};
    const enabled = Boolean(appwrite.enabled);
    const endpoint = String(appwrite.endpoint ?? "").trim();
    const projectId = String(appwrite.projectId ?? "").trim();
    const secretName =
      String(appwrite.apiKeySecretName ?? "").trim() ||
      defaultAppwriteSecretName(workspaceId);

    if (!enabled) {
      const err = new Error(
        `AppWrite integration is disabled for workspace '${workspaceId}'`,
      );
      // @ts-expect-error attach status
      err.status = 503;
      throw err;
    }
    if (!endpoint || !projectId) {
      const err = new Error(
        `AppWrite integration for workspace '${workspaceId}' is missing endpoint or projectId`,
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
          ? `AppWrite API key secret '${secretName}' is not configured`
          : `Failed to load AppWrite API key (${secretRes.status})`,
      );
      // @ts-expect-error attach status
      err.status = secretRes.status === 404 ? 400 : 502;
      throw err;
    }
    const apiKey = String(secretRes.text ?? "").trim();
    if (!apiKey) {
      const err = new Error("AppWrite API key secret is empty");
      // @ts-expect-error attach status
      err.status = 400;
      throw err;
    }

    const config = { enabled, endpoint, projectId, apiKey, secretName };
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
  return async function handleAppwriteProxy(req, res) {
    const rawUrl = req.url ?? "/";
    if (!isAppwriteProxyRequest(rawUrl)) {
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
      relativePath = rewriteAppwriteProxyPath(url.pathname);
    } catch (err) {
      writeJson(res, 400, {
        detail: err instanceof Error ? err.message : "Bad path",
      });
      return;
    }

    const targetUrl = buildAppwriteTargetUrl(config.endpoint, relativePath);
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
            "X-Appwrite-Project": config.projectId,
            "X-Appwrite-Key": config.apiKey,
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
            detail: `AppWrite upstream error: ${err.message}`,
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
