/**
 * Plane integration reverse proxy for Agent Canvas.
 *
 * Mounted at `/api/integrations/plane` on ingress / static-server.
 * Resolves base URL / workspace slug / project / API key from the
 * agent-server settings + secrets store using the caller's session key,
 * then forwards to Plane. The API key never leaves the proxy process.
 *
 * `GET /api/integrations/plane/test` verifies the stored project (and
 * optional module) without requiring the browser to know Plane path details.
 */

import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { URL } from "node:url";

export const PLANE_PROXY_PATH_PREFIX = "/api/integrations/plane";
export const PLANE_WORKSPACE_ID_HEADER = "x-openhands-workspace-id";
export const CONFIG_CACHE_TTL_MS = 30_000;

/**
 * @param {string} workspaceId
 */
export function defaultPlaneSecretName(workspaceId) {
  const cleaned = String(workspaceId ?? "")
    .trim()
    .replace(/[^A-Za-z0-9_]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return `INTEGRATION_PLANE_API_KEY_${cleaned || "workspace"}`;
}

/**
 * @param {string} url
 */
export function isPlaneProxyRequest(url) {
  const path = (url ?? "/").split("?")[0];
  return (
    path === PLANE_PROXY_PATH_PREFIX ||
    path.startsWith(`${PLANE_PROXY_PATH_PREFIX}/`)
  );
}

/**
 * Strip the proxy prefix; remaining path is forwarded under the Plane base URL.
 *
 * @param {string} urlPath pathname only (no query)
 * @returns {string}
 */
export function rewritePlaneProxyPath(urlPath) {
  const path = (urlPath ?? "/").split("?")[0];
  if (path === PLANE_PROXY_PATH_PREFIX) {
    return "";
  }
  const prefix = `${PLANE_PROXY_PATH_PREFIX}/`;
  if (!path.startsWith(prefix)) {
    throw new Error(`Not a Plane proxy path: ${path}`);
  }
  const rest = path.slice(prefix.length);
  return rest ? `/${rest.replace(/^\/+/, "")}` : "";
}

/**
 * Join Plane base URL with an API-relative path.
 *
 * @param {string} baseUrl
 * @param {string} relativePath
 */
export function buildPlaneTargetUrl(baseUrl, relativePath) {
  const base = String(baseUrl ?? "").trim().replace(/\/+$/, "");
  if (!base) {
    throw new Error("Plane base URL is empty");
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
  const header = req.headers[PLANE_WORKSPACE_ID_HEADER];
  if (typeof header === "string" && header.trim()) {
    return header.trim();
  }
  if (Array.isArray(header) && header[0]?.trim()) {
    return header[0].trim();
  }
  return null;
}

/**
 * @param {string} baseUrl
 * @param {string} workspaceSlug
 * @param {string} projectId
 * @param {string} [moduleId]
 */
export function buildPlaneTestPath(
  baseUrl,
  workspaceSlug,
  projectId,
  moduleId,
) {
  void baseUrl;
  const slug = encodeURIComponent(workspaceSlug);
  const project = encodeURIComponent(projectId);
  const module = String(moduleId ?? "").trim();
  if (module) {
    return `/api/v1/workspaces/${slug}/projects/${project}/modules/${encodeURIComponent(module)}/`;
  }
  return `/api/v1/workspaces/${slug}/projects/${project}/`;
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
  const target = new URL(
    path,
    agentServerUrl.endsWith("/") ? agentServerUrl : `${agentServerUrl}/`,
  );
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
          Accept:
            opts.responseType === "text" ? "text/plain" : "application/json",
          "X-Session-API-Key": sessionApiKey,
        },
      },
      (res) => {
        /** @type {Buffer[]} */
        const chunks = [];
        res.on("data", (c) =>
          chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c)),
        );
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
 *   workspaceSlug: string,
 *   projectId: string,
 *   moduleId: string,
 *   apiKey: string,
 *   secretName: string,
 * }} ResolvedPlaneConfig
 */

/**
 * @param {{
 *   agentServerUrl: string,
 *   cacheTtlMs?: number,
 *   fetchImpl?: typeof agentServerFetch,
 * }} options
 */
export function createPlaneProxyHandler(options) {
  const agentServerUrl = options.agentServerUrl;
  const cacheTtlMs = options.cacheTtlMs ?? CONFIG_CACHE_TTL_MS;
  const fetchImpl = options.fetchImpl ?? agentServerFetch;

  /** @type {Map<string, { expiresAt: number, config: ResolvedPlaneConfig }>} */
  const cache = new Map();

  /**
   * @param {string} sessionApiKey
   * @param {string} workspaceId
   * @returns {Promise<ResolvedPlaneConfig>}
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
      settings?.misc_settings?.integrations?.plane?.byWorkspace ?? {};
    const plane = byWorkspace[workspaceId] ?? {};
    const enabled = Boolean(plane.enabled);
    const baseUrl = String(plane.baseUrl ?? "").trim();
    const workspaceSlug = String(plane.workspaceSlug ?? "").trim();
    const projectId = String(plane.projectId ?? "").trim();
    const moduleId = String(plane.moduleId ?? "").trim();
    const secretName =
      String(plane.apiKeySecretName ?? "").trim() ||
      defaultPlaneSecretName(workspaceId);

    if (!enabled) {
      const err = new Error(
        `Plane integration is disabled for workspace '${workspaceId}'`,
      );
      // @ts-expect-error attach status
      err.status = 503;
      throw err;
    }
    if (!baseUrl || !workspaceSlug || !projectId) {
      const err = new Error(
        `Plane integration for workspace '${workspaceId}' is missing baseUrl, workspaceSlug, or projectId`,
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
          ? `Plane API key secret '${secretName}' is not configured`
          : `Failed to load Plane API key (${secretRes.status})`,
      );
      // @ts-expect-error attach status
      err.status = secretRes.status === 404 ? 400 : 502;
      throw err;
    }
    const apiKey = String(secretRes.text ?? "").trim();
    if (!apiKey) {
      const err = new Error("Plane API key secret is empty");
      // @ts-expect-error attach status
      err.status = 400;
      throw err;
    }

    const config = {
      enabled,
      baseUrl,
      workspaceSlug,
      projectId,
      moduleId,
      apiKey,
      secretName,
    };
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
  return async function handlePlaneProxy(req, res) {
    const rawUrl = req.url ?? "/";
    if (!isPlaneProxyRequest(rawUrl)) {
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
      relativePath = rewritePlaneProxyPath(url.pathname);
    } catch (err) {
      writeJson(res, 400, {
        detail: err instanceof Error ? err.message : "Bad path",
      });
      return;
    }

    const isTestRequest = relativePath === "/test" || relativePath === "";
    if (isTestRequest) {
      relativePath = buildPlaneTestPath(
        config.baseUrl,
        config.workspaceSlug,
        config.projectId,
        config.moduleId,
      );
    }

    const targetUrl = buildPlaneTargetUrl(config.baseUrl, relativePath);
    const target = new URL(targetUrl);
    if (url.search && !isTestRequest) {
      target.search = url.search;
    }

    const method = isTestRequest
      ? "GET"
      : (req.method ?? "GET").toUpperCase();
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
            "X-API-Key": config.apiKey,
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
            detail: `Plane upstream error: ${err.message}`,
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
