/**
 * MVP stub for mobile APK artifacts when Engagement Manager (190/191)
 * routes are not yet available on the local stack.
 *
 * Mounted at `/api/pentest/engagements` on ingress / static-server.
 * - POST   /api/pentest/engagements/:id/mobile/apk
 * - GET    /api/pentest/engagements/:id/mobile/artifacts
 * - POST   /api/pentest/engagements/:id/mobile/artifacts/:aid/install
 *
 * Auth: X-Session-API-Key validated against agent-server /server_info.
 * In-memory store only — not for production EngMgr.
 */

import { randomUUID } from "node:crypto";
import { readSessionApiKey } from "./appwrite-proxy.mjs";

export const MOBILE_ARTIFACTS_PATH_PREFIX = "/api/pentest/engagements";
export const MAX_APK_BYTES = 200 * 1024 * 1024;

/** @type {Map<string, Array<Record<string, unknown>>>} */
const artifactsByEngagement = new Map();

/**
 * @param {string} url
 */
export function isMobileArtifactsProxyRequest(url) {
  const path = (url ?? "/").split("?")[0];
  return (
    path === MOBILE_ARTIFACTS_PATH_PREFIX ||
    path.startsWith(`${MOBILE_ARTIFACTS_PATH_PREFIX}/`)
  );
}

/**
 * @param {import('node:http').ServerResponse} res
 * @param {number} status
 * @param {unknown} body
 */
function writeJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

/**
 * @param {import('node:http').IncomingMessage} req
 * @returns {Promise<Buffer>}
 */
function readBody(req) {
  return new Promise((resolve, reject) => {
    /** @type {Buffer[]} */
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_APK_BYTES + 1024 * 1024) {
        reject(Object.assign(new Error("Payload too large"), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

/**
 * Minimal multipart parser for a single `file` field.
 * @param {Buffer} body
 * @param {string} contentType
 * @returns {{ filename: string, buffer: Buffer } | null}
 */
export function parseMultipartFile(body, contentType) {
  const match = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType ?? "");
  if (!match) return null;
  const boundary = match[1] || match[2];
  const parts = body.toString("binary").split(`--${boundary}`);
  for (const part of parts) {
    if (!part.includes("Content-Disposition") || !part.includes('name="file"')) {
      continue;
    }
    const headerEnd = part.indexOf("\r\n\r\n");
    if (headerEnd < 0) continue;
    const headers = part.slice(0, headerEnd);
    const filenameMatch = /filename="([^"]+)"/i.exec(headers);
    const filename = filenameMatch?.[1] ?? "upload.apk";
    let content = part.slice(headerEnd + 4);
    if (content.endsWith("\r\n")) {
      content = content.slice(0, -2);
    }
    if (content.endsWith("--")) {
      content = content.slice(0, -2);
    }
    return { filename, buffer: Buffer.from(content, "binary") };
  }
  return null;
}

/**
 * @param {string} filename
 */
export function basenameOnly(filename) {
  return String(filename ?? "")
    .replace(/\\/g, "/")
    .split("/")
    .pop()
    ?.trim() || "upload.apk";
}

/**
 * @param {{
 *   agentServerUrl: string,
 *   fetchImpl?: typeof fetch,
 * }} options
 */
export function createMobileArtifactsProxyHandler(options) {
  const agentServerUrl = options.agentServerUrl;
  const fetchImpl = options.fetchImpl ?? fetch;

  /**
   * @param {string} sessionApiKey
   */
  async function validateSession(sessionApiKey) {
    const base = agentServerUrl.replace(/\/+$/, "");
    const response = await fetchImpl(`${base}/server_info`, {
      method: "GET",
      headers: {
        Accept: "application/json",
        "X-Session-API-Key": sessionApiKey,
      },
    });
    if (response.status === 401) {
      const err = new Error("Invalid X-Session-API-Key");
      // @ts-expect-error status
      err.status = 401;
      throw err;
    }
    if (!response.ok) {
      const err = new Error(
        `Failed to validate session (${response.status})`,
      );
      // @ts-expect-error status
      err.status = 502;
      throw err;
    }
  }

  /**
   * @param {import('node:http').IncomingMessage} req
   * @param {import('node:http').ServerResponse} res
   */
  async function handleHttp(req, res) {
    const rawUrl = req.url ?? "/";
    if (!isMobileArtifactsProxyRequest(rawUrl)) {
      writeJson(res, 404, { detail: "Not found" });
      return;
    }

    const sessionApiKey = readSessionApiKey(req);
    if (!sessionApiKey) {
      writeJson(res, 401, { detail: "Missing X-Session-API-Key" });
      return;
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
      });
      return;
    }

    const url = new URL(rawUrl, "http://localhost");
    const segments = url.pathname
      .replace(MOBILE_ARTIFACTS_PATH_PREFIX, "")
      .split("/")
      .filter(Boolean);
    // /:engagementId/mobile/apk
    // /:engagementId/mobile/artifacts
    // /:engagementId/mobile/artifacts/:aid/install
    if (segments.length < 3 || segments[1] !== "mobile") {
      writeJson(res, 404, { detail: "Not found" });
      return;
    }

    const engagementId = decodeURIComponent(segments[0]);
    const list = artifactsByEngagement.get(engagementId) ?? [];

    if (
      segments.length === 3 &&
      segments[2] === "apk" &&
      req.method === "POST"
    ) {
      let body;
      try {
        body = await readBody(req);
      } catch (err) {
        const status =
          err && typeof err === "object" && "status" in err
            ? Number(err.status) || 400
            : 400;
        writeJson(res, status, {
          detail: err instanceof Error ? err.message : "Bad request",
        });
        return;
      }
      const contentType = String(req.headers["content-type"] ?? "");
      const parsed = parseMultipartFile(body, contentType);
      if (!parsed) {
        writeJson(res, 400, { detail: "Expected multipart file field `file`" });
        return;
      }
      const name = basenameOnly(parsed.filename);
      if (!/\.apk$/i.test(name)) {
        writeJson(res, 400, { detail: "Only .apk uploads are accepted" });
        return;
      }
      if (parsed.buffer.length > MAX_APK_BYTES) {
        writeJson(res, 413, { detail: "APK exceeds maximum size" });
        return;
      }
      const artifact = {
        artifact_id: randomUUID(),
        filename: name,
        path: `mobile/${engagementId}/${name}`,
        scan_status: "queued",
        mobsf_scan_id: randomUUID(),
        size_bytes: parsed.buffer.length,
        created_at: new Date().toISOString(),
      };
      // Simulate scan progressing after a short delay for UI polling.
      setTimeout(() => {
        artifact.scan_status = "scanning";
      }, 50);
      setTimeout(() => {
        artifact.scan_status = "ready";
      }, 200);
      list.unshift(artifact);
      artifactsByEngagement.set(engagementId, list);
      writeJson(res, 200, {
        artifact_id: artifact.artifact_id,
        path: artifact.path,
        filename: artifact.filename,
        mobsf_scan_id: artifact.mobsf_scan_id,
        scan_status: artifact.scan_status,
      });
      return;
    }

    if (
      segments.length === 3 &&
      segments[2] === "artifacts" &&
      req.method === "GET"
    ) {
      writeJson(res, 200, {
        items: list.map((item) => ({
          artifact_id: item.artifact_id,
          filename: item.filename,
          scan_status: item.scan_status,
          size_bytes: item.size_bytes,
          created_at: item.created_at,
        })),
      });
      return;
    }

    if (
      segments.length === 5 &&
      segments[2] === "artifacts" &&
      segments[4] === "install" &&
      req.method === "POST"
    ) {
      const aid = decodeURIComponent(segments[3]);
      const artifact = list.find((item) => item.artifact_id === aid);
      if (!artifact) {
        writeJson(res, 404, { detail: "Artifact not found" });
        return;
      }
      writeJson(res, 200, {
        ok: true,
        artifact_id: artifact.artifact_id,
        status: "installed",
      });
      return;
    }

    writeJson(res, 404, { detail: "Not found" });
  }

  return handleHttp;
}

/** Test helper — clear in-memory store. */
export function resetMobileArtifactsStore() {
  artifactsByEngagement.clear();
}
