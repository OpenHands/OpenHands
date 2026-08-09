/**
 * Lightweight text-translation proxy for Agent Canvas.
 *
 * Mounted at `/api/integrations/translate` on ingress / static-server.
 * Used to localize dynamic security-finding prose (OpenGrep / Dependency-Track)
 * into the user's UI language — primarily Brazilian Portuguese.
 *
 * Backend: MyMemory free API (no key). Optional override via
 * ``TRANSLATE_API_URL`` (must accept GET ``?q=&langpair=`` and return
 * ``{ responseData: { translatedText } }`` MyMemory-compatible JSON).
 */

import { request as httpsRequest } from "node:https";
import { request as httpRequest } from "node:http";
import { URL } from "node:url";

export const TRANSLATE_PROXY_PATH = "/api/integrations/translate";

const DEFAULT_MYMEMORY_URL = "https://api.mymemory.translated.net/get";
const MAX_TEXTS_PER_REQUEST = 40;
const MAX_CHARS_PER_TEXT = 1500;
const MYMEMORY_CHUNK_LIMIT = 450;

/**
 * @param {string} url
 */
export function isTranslateProxyRequest(url) {
  const path = (url ?? "/").split("?")[0];
  return path === TRANSLATE_PROXY_PATH;
}

/**
 * Map Canvas/i18n language codes onto MyMemory ``langpair`` targets.
 *
 * @param {string} language
 * @returns {string | null} null when translation should be skipped
 */
export function resolveTranslateTarget(language) {
  const raw = String(language ?? "").trim();
  if (!raw) return null;
  const base = raw.split("-")[0].toLowerCase();
  if (base === "en") return null;
  if (base === "pt") return "pt-BR";
  return base;
}

/**
 * @param {string} text
 * @param {number} maxLen
 * @returns {string[]}
 */
export function chunkTextForTranslation(text, maxLen = MYMEMORY_CHUNK_LIMIT) {
  const value = String(text ?? "");
  if (value.length <= maxLen) return value ? [value] : [];
  const chunks = [];
  let remaining = value;
  while (remaining.length > maxLen) {
    let cut = remaining.lastIndexOf(" ", maxLen);
    if (cut < Math.floor(maxLen * 0.5)) cut = maxLen;
    chunks.push(remaining.slice(0, cut).trimEnd());
    remaining = remaining.slice(cut).trimStart();
  }
  if (remaining) chunks.push(remaining);
  return chunks;
}

/**
 * @param {unknown} body
 * @returns {{ texts: string[], source: string, target: string }}
 */
export function parseTranslateRequestBody(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new Error("Request body must be a JSON object");
  }
  const record = /** @type {Record<string, unknown>} */ (body);
  const target = resolveTranslateTarget(
    typeof record.target === "string" ? record.target : "",
  );
  if (!target) {
    throw new Error("Unsupported or English target language");
  }
  const source =
    typeof record.source === "string" && record.source.trim()
      ? record.source.trim()
      : "en";
  if (!Array.isArray(record.texts)) {
    throw new Error("`texts` must be an array of strings");
  }
  if (record.texts.length > MAX_TEXTS_PER_REQUEST) {
    throw new Error(`At most ${MAX_TEXTS_PER_REQUEST} texts per request`);
  }
  const texts = [];
  for (const item of record.texts) {
    if (typeof item !== "string") {
      throw new Error("Each text must be a string");
    }
    const trimmed = item.trim();
    if (!trimmed) continue;
    if (trimmed.length > MAX_CHARS_PER_TEXT) {
      throw new Error(
        `Each text must be at most ${MAX_CHARS_PER_TEXT} characters`,
      );
    }
    texts.push(trimmed);
  }
  return { texts, source, target };
}

/**
 * @param {string} url
 * @param {import("node:http").IncomingMessage} req
 * @returns {Promise<import("node:http").IncomingMessage>}
 */
function requestUrl(url, req) {
  const parsed = new URL(url);
  const transport = parsed.protocol === "http:" ? httpRequest : httpsRequest;
  return new Promise((resolve, reject) => {
    const upstream = transport(
      {
        protocol: parsed.protocol,
        hostname: parsed.hostname,
        port: parsed.port || undefined,
        path: `${parsed.pathname}${parsed.search}`,
        method: "GET",
        headers: {
          Accept: "application/json",
          "User-Agent": "OpenHands-Agent-Canvas-Translate/1.0",
        },
      },
      resolve,
    );
    upstream.on("error", reject);
    // Abort upstream if the client disconnects.
    req.on("aborted", () => upstream.destroy());
    upstream.end();
  });
}

/**
 * @param {import("node:http").IncomingMessage} res
 * @returns {Promise<string>}
 */
function readResponseBody(res) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    res.on("data", (c) => chunks.push(c));
    res.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    res.on("error", reject);
  });
}

/**
 * @param {string} text
 * @param {string} source
 * @param {string} target
 * @param {import("node:http").IncomingMessage} clientReq
 * @param {string} apiBase
 */
export async function translateTextChunk(
  text,
  source,
  target,
  clientReq,
  apiBase = process.env.TRANSLATE_API_URL || DEFAULT_MYMEMORY_URL,
) {
  const url = new URL(apiBase);
  url.searchParams.set("q", text);
  url.searchParams.set("langpair", `${source}|${target}`);
  const upstream = await requestUrl(url.toString(), clientReq);
  const raw = await readResponseBody(upstream);
  if ((upstream.statusCode ?? 500) >= 400) {
    throw new Error(`Translate upstream HTTP ${upstream.statusCode}`);
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("Translate upstream returned non-JSON");
  }
  const translated = parsed?.responseData?.translatedText;
  if (typeof translated !== "string" || !translated.trim()) {
    throw new Error("Translate upstream omitted translatedText");
  }
  // MyMemory returns the English tip string when quota is exceeded.
  if (/MYMEMORY WARNING/i.test(translated)) {
    throw new Error("Translate upstream quota exceeded");
  }
  return translated;
}

/**
 * @param {string} text
 * @param {string} source
 * @param {string} target
 * @param {import("node:http").IncomingMessage} clientReq
 */
export async function translateText(text, source, target, clientReq) {
  const chunks = chunkTextForTranslation(text);
  if (chunks.length === 0) return text;
  const parts = [];
  for (const chunk of chunks) {
    parts.push(await translateTextChunk(chunk, source, target, clientReq));
  }
  return parts.join(" ");
}

/**
 * @param {import("node:http").IncomingMessage} req
 * @returns {Promise<unknown>}
 */
function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw.trim()) {
        reject(new Error("Empty request body"));
        return;
      }
      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(new Error("Invalid JSON body"));
      }
    });
    req.on("error", reject);
  });
}

/**
 * @param {import("node:http").ServerResponse} res
 * @param {number} status
 * @param {unknown} payload
 */
function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

/**
 * @returns {(req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => Promise<void>}
 */
export function createTranslateProxyHandler() {
  return async (req, res) => {
    if (req.method === "OPTIONS") {
      res.writeHead(204, {
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Session-API-Key",
      });
      res.end();
      return;
    }
    if (req.method !== "POST") {
      sendJson(res, 405, { error: "Method not allowed" });
      return;
    }

    try {
      const body = await readJsonBody(req);
      const { texts, source, target } = parseTranslateRequestBody(body);
      /** @type {Record<string, string>} */
      const translations = {};
      const unique = [...new Set(texts)];
      for (const text of unique) {
        try {
          translations[text] = await translateText(text, source, target, req);
        } catch (err) {
          // Per-text failure: keep original so the UI never blanks out.
          console.warn(
            `[translate-proxy] failed for text (${text.slice(0, 40)}…):`,
            err instanceof Error ? err.message : err,
          );
          translations[text] = text;
        }
      }
      sendJson(res, 200, { translations, source, target });
    } catch (err) {
      sendJson(res, 400, {
        error: err instanceof Error ? err.message : "Bad request",
      });
    }
  };
}
