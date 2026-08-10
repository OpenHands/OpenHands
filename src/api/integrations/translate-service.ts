import axios from "axios";
import { NoBackendAvailableError } from "#/api/agent-server-client-options";
import { getEffectiveLocalBackend } from "#/api/backend-registry/active-store";

/**
 * Canvas-owned translate proxy path. Requests go to ingress/static-server,
 * which calls MyMemory (or ``TRANSLATE_API_URL``) server-side to avoid CORS.
 */
export const TRANSLATE_PROXY_BASE = "/api/integrations/translate";

const translationCache = new Map<string, string>();

function cacheKey(source: string, target: string, text: string): string {
  return `${source}|${target}|${text}`;
}

/**
 * Map Canvas i18n language → MyMemory target. Returns ``null`` when the UI
 * is already English (no translation needed).
 */
export function resolveUiTranslateTarget(
  language: string | null | undefined,
): string | null {
  const raw = String(language ?? "").trim();
  if (!raw) return null;
  const base = raw.split("-")[0]?.toLowerCase() ?? "";
  if (!base || base === "en") return null;
  if (base === "pt") return "pt-BR";
  return base;
}

export interface TranslateBatchResult {
  translations: Record<string, string>;
  source: string;
  target: string;
}

/**
 * Translate dynamic security-finding prose for the active UI language.
 * Results are cached in-memory for the session so re-renders / re-scans of
 * the same messages do not re-hit the upstream API.
 */
export class TranslateService {
  static clearCache(): void {
    translationCache.clear();
  }

  static async translateBatch(
    texts: readonly string[],
    targetLanguage: string,
    sourceLanguage = "en",
  ): Promise<Map<string, string>> {
    const target = resolveUiTranslateTarget(targetLanguage);
    const result = new Map<string, string>();
    if (!target) {
      for (const text of texts) {
        if (text.trim()) result.set(text, text);
      }
      return result;
    }

    const unique = [
      ...new Set(texts.map((t) => t.trim()).filter((t) => t.length > 0)),
    ];
    const missing: string[] = [];
    for (const text of unique) {
      const key = cacheKey(sourceLanguage, target, text);
      const cached = translationCache.get(key);
      if (cached != null) {
        result.set(text, cached);
      } else {
        missing.push(text);
      }
    }

    if (missing.length === 0) return result;

    const backend = getEffectiveLocalBackend();
    if (!backend) {
      throw new NoBackendAvailableError();
    }

    const url = `${backend.host.replace(/\/+$/, "")}${TRANSLATE_PROXY_BASE}`;
    const apiKey = backend.apiKey?.trim();

    const response = await axios.post<TranslateBatchResult>(
      url,
      {
        texts: missing,
        source: sourceLanguage,
        target: targetLanguage,
      },
      {
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          ...(apiKey ? { "X-Session-API-Key": apiKey } : {}),
        },
      },
    );

    const translations = response.data?.translations ?? {};
    for (const text of missing) {
      const translated =
        typeof translations[text] === "string" && translations[text].trim()
          ? translations[text]
          : text;
      translationCache.set(cacheKey(sourceLanguage, target, text), translated);
      result.set(text, translated);
    }
    return result;
  }
}
