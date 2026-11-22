import type { Backend } from "./types";

/**
 * Owner of the persistence policy for backend session keys.
 *
 * `Backend.apiKey` carries real credentials: agent-server session keys for
 * local backends and the device-flow OAuth access token for API-key cloud
 * backends. Persisting them as plaintext in `localStorage` keeps a durable,
 * cross-tab copy on disk that anything executing on the canvas origin can
 * read. This module scopes those keys to the current tab via
 * `sessionStorage` instead: a reload re-hydrates them, a fresh tab surfaces
 * an empty record so the existing entry screens ask for the key again.
 *
 * This narrows the at-rest exposure; it does not defend against same-origin
 * script, which can read any key the page itself uses.
 *
 * Loopback keys stay in the persisted record: they are re-derived from the
 * launcher on every session and are the mechanism the published binary and
 * e2e fixtures rely on. MARS guest keys are a public constant bundled into
 * the client, so scoping them buys nothing.
 */

export const SESSION_KEYS_STORAGE_KEY = "openhands-session-keys";

export function isLoopbackUrl(value: string): boolean {
  try {
    const { hostname } = new URL(value);
    return (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "::1" ||
      hostname === "[::1]"
    );
  } catch {
    return false;
  }
}

/**
 * Whether a backend's `apiKey` may stay inside the record persisted to
 * `localStorage`. Records governed by the tab-scoped policy are persisted
 * with a blanked key and re-hydrated from `sessionStorage` on read.
 */
export function shouldPersistApiKeyInRecord(backend: Backend): boolean {
  if (backend.kind === "local") {
    return (
      isLoopbackUrl(backend.host) ||
      backend.marsSessionId !== undefined ||
      backend.marsConfigId !== undefined
    );
  }
  // Cookie-auth cloud records carry "" (the credential lives in cookies), so
  // their record needs no scoping. Every other cloud record is a device-flow
  // OAuth token — the decision must not depend on the current key value,
  // which flips to "" once the record is persisted blanked.
  return backend.authMode === "cookie";
}

function readSessionKeys(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.sessionStorage.getItem(SESSION_KEYS_STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return {};
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      ),
    );
  } catch {
    return {};
  }
}

export function readSessionApiKey(backendId: string): string | null {
  return readSessionKeys()[backendId] ?? null;
}

/**
 * Rewrite the tab-scoped key map to match the given in-memory backends, so
 * keys follow the same create/update/delete transitions as the records:
 * removed backends and keys cleared to "" leave no stale entry behind.
 */
export function syncSessionApiKeys(backends: Backend[]): void {
  if (typeof window === "undefined") return;
  const keys: Record<string, string> = {};
  for (const backend of backends) {
    if (!shouldPersistApiKeyInRecord(backend) && backend.apiKey.length > 0) {
      keys[backend.id] = backend.apiKey;
    }
  }
  try {
    if (Object.keys(keys).length === 0) {
      window.sessionStorage.removeItem(SESSION_KEYS_STORAGE_KEY);
    } else {
      window.sessionStorage.setItem(
        SESSION_KEYS_STORAGE_KEY,
        JSON.stringify(keys),
      );
    }
  } catch {
    /* ignore quota / serialization errors */
  }
}
