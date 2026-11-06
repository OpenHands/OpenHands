/**
 * Pairing another device: a device that is already in shows a one-time link
 * (`<base path>/pair#<secret>`, also as a QR code), and the new device opens
 * it. The Canvas server (scripts/mars-web-bridge.mjs) redeems the secret
 * once, within five minutes, for the same access as typing the Canvas key.
 */

import {
  DEFAULT_LOCAL_BACKEND_NAME,
  SEEDED_DEFAULT_BACKEND_ID,
} from "#/api/backend-registry/default-backend";
import {
  readStoredBackends,
  writeStoredActiveBackend,
  writeStoredBackends,
} from "#/api/backend-registry/storage";
import type { Backend } from "#/api/backend-registry/types";
import { buildAgentCanvasPath } from "#/utils/base-path";
import { isCanvasRootBackend } from "./canvas-root-backend";

/** The page a pairing link opens, under the Canvas base path. */
export const PAIR_PAGE_PATH = "/pair";
const PAIR_ROUTE = "/mars/pair";
/** Set for this tab when a link did not work, so the key screen can say so. */
export const PAIRING_FAILED_STORAGE_KEY = "openhands-pairing-failed";

/** `crypto.randomUUID` exists only on HTTPS and localhost; a LAN address is plain HTTP. */
function newBackendId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `backend-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Keep the key where a typed key goes: this origin's own backend, never a
 * DigitalOcean session's (`/mars/sessions/<id>` on the same origin).
 */
function saveLocalBackendKey(host: string, apiKey: string): void {
  const backends = readStoredBackends();
  const existing = backends.find(isCanvasRootBackend);
  const saved: Backend = existing
    ? { ...existing, apiKey }
    : {
        id: backends.some((backend) => backend.id === SEEDED_DEFAULT_BACKEND_ID)
          ? newBackendId()
          : SEEDED_DEFAULT_BACKEND_ID,
        name: DEFAULT_LOCAL_BACKEND_NAME,
        host,
        apiKey,
        kind: "local",
      };
  writeStoredBackends(
    existing
      ? backends.map((backend) => (backend.id === saved.id ? saved : backend))
      : [...backends, saved],
  );
  writeStoredActiveBackend({ backendId: saved.id });
}

/**
 * Redeem a pairing link before the app renders. The secret leaves the
 * address bar at once; it works only once anyway. Resolves true when this
 * device is now in, and the caller reloads so everything starts with the
 * saved key. A link that does not work leaves a note for the key screen.
 */
export async function redeemPairingLink(
  fetchImpl: typeof fetch = (...params) => fetch(...params),
): Promise<boolean> {
  const pairPage = buildAgentCanvasPath(PAIR_PAGE_PATH);
  const { pathname } = window.location;
  if (pathname !== pairPage && pathname !== `${pairPage}/`) return false;
  const secret = window.location.hash.slice(1);
  window.history.replaceState(null, "", buildAgentCanvasPath("/"));
  try {
    const response = await fetchImpl(PAIR_ROUTE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ secret }),
    });
    const body = (await response.json()) as {
      result?: { sessionApiKey?: unknown };
    };
    const key = body.result?.sessionApiKey;
    if (!response.ok || typeof key !== "string" || !key) {
      throw new Error("pairing refused");
    }
    saveLocalBackendKey(window.location.origin, key);
    return true;
  } catch {
    try {
      window.sessionStorage.setItem(PAIRING_FAILED_STORAGE_KEY, "1");
    } catch {
      // Without storage the key screen simply shows as usual.
    }
    return false;
  }
}

/** Whether this tab arrived through a pairing link that did not work. */
export function hadPairingFailure(): boolean {
  try {
    return window.sessionStorage.getItem(PAIRING_FAILED_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}
