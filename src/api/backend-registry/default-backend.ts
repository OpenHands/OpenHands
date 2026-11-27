import {
  getAgentServerBaseUrl,
  getAgentServerSessionApiKey,
  getCookieAuthCloudHost,
  getLockedCloudHost,
} from "../agent-server-config";
import type { Backend } from "./types";

/**
 * Stable id for the seeded default local backend that is auto-registered in
 * the backend registry when the launcher provides both a backend host and
 * API key. After seeding, this backend is a normal registered entry — the
 * user can rename it, edit its host/api key, or remove it like any other
 * backend.
 */
export const SEEDED_DEFAULT_BACKEND_ID = "default-local";

export const DEFAULT_LOCAL_BACKEND_NAME = "Local";
export const LOCKED_CLOUD_BACKEND_ID = "locked-cloud";
export const LOCKED_CLOUD_BACKEND_NAME = "OpenHands Cloud";

export function makeLockedCloudBackend(): Backend | null {
  if (!getLockedCloudHost()) return null;

  const host = getCookieAuthCloudHost();
  if (!host) return null;

  return {
    id: LOCKED_CLOUD_BACKEND_ID,
    name: LOCKED_CLOUD_BACKEND_NAME,
    host,
    apiKey: "",
    kind: "cloud",
    authMode: "cookie",
  };
}

/**
 * Construct the default local backend from environment/runtime config.
 * Returns null unless both a backend location and API key are available.
 *
 * Used as the seed entry written to `openhands-backends` on first load;
 * if it returns null, onboarding is responsible for collecting backend
 * connection details from the user.
 *
 * Returns null when the deployment is locked to a single OpenHands Cloud
 * host (`VITE_LOCK_TO_CLOUD` / `--lock-to-cloud`). In locked mode the user
 * can only authenticate against the configured Cloud URL, so seeding a
 * Local backend from a baked/injected session key would short-circuit the
 * first-run onboarding gate and strand the user on the Manage Backends
 * recovery modal with a disconnected Local entry.
 */
export function makeDefaultLocalBackend(): Backend | null {
  // Locked-to-Cloud deployments must never auto-seed a Local backend —
  // see the docblock above.
  if (getLockedCloudHost()) return null;

  const host = getAgentServerBaseUrl();
  const apiKey = getAgentServerSessionApiKey();

  if (!host || !apiKey) return null;

  return {
    id: SEEDED_DEFAULT_BACKEND_ID,
    name: DEFAULT_LOCAL_BACKEND_NAME,
    host,
    apiKey,
    kind: "local",
  };
}
