import { getUserFacingConnectionErrorMessage } from "#/utils/user-facing-error";

export const BACKEND_HEALTH_STORAGE_KEY = "openhands-backend-health";

/**
 * Once a backend has failed this many probes in a row, polling stops
 * until the user edits the backend's host or apiKey (which resets the
 * counter via the active-backend store).
 */
export const MAX_CONSECUTIVE_FAILURES = 5;

/** Max characters of `error.message` we persist — keeps localStorage tidy. */
const MAX_ERROR_MESSAGE_LENGTH = 500;

export interface BackendHealthEntry {
  consecutiveFailures: number;
  lastError: string | null;
  lastFailureAt: number | null;
  disabled: boolean;
}

export type BackendHealthMap = Record<string, BackendHealthEntry>;

function isValidEntry(value: unknown): value is BackendHealthEntry {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Partial<BackendHealthEntry>;
  // localStorage is user-writable. Reject anything outside the range
  // our own writer produces — a tampered `-1` would never reach the
  // cap and would defeat the whole disable mechanism, and a giant
  // value would clutter the UI for no reason.
  return (
    Number.isInteger(v.consecutiveFailures) &&
    (v.consecutiveFailures as number) >= 0 &&
    (v.consecutiveFailures as number) <= MAX_CONSECUTIVE_FAILURES &&
    (v.lastError === null || typeof v.lastError === "string") &&
    (v.lastFailureAt === null || typeof v.lastFailureAt === "number") &&
    typeof v.disabled === "boolean"
  );
}

export function readStoredHealth(): BackendHealthMap {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(BACKEND_HEALTH_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return {};

    const out: BackendHealthMap = {};
    for (const [id, entry] of Object.entries(parsed)) {
      if (typeof id === "string" && id.length > 0 && isValidEntry(entry)) {
        out[id] = entry;
      }
    }
    return out;
  } catch {
    return {};
  }
}

export function writeStoredHealth(map: BackendHealthMap): void {
  if (typeof window === "undefined") return;
  try {
    if (Object.keys(map).length === 0) {
      window.localStorage.removeItem(BACKEND_HEALTH_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(
      BACKEND_HEALTH_STORAGE_KEY,
      JSON.stringify(map),
    );
  } catch {
    /* ignore quota / serialization errors */
  }
}

export function truncateErrorMessage(error: unknown): string {
  const message = getUserFacingConnectionErrorMessage(error) ?? "Unknown error";
  return message.length > MAX_ERROR_MESSAGE_LENGTH
    ? `${message.slice(0, MAX_ERROR_MESSAGE_LENGTH)}…`
    : message;
}
