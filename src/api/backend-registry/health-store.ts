import {
  MAX_CONSECUTIVE_FAILURES,
  readStoredHealth,
  truncateErrorMessage,
  writeStoredHealth,
  type BackendHealthEntry,
  type BackendHealthMap,
} from "./health-storage";

type Listener = () => void;

let healthMap: BackendHealthMap = readStoredHealth();
const listeners = new Set<Listener>();

function notify(): void {
  listeners.forEach((listener) => listener());
}

function commit(next: BackendHealthMap): void {
  healthMap = next;
  writeStoredHealth(next);
  notify();
}

export function getHealthSnapshot(): BackendHealthMap {
  return healthMap;
}

export function getBackendHealthEntry(id: string): BackendHealthEntry | null {
  return healthMap[id] ?? null;
}

export function subscribeBackendHealth(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Increment the per-backend failure counter. Once the count reaches
 * `MAX_CONSECUTIVE_FAILURES`, the entry is marked `disabled: true`
 * which the polling hook reads to stop firing probes — including on a
 * fresh page load, since the state lives in localStorage.
 */
export function recordBackendFailure(id: string, error: unknown): void {
  const prev = healthMap[id];
  const consecutiveFailures = Math.min(
    (prev?.consecutiveFailures ?? 0) + 1,
    MAX_CONSECUTIVE_FAILURES,
  );
  const disabled =
    prev?.disabled === true || consecutiveFailures >= MAX_CONSECUTIVE_FAILURES;

  const nextEntry: BackendHealthEntry = {
    consecutiveFailures,
    lastError: truncateErrorMessage(error),
    lastFailureAt: Date.now(),
    disabled,
  };

  commit({ ...healthMap, [id]: nextEntry });
}

/**
 * Clear the entry so the next 10s tick can mark it healthy again.
 * Called on a successful probe — also covers the case where a backend
 * had a few failures but recovered before hitting the cap.
 */
export function recordBackendSuccess(id: string): void {
  if (!(id in healthMap)) return;
  const { [id]: _removed, ...rest } = healthMap;
  commit(rest);
}

/**
 * Re-arm polling for a backend after the user updates its config.
 * Identical to `recordBackendSuccess` today, but kept distinct so the
 * call sites read clearly.
 */
export function resetBackendHealth(id: string): void {
  if (!(id in healthMap)) return;
  const { [id]: _removed, ...rest } = healthMap;
  commit(rest);
}

/** Drop the entry entirely — used when the backend is deleted. */
export function dropBackendHealth(id: string): void {
  if (!(id in healthMap)) return;
  const { [id]: _removed, ...rest } = healthMap;
  commit(rest);
}

/** Test-only: re-read storage and clear listeners. */

export function __resetHealthStoreForTests(): void {
  healthMap = readStoredHealth();
  listeners.clear();
}
