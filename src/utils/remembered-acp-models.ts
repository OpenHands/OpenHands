import type { ACPModelOption } from "#/constants/acp-providers";

const STORAGE_PREFIX = "openhands-acp-models";

const listeners = new Set<() => void>();

function storageKey(backendId: string, providerKey: string): string {
  return `${STORAGE_PREFIX}:${backendId}:${providerKey}`;
}

function isModelOption(value: unknown): value is ACPModelOption {
  if (typeof value !== "object" || value === null) return false;
  const { id, label } = value as Record<string, unknown>;
  return typeof id === "string" && !!id && typeof label === "string";
}

/** The stored JSON for {@link readRememberedAcpModels}, or null. */
export function readRememberedAcpModelsRaw(
  backendId: string,
  providerKey: string | null | undefined,
): string | null {
  if (typeof window === "undefined" || !providerKey) return null;
  try {
    return window.localStorage.getItem(storageKey(backendId, providerKey));
  } catch {
    return null;
  }
}

export function parseRememberedAcpModels(raw: string | null): ACPModelOption[] {
  if (!raw) return [];
  try {
    const stored = JSON.parse(raw);
    return Array.isArray(stored) ? stored.filter(isModelOption) : [];
  } catch {
    return [];
  }
}

/** The models a built-in ACP agent last reported on this backend, in this browser. */
export function readRememberedAcpModels(
  backendId: string,
  providerKey: string | null | undefined,
): ACPModelOption[] {
  return parseRememberedAcpModels(
    readRememberedAcpModelsRaw(backendId, providerKey),
  );
}

/** Call ``listener`` whenever a remembered list changes, in this tab or another. */
export function subscribeRememberedAcpModels(listener: () => void) {
  listeners.add(listener);
  window.addEventListener("storage", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

export function rememberAcpModels(
  backendId: string,
  providerKey: string,
  models: readonly ACPModelOption[],
): void {
  if (typeof window === "undefined" || !models.length) return;
  const raw = JSON.stringify(models.map(({ id, label }) => ({ id, label })));
  try {
    const key = storageKey(backendId, providerKey);
    if (window.localStorage.getItem(key) === raw) return;
    window.localStorage.setItem(key, raw);
  } catch {
    // Storage is unavailable; the list is only a convenience.
    return;
  }
  listeners.forEach((listener) => listener());
}
