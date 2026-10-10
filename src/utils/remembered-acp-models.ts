import type { ACPModelOption } from "#/constants/acp-providers";

const STORAGE_PREFIX = "openhands-acp-models";

const listeners = new Set<() => void>();

type RememberedLists = Record<string, ACPModelOption[]>;

function storageKey(backendId: string, providerKey: string): string {
  return `${STORAGE_PREFIX}:${backendId}:${providerKey}`;
}

function isModelOption(value: unknown): value is ACPModelOption {
  if (typeof value !== "object" || value === null) return false;
  const { id, label } = value as Record<string, unknown>;
  return typeof id === "string" && !!id && typeof label === "string";
}

function parseLists(raw: string | null): RememberedLists {
  if (!raw) return {};
  try {
    const stored = JSON.parse(raw);
    if (typeof stored !== "object" || stored === null || Array.isArray(stored))
      return {};
    return Object.fromEntries(
      Object.entries(stored).map(([scope, models]) => [
        scope,
        Array.isArray(models) ? models.filter(isModelOption) : [],
      ]),
    );
  } catch {
    return {};
  }
}

/** The scope a list is remembered under: the secrets the agent could use. */
export function acpModelScope(
  secretRefs: readonly string[] | null | undefined,
): string {
  return secretRefs ? JSON.stringify([...new Set(secretRefs)].sort()) : "*";
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

/** The list remembered under ``scope``, or every remembered model when omitted. */
export function parseRememberedAcpModels(
  raw: string | null,
  scope?: string,
): ACPModelOption[] {
  const lists = parseLists(raw);
  if (scope !== undefined) return lists[scope] ?? [];
  const byId = new Map<string, ACPModelOption>();
  Object.values(lists)
    .flat()
    .forEach((model) => {
      if (!byId.has(model.id)) byId.set(model.id, model);
    });
  return [...byId.values()];
}

/** The models a built-in ACP agent last reported on this backend, in this browser. */
export function readRememberedAcpModels(
  backendId: string,
  providerKey: string | null | undefined,
  scope?: string,
): ACPModelOption[] {
  return parseRememberedAcpModels(
    readRememberedAcpModelsRaw(backendId, providerKey),
    scope,
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
  scope: string,
  models: readonly ACPModelOption[],
): void {
  if (typeof window === "undefined" || !models.length) return;
  const list = models.map(({ id, label }) => ({ id, label }));
  try {
    const key = storageKey(backendId, providerKey);
    const lists = parseLists(window.localStorage.getItem(key));
    if (JSON.stringify(lists[scope]) === JSON.stringify(list)) return;
    window.localStorage.setItem(
      key,
      JSON.stringify({ ...lists, [scope]: list }),
    );
  } catch {
    // Storage is unavailable; the list is only a convenience.
    return;
  }
  listeners.forEach((listener) => listener());
}
