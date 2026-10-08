import type { ACPModelOption } from "#/constants/acp-providers";

const STORAGE_PREFIX = "openhands-acp-models";

function storageKey(backendId: string, providerKey: string): string {
  return `${STORAGE_PREFIX}:${backendId}:${providerKey}`;
}

function isModelOption(value: unknown): value is ACPModelOption {
  if (typeof value !== "object" || value === null) return false;
  const { id, label } = value as Record<string, unknown>;
  return typeof id === "string" && !!id && typeof label === "string";
}

/** The models a built-in ACP agent last reported on this backend, in this browser. */
export function readRememberedAcpModels(
  backendId: string,
  providerKey: string | null | undefined,
): ACPModelOption[] {
  if (typeof window === "undefined" || !providerKey) return [];
  try {
    const stored = JSON.parse(
      window.localStorage.getItem(storageKey(backendId, providerKey)) ?? "[]",
    );
    return Array.isArray(stored) ? stored.filter(isModelOption) : [];
  } catch {
    return [];
  }
}

export function rememberAcpModels(
  backendId: string,
  providerKey: string,
  models: readonly ACPModelOption[],
): void {
  if (typeof window === "undefined" || !models.length) return;
  try {
    window.localStorage.setItem(
      storageKey(backendId, providerKey),
      JSON.stringify(models.map(({ id, label }) => ({ id, label }))),
    );
  } catch {
    // Storage is unavailable; the list is only a convenience.
  }
}
