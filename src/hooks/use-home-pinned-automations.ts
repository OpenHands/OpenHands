import { useLocalStorage } from "@uidotdev/usehooks";
import { useCallback, useMemo } from "react";

export const HOME_PINNED_AUTOMATIONS_KEY = "oh:home-pinned-automations";

function sanitizePinnedIds(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const seen = new Set<string>();
  const next: string[] = [];
  for (const entry of value) {
    if (typeof entry !== "string" || !entry || seen.has(entry)) {
      continue;
    }
    seen.add(entry);
    next.push(entry);
  }
  return next;
}

/**
 * Pin state for home automation activity rows. Persists pinned ids in
 * localStorage so dashboard modules survive reload. Resolution against live
 * automations happens in the consuming components.
 */
export function useHomePinnedAutomations() {
  const [rawPinnedIds, setRawPinnedIds] = useLocalStorage<string[]>(
    HOME_PINNED_AUTOMATIONS_KEY,
    [],
  );

  const pinnedIds = useMemo(
    () => sanitizePinnedIds(rawPinnedIds),
    [rawPinnedIds],
  );

  const isPinned = useCallback(
    (id: string) => pinnedIds.includes(id),
    [pinnedIds],
  );

  const pin = useCallback(
    (id: string) => {
      setRawPinnedIds((current) => {
        const next = sanitizePinnedIds(current);
        if (next.includes(id)) return next;
        return [...next, id];
      });
    },
    [setRawPinnedIds],
  );

  const unpin = useCallback(
    (id: string) => {
      setRawPinnedIds((current) =>
        sanitizePinnedIds(current).filter((pinnedId) => pinnedId !== id),
      );
    },
    [setRawPinnedIds],
  );

  const togglePin = useCallback(
    (id: string) => {
      if (isPinned(id)) {
        unpin(id);
        return;
      }
      pin(id);
    },
    [isPinned, pin, unpin],
  );

  return {
    pinnedIds,
    isPinned,
    pin,
    unpin,
    togglePin,
  };
}
