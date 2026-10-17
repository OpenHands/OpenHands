import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useLocalStorage } from "@uidotdev/usehooks";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getActiveBackend,
  isNoBackend,
} from "#/api/backend-registry/active-store";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  HOME_AUTOMATIONS_DEMO_PINNED_IDS,
  isHomeAutomationsDemoEnabled,
} from "#/fixtures/home-automations-demo";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { useSettings } from "#/hooks/query/use-settings";
import type { Settings } from "#/types/settings";

export const HOME_PINNED_AUTOMATIONS_KEY = "oh:home-pinned-automations";

/**
 * Pins are stored per backend + org: automation ids only resolve against the
 * backend that issued them, and `pruneMissing` compares against the active
 * backend's list — a shared key would let one backend wipe another's pins.
 *
 * Local backends keep pins in server settings and use this key only for the
 * one-time migration of older browser-stored pins; cloud still stores here.
 */
export function getHomePinnedAutomationsKey(
  backendId: string,
  orgId: string | null,
): string {
  return `${HOME_PINNED_AUTOMATIONS_KEY}:${backendId}:${orgId ?? "-"}`;
}

/** Soft preview cap for the home pinned dashboard before "View more". */
export const HOME_PINNED_PREVIEW_LIMIT = 6;

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

function hasSameIds(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((id, index) => id === b[index]);
}

/**
 * Legacy keys whose pins are being copied to the server. Several components
 * mount this hook at once; this keeps them from starting duplicate copies.
 */
const migratingLegacyKeys = new Set<string>();

/** Reorder `base` to match `preferred` where possible; append leftovers. */
export function applyPinnedOrder(
  base: readonly string[],
  preferred: readonly string[],
): string[] {
  const remaining = new Set(base);
  const ordered: string[] = [];
  for (const id of preferred) {
    if (remaining.has(id)) {
      ordered.push(id);
      remaining.delete(id);
    }
  }
  for (const id of base) {
    if (remaining.has(id)) ordered.push(id);
  }
  return ordered;
}

export function movePinnedId(
  ids: readonly string[],
  activeId: string,
  targetId: string,
  position: "before" | "after" = "after",
): string[] {
  if (activeId === targetId) return [...ids];
  const fromIndex = ids.indexOf(activeId);
  const toIndex = ids.indexOf(targetId);
  if (fromIndex < 0 || toIndex < 0) return [...ids];

  const next = [...ids];
  next.splice(fromIndex, 1);
  const adjustedTarget = next.indexOf(targetId);
  const insertIndex =
    position === "before" ? adjustedTarget : adjustedTarget + 1;
  next.splice(insertIndex, 0, activeId);
  return next;
}

/**
 * Pin state for home automation activity rows. Local backends persist pinned
 * ids in `misc_settings.app_preferences` on the agent-server so they follow
 * the user across browsers; cloud backends keep them in localStorage.
 * Resolution against live automations happens in the consuming components.
 */
export function useHomePinnedAutomations() {
  const demo = isHomeAutomationsDemoEnabled();
  const active = useActiveBackend();
  const backendId = active.backend.id;
  const { orgId } = active;
  const storageKey = getHomePinnedAutomationsKey(backendId, orgId);
  const usesServerSettings =
    !demo && active.backend.kind === "local" && !isNoBackend(active.backend);
  // No initial value: an initial value would re-create the legacy key right
  // after the migration below removes it.
  const [rawPinnedIds, setRawPinnedIds] = useLocalStorage<string[] | undefined>(
    storageKey,
  );
  // Demo pins are fixture-backed; keep session order overrides in memory so
  // drag reorder still works while testing with mock data.
  const [demoOrder, setDemoOrder] = useState<string[] | null>(null);

  const queryClient = useQueryClient();
  const settings = useSettings();
  const settingsQueryKey = useMemo(
    () => [...SETTINGS_QUERY_KEYS.byScope("personal"), backendId, orgId],
    [backendId, orgId],
  );
  const { mutate: savePinnedIds, mutateAsync: savePinnedIdsAsync } =
    useMutation({
      mutationKey: [storageKey],
      // Run saves one at a time so an older list never lands after a newer one.
      scope: { id: storageKey },
      mutationFn: async (ids: string[]) => {
        // A save queued on one backend must never write into another.
        const current = getActiveBackend();
        if (current.backend.id !== backendId || current.orgId !== orgId) {
          return false;
        }
        await SettingsService.saveSettings({ home_pinned_automations: ids });
        return true;
      },
      onSettled: async () => {
        // Refetch only after the last queued save: an earlier refetch would
        // replace the newer optimistic list with an older server value.
        if (queryClient.isMutating({ mutationKey: [storageKey] }) === 1) {
          await queryClient.invalidateQueries({ queryKey: settingsQueryKey });
        }
      },
    });

  const readServerPinnedIds = useCallback(() => {
    const cached = queryClient.getQueryData<Settings>(settingsQueryKey);
    return cached ? sanitizePinnedIds(cached.home_pinned_automations) : null;
  }, [queryClient, settingsQueryKey]);

  const setServerPinnedIds = useCallback(
    (ids: string[]) => {
      void queryClient.cancelQueries({ queryKey: settingsQueryKey });
      queryClient.setQueryData<Settings>(settingsQueryKey, (cached) =>
        cached ? { ...cached, home_pinned_automations: ids } : cached,
      );
    },
    [queryClient, settingsQueryKey],
  );

  const updatePinnedIds = useCallback(
    (update: (current: string[]) => string[]) => {
      if (!usesServerSettings) {
        setRawPinnedIds((current) => update(sanitizePinnedIds(current)));
        return;
      }
      // Until settings load there is no base list, and saving a partial list
      // would overwrite the user's server pins.
      const current = readServerPinnedIds();
      if (!current) return;
      const next = update(current);
      if (hasSameIds(current, next)) return;
      setServerPinnedIds(next);
      savePinnedIds(next);
    },
    [
      usesServerSettings,
      setRawPinnedIds,
      readServerPinnedIds,
      setServerPinnedIds,
      savePinnedIds,
    ],
  );

  // `useLocalStorage` parses a new array every render; the string stays
  // stable, so the effect below runs only when the stored pins change.
  const legacyPinsJson =
    rawPinnedIds == null ? null : JSON.stringify(rawPinnedIds);
  useEffect(() => {
    if (!usesServerSettings || !settings.isSuccess || legacyPinsJson === null) {
      return;
    }
    if (migratingLegacyKeys.has(storageKey)) return;
    const current = readServerPinnedIds();
    if (!current) return;
    migratingLegacyKeys.add(storageKey);
    const next = sanitizePinnedIds([
      ...current,
      ...sanitizePinnedIds(JSON.parse(legacyPinsJson)),
    ]);
    let saved: Promise<boolean> = Promise.resolve(true);
    if (!hasSameIds(current, next)) {
      setServerPinnedIds(next);
      saved = savePinnedIdsAsync(next);
    }
    saved
      .then((didSave) => {
        if (didSave) setRawPinnedIds(undefined);
      })
      // Keep the legacy pins on failure; the next mount retries the copy.
      .catch(() => undefined)
      .finally(() => migratingLegacyKeys.delete(storageKey));
  }, [
    usesServerSettings,
    settings.isSuccess,
    legacyPinsJson,
    storageKey,
    readServerPinnedIds,
    setServerPinnedIds,
    savePinnedIdsAsync,
    setRawPinnedIds,
  ]);

  const pinnedIds = useMemo(() => {
    if (demo) {
      return demoOrder
        ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, demoOrder)
        : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
    }
    return sanitizePinnedIds(
      usesServerSettings
        ? settings.data?.home_pinned_automations
        : rawPinnedIds,
    );
  }, [
    demo,
    demoOrder,
    usesServerSettings,
    settings.data?.home_pinned_automations,
    rawPinnedIds,
  ]);

  const isPinned = useCallback(
    (id: string) => pinnedIds.includes(id),
    [pinnedIds],
  );

  const pin = useCallback(
    (id: string) => {
      if (demo) {
        setDemoOrder((current) => {
          const base = current
            ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, current)
            : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
          if (base.includes(id)) return base;
          return [...base, id];
        });
        return;
      }
      updatePinnedIds((current) =>
        current.includes(id) ? current : [...current, id],
      );
    },
    [demo, updatePinnedIds],
  );

  const unpin = useCallback(
    (id: string) => {
      if (demo) {
        setDemoOrder((current) => {
          const base = current
            ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, current)
            : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
          return base.filter((pinnedId) => pinnedId !== id);
        });
        return;
      }
      updatePinnedIds((current) =>
        current.filter((pinnedId) => pinnedId !== id),
      );
    },
    [demo, updatePinnedIds],
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

  const reorder = useCallback(
    (
      activeId: string,
      targetId: string,
      position: "before" | "after" = "after",
    ) => {
      if (demo) {
        setDemoOrder((current) => {
          const base = current
            ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, current)
            : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
          return movePinnedId(base, activeId, targetId, position);
        });
        return;
      }
      updatePinnedIds((current) =>
        movePinnedId(current, activeId, targetId, position),
      );
    },
    [demo, updatePinnedIds],
  );

  /** Drop pin ids that no longer exist on the backend (deleted automations). */
  const pruneMissing = useCallback(
    (knownIds: ReadonlySet<string>) => {
      if (demo) return;
      updatePinnedIds((current) => current.filter((id) => knownIds.has(id)));
    },
    [demo, updatePinnedIds],
  );

  return {
    pinnedIds,
    isPinned,
    pin,
    unpin,
    togglePin,
    reorder,
    pruneMissing,
  };
}
