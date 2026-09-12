import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isNoBackend } from "#/api/backend-registry/active-store";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  HOME_AUTOMATIONS_DEMO_PINNED_IDS,
  isHomeAutomationsDemoEnabled,
} from "#/fixtures/home-automations-demo";
import { HOME_PINNED_AUTOMATIONS_QUERY_KEYS } from "#/hooks/query/query-keys";

export const HOME_PINNED_AUTOMATIONS_KEY = "oh:home-pinned-automations";

/**
 * Returns the localStorage key for a given backend+org pair. Kept for the
 * one-time migration out of localStorage on first server-backed load.
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
 * Pin state for home automation activity rows. For local backends, persists
 * pinned ids in `misc_settings.ui_preferences` so the dashboard syncs across
 * devices. Migrates any existing localStorage data on first load.
 *
 * Optimistic updates go through the React Query cache (`setQueryData`) so
 * every component calling this hook sees the change immediately — no separate
 * local state per instance.
 *
 * Demo mode and cloud backends use in-memory state only.
 */
export function useHomePinnedAutomations() {
  const demo = isHomeAutomationsDemoEnabled();
  const active = useActiveBackend();
  const hasBackend = !isNoBackend(active.backend);
  const isCloud = active.backend.kind === "cloud";
  const useServerSync = !demo && hasBackend && !isCloud;
  const queryClient = useQueryClient();

  // Demo / cloud: session-order overrides kept in memory.
  const [inMemoryOrder, setInMemoryOrder] = useState<string[] | null>(null);

  const currentKey = `${active.backend.id}:${active.orgId ?? "-"}`;
  const lsKey = getHomePinnedAutomationsKey(active.backend.id, active.orgId);

  // Prevent migration from running more than once per backend+org.
  const migratedRef = useRef<Set<string>>(new Set());

  // Stabilise the query key so writePins / useEffect deps don't churn.
  const queryKey = useMemo(
    () =>
      HOME_PINNED_AUTOMATIONS_QUERY_KEYS.byBackend(
        active.backend.id,
        active.orgId,
      ),
    [active.backend.id, active.orgId],
  );

  // Fetch the server-side pinned IDs for the current backend+org.
  // `null` means "server has no entry yet"; `[]` means "explicitly empty".
  const { data: serverIds, isSuccess: serverReady } = useQuery({
    queryKey,
    queryFn: async () => {
      const key = `${active.backend.id}:${active.orgId ?? "-"}`;
      const response = await SettingsService.fetchSettingsFromApi();
      return (
        response.misc_settings?.ui_preferences?.home_pinned_automations?.[
          key
        ] ?? null
      );
    },
    enabled: useServerSync,
    staleTime: 5 * 60 * 1000,
    gcTime: 15 * 60 * 1000,
    meta: { disableToast: true },
  });

  // Write ids to the server and update the shared React Query cache
  // optimistically so every hook instance re-renders immediately.
  const writePins = useCallback(
    (ids: string[]) => {
      if (!useServerSync) return;
      // Update the cache first — all instances of this hook share the same
      // query key and will re-render synchronously on the next tick.
      queryClient.setQueryData(queryKey, ids);
      void SettingsService.patchUiPreferences({
        home_pinned_automations: { [currentKey]: ids },
      }).then(() => {
        void queryClient.invalidateQueries({ queryKey });
      });
    },
    [useServerSync, currentKey, queryClient, queryKey],
  );

  // One-time migration: when the server query first settles for this
  // backend+org, migrate any existing localStorage pins to the server.
  useEffect(() => {
    if (!serverReady || migratedRef.current.has(currentKey)) return;
    migratedRef.current.add(currentKey);

    if (serverIds !== null) {
      // Server already has pin data — just clean up any stale localStorage key.
      try {
        localStorage.removeItem(lsKey);
      } catch {
        /* ignore storage errors */
      }
    } else {
      // Server has no data — check localStorage for a one-time migration.
      try {
        const raw = localStorage.getItem(lsKey);
        const ids = sanitizePinnedIds(raw !== null ? JSON.parse(raw) : []);
        if (ids.length > 0) {
          writePins(ids);
        } else {
          // Initialise the cache to an empty array so consumers know we've
          // settled and won't treat `null` as "still loading".
          queryClient.setQueryData(queryKey, [] as string[]);
        }
        localStorage.removeItem(lsKey);
      } catch {
        queryClient.setQueryData(queryKey, [] as string[]);
      }
    }
  }, [
    serverReady,
    serverIds,
    currentKey,
    lsKey,
    writePins,
    queryClient,
    queryKey,
  ]);

  // Apply an updater to the current pin list, write optimistically via the
  // shared query cache. Updaters that return the same reference as `current`
  // are treated as no-ops (no cache update, no server write) — used by
  // pruneMissing.
  const updateIds = useCallback(
    (updater: (current: string[]) => string[]) => {
      const current = sanitizePinnedIds(serverIds ?? []);
      const next = updater(current);
      if (next === current) return;
      writePins(next);
    },
    [serverIds, writePins],
  );

  const pinnedIds = useMemo(() => {
    if (demo) {
      return inMemoryOrder
        ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, inMemoryOrder)
        : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
    }
    if (isCloud) {
      return sanitizePinnedIds(inMemoryOrder ?? []);
    }
    return sanitizePinnedIds(serverIds ?? []);
  }, [demo, isCloud, inMemoryOrder, serverIds]);

  const isPinned = useCallback(
    (id: string) => pinnedIds.includes(id),
    [pinnedIds],
  );

  const pin = useCallback(
    (id: string) => {
      if (demo) {
        setInMemoryOrder((current) => {
          const base = current
            ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, current)
            : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
          if (base.includes(id)) return base;
          return [...base, id];
        });
        return;
      }
      if (isCloud) {
        setInMemoryOrder((current) => {
          const base = sanitizePinnedIds(current ?? []);
          if (base.includes(id)) return base;
          return [...base, id];
        });
        return;
      }
      updateIds((current) => {
        const next = sanitizePinnedIds(current);
        if (next.includes(id)) return next;
        return [...next, id];
      });
    },
    [demo, isCloud, updateIds],
  );

  const unpin = useCallback(
    (id: string) => {
      if (demo) {
        setInMemoryOrder((current) => {
          const base = current
            ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, current)
            : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
          return base.filter((pinnedId) => pinnedId !== id);
        });
        return;
      }
      if (isCloud) {
        setInMemoryOrder((current) =>
          sanitizePinnedIds(current ?? []).filter(
            (pinnedId) => pinnedId !== id,
          ),
        );
        return;
      }
      updateIds((current) =>
        sanitizePinnedIds(current).filter((pinnedId) => pinnedId !== id),
      );
    },
    [demo, isCloud, updateIds],
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
        setInMemoryOrder((current) => {
          const base = current
            ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, current)
            : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
          return movePinnedId(base, activeId, targetId, position);
        });
        return;
      }
      if (isCloud) {
        setInMemoryOrder((current) =>
          movePinnedId(
            sanitizePinnedIds(current ?? []),
            activeId,
            targetId,
            position,
          ),
        );
        return;
      }
      updateIds((current) =>
        movePinnedId(sanitizePinnedIds(current), activeId, targetId, position),
      );
    },
    [demo, isCloud, updateIds],
  );

  /** Drop pin ids that no longer exist on the backend (deleted automations). */
  const pruneMissing = useCallback(
    (knownIds: ReadonlySet<string>) => {
      if (demo) return;
      if (isCloud) {
        setInMemoryOrder((current) =>
          sanitizePinnedIds(current ?? []).filter((id) => knownIds.has(id)),
        );
        return;
      }
      updateIds((current) => {
        const sanitized = sanitizePinnedIds(current);
        const next = sanitized.filter((id) => knownIds.has(id));
        if (
          next.length === sanitized.length &&
          next.every((id, index) => id === sanitized[index])
        ) {
          return current; // identity signals no-op — skip server write
        }
        return next;
      });
    },
    [demo, isCloud, updateIds],
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
