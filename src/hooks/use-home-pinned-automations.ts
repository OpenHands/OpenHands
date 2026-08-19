import { useLocalStorage } from "@uidotdev/usehooks";
import { useCallback, useMemo, useState } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  HOME_AUTOMATIONS_DEMO_PINNED_IDS,
  isHomeAutomationsDemoEnabled,
} from "#/fixtures/home-automations-demo";

export const HOME_PINNED_AUTOMATIONS_KEY = "oh:home-pinned-automations";

/**
 * Pins are stored per backend + org: automation ids only resolve against the
 * backend that issued them, and `pruneMissing` compares against the active
 * backend's list — a shared key would let one backend wipe another's pins.
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
 * Pin state for home automation activity rows. Persists pinned ids in
 * localStorage so dashboard modules survive reload. Resolution against live
 * automations happens in the consuming components.
 */
export function useHomePinnedAutomations() {
  const demo = isHomeAutomationsDemoEnabled();
  const active = useActiveBackend();
  const [rawPinnedIds, setRawPinnedIds] = useLocalStorage<string[]>(
    getHomePinnedAutomationsKey(active.backend.id, active.orgId),
    [],
  );
  // Demo pins are fixture-backed; keep session order overrides in memory so
  // drag reorder still works while testing with mock data.
  const [demoOrder, setDemoOrder] = useState<string[] | null>(null);

  const pinnedIds = useMemo(() => {
    if (demo) {
      return demoOrder
        ? applyPinnedOrder(HOME_AUTOMATIONS_DEMO_PINNED_IDS, demoOrder)
        : [...HOME_AUTOMATIONS_DEMO_PINNED_IDS];
    }
    return sanitizePinnedIds(rawPinnedIds);
  }, [demo, demoOrder, rawPinnedIds]);

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
      setRawPinnedIds((current) => {
        const next = sanitizePinnedIds(current);
        if (next.includes(id)) return next;
        return [...next, id];
      });
    },
    [demo, setRawPinnedIds],
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
      setRawPinnedIds((current) =>
        sanitizePinnedIds(current).filter((pinnedId) => pinnedId !== id),
      );
    },
    [demo, setRawPinnedIds],
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
      setRawPinnedIds((current) =>
        movePinnedId(sanitizePinnedIds(current), activeId, targetId, position),
      );
    },
    [demo, setRawPinnedIds],
  );

  /** Drop pin ids that no longer exist on the backend (deleted automations). */
  const pruneMissing = useCallback(
    (knownIds: ReadonlySet<string>) => {
      if (demo) return;
      setRawPinnedIds((current) => {
        const sanitized = sanitizePinnedIds(current);
        const next = sanitized.filter((id) => knownIds.has(id));
        if (
          next.length === sanitized.length &&
          next.every((id, index) => id === sanitized[index])
        ) {
          return sanitized;
        }
        return next;
      });
    },
    [demo, setRawPinnedIds],
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
