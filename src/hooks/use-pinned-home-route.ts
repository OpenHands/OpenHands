import { useLocalStorage } from "@uidotdev/usehooks";
import { useCallback, useMemo } from "react";
import {
  getActiveBackend,
  isNoBackend,
} from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  automationListPath,
  hasAutomationInterface,
} from "#/manifests/automation-interface";

export const PINNED_HOME_ROUTE_KEY = "oh:pinned-home-route";

/** Sidebar Customize entry; also the pin target, so the two cannot drift. */
export const CUSTOMIZE_PATH = "/customize";

/**
 * The pin is stored per backend + org: it may reference a surface that only
 * exists on the backend that set it (e.g. /automations requires that
 * deployment's interface manifest) — a shared key would let one backend
 * redirect `/` on another.
 */
export function getPinnedHomeRouteKey(
  backendId: string,
  orgId: string | null,
): string {
  return `${PINNED_HOME_ROUTE_KEY}:${backendId}:${orgId ?? "-"}`;
}

/**
 * Whether `path` may serve as the home route right now. Shared by the
 * sidebar pin affordance and the `/` loader, so a stored pin that stops
 * resolving (backend switch, manifest absent) is ignored rather than an
 * error. `/` is never pinnable, which makes a redirect loop impossible.
 * Canvas Extensions pages can later add an `/extensions/…` branch here
 * without any storage or loader change.
 */
export function isPinnableRoute(path: string): boolean {
  if (path === CUSTOMIZE_PATH) return true;
  if (path === automationListPath()) return hasAutomationInterface();
  return false;
}

function sanitizePinnedRoute(value: unknown): string | null {
  if (typeof value !== "string" || !value) return null;
  return isPinnableRoute(value) ? value : null;
}

/**
 * Synchronous pin read for route loaders (no React context). Reads the key
 * `usePinnedHomeRoute` writes; `useLocalStorage` JSON-serializes values, so
 * parse defensively and treat anything unreadable as "no pin". An invalid
 * pin is ignored, not cleared — it may become valid again (e.g. the
 * automations interface returning after a backend switch back).
 */
export function readPinnedHomeRoute(): string | null {
  const active = getActiveBackend();
  if (isNoBackend(active.backend)) return null;

  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(
      getPinnedHomeRouteKey(active.backend.id, active.orgId),
    );
  } catch {
    return null;
  }
  if (raw === null) return null;

  try {
    return sanitizePinnedRoute(JSON.parse(raw));
  } catch {
    return null;
  }
}

/** Pin state for the home route: `/` redirects to the pinned sidebar page. */
export function usePinnedHomeRoute() {
  const active = useActiveBackend();
  const [rawPinnedRoute, setRawPinnedRoute] = useLocalStorage<string | null>(
    getPinnedHomeRouteKey(active.backend.id, active.orgId),
    null,
  );

  const pinnedRoute = useMemo(
    () => sanitizePinnedRoute(rawPinnedRoute),
    [rawPinnedRoute],
  );

  const isPinnedRoute = useCallback(
    (path: string) => pinnedRoute === path,
    [pinnedRoute],
  );

  const togglePinnedRoute = useCallback(
    (path: string) => {
      if (pinnedRoute === path) {
        setRawPinnedRoute(null);
        return;
      }
      if (!isPinnableRoute(path)) return;
      setRawPinnedRoute(path);
    },
    [pinnedRoute, setRawPinnedRoute],
  );

  return {
    pinnedRoute,
    isPinnedRoute,
    togglePinnedRoute,
  };
}
