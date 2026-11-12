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
 * Canvas Extension pages are pinnable through `extensionHrefs`: a pin is
 * valid only while the page's extension is currently enabled, so callers
 * that know the live pages (sidebar, index route) pass their href set.
 */
export function isPinnableRoute(
  path: string,
  extensionHrefs?: ReadonlySet<string>,
): boolean {
  if (path === CUSTOMIZE_PATH) return true;
  if (path === automationListPath()) return hasAutomationInterface();
  if (isExtensionHomePath(path)) return extensionHrefs?.has(path) ?? false;
  return false;
}

/** Structural match for a Canvas Extension page href (#16995). */
export function isExtensionHomePath(path: string): boolean {
  return /^\/extensions\/[^/]+(\/|$)/.test(path);
}

function sanitizePinnedRoute(
  value: unknown,
  extensionHrefs?: ReadonlySet<string>,
): string | null {
  if (typeof value !== "string" || !value) return null;
  return isPinnableRoute(value, extensionHrefs) ? value : null;
}

/**
 * Synchronous pin read for route loaders (no React context). Reads the key
 * `usePinnedHomeRoute` writes; `useLocalStorage` JSON-serializes values, so
 * parse defensively and treat anything unreadable as "no pin". An invalid
 * pin is ignored, not cleared — it may become valid again (e.g. the
 * automations interface returning after a backend switch back). Without a
 * live page set this deliberately rejects Canvas Extension hrefs: whether
 * such a pin resolves is only known once the extension runtime loads, so
 * the index route performs that redirect from React state instead.
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

/**
 * Pin state for the home route: `/` redirects to the pinned sidebar page.
 * `extensionHrefs` names the Canvas Extension pages that are enabled right
 * now; an extension pin is only valid while its page is in the set, so a
 * disabled or uninstalled extension falls back to the default home (#16995).
 */
export function usePinnedHomeRoute(extensionHrefs?: ReadonlySet<string>) {
  const active = useActiveBackend();
  const [rawPinnedRoute, setRawPinnedRoute] = useLocalStorage<string | null>(
    getPinnedHomeRouteKey(active.backend.id, active.orgId),
    null,
  );

  const pinnedRoute = useMemo(
    () => sanitizePinnedRoute(rawPinnedRoute, extensionHrefs),
    [rawPinnedRoute, extensionHrefs],
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
      if (!isPinnableRoute(path, extensionHrefs)) return;
      setRawPinnedRoute(path);
    },
    [pinnedRoute, setRawPinnedRoute, extensionHrefs],
  );

  return {
    pinnedRoute,
    isPinnedRoute,
    togglePinnedRoute,
  };
}
