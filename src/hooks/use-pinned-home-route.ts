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
 * Prefix for Canvas Extension pages. Matches the form produced by
 * `buildCanvasExtensionPageHref` in
 * `src/components/features/canvas-extensions/canvas-extensions-runtime.tsx`
 * (e.g. `/extensions/demo-extension/some-page`). A single constant
 * keeps the prefix in sync with the producer; importing from the
 * runtime module would create a hook-shaped import in this file.
 */
export const EXTENSION_PATH_PREFIX = "/extensions/";

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
 * Canvas Extension pages (built by `buildCanvasExtensionPageHref` as
 * `/extensions/<name>/<contribution>`) are pinnable when the extension
 * runtime is currently rendering them, so the pin affordance only
 * appears on links that can actually be resolved.
 */
export function isPinnableRoute(path: string): boolean {
  if (path === CUSTOMIZE_PATH) return true;
  if (path === automationListPath()) return hasAutomationInterface();
  if (path.startsWith(EXTENSION_PATH_PREFIX)) return true;
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

/**
 * Extracts the extension name from a `/extensions/<name>/...` path.
 */
export function getExtensionNameFromPath(path: string): string | null {
  if (!path.startsWith(EXTENSION_PATH_PREFIX)) return null;
  const remainder = path.slice(EXTENSION_PATH_PREFIX.length);
  const segment = remainder.split("/")[0];
  return segment ? decodeURIComponent(segment) : null;
}

/**
 * Clear the stored pinned home route if it points to a page under the
 * specified extension name. Used during disable/uninstall flows so visiting
 * `/` falls back to the default home screen.
 */
export function clearPinnedExtensionRoute(
  extensionName: string,
  backendId?: string,
  orgId?: string | null,
): void {
  const active = getActiveBackend();
  const targetBackendId =
    backendId ?? (isNoBackend(active.backend) ? undefined : active.backend.id);
  if (!targetBackendId) return;
  const targetOrgId = orgId !== undefined ? orgId : active.orgId;
  const key = getPinnedHomeRouteKey(targetBackendId, targetOrgId);
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return;
    const path = JSON.parse(raw);
    if (
      typeof path === "string" &&
      getExtensionNameFromPath(path) === extensionName
    ) {
      window.localStorage.removeItem(key);
    }
  } catch {
    // Ignore invalid JSON / storage errors
  }
}

/**
 * Clear the stored pinned home route if it points to a Canvas Extension page
 * whose extension is not in the list of currently enabled extensions.
 */
export function clearStalePinnedExtensionRoutes(
  backendId: string,
  orgId: string | null,
  enabledExtensionNames: readonly string[],
): void {
  const key = getPinnedHomeRouteKey(backendId, orgId);
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return;
    const path = JSON.parse(raw);
    if (typeof path === "string" && path.startsWith(EXTENSION_PATH_PREFIX)) {
      const extName = getExtensionNameFromPath(path);
      if (extName && !enabledExtensionNames.includes(extName)) {
        window.localStorage.removeItem(key);
      }
    }
  } catch {
    // Ignore storage errors
  }
}
