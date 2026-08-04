import type { Backend, BackendSelection, ResolvedActiveBackend } from "./types";

/**
 * Query parameters that pin a link to the backend (and cloud org) that owns
 * the linked resource.
 *
 * The active backend is tab-scoped: `readStoredActiveBackend()` prefers
 * `sessionStorage` and only falls back to `localStorage`. A tab opened with
 * cmd/ctrl-click or middle-click does not reliably inherit the opener's
 * `sessionStorage`, so it can boot from the `localStorage` fallback — which
 * holds whichever backend was selected *last in any tab*, not the one the
 * sidebar we were just looking at belongs to. The new tab then resolves the
 * conversation id against the wrong backend and shows "conversation not
 * found". Carrying the identity in the URL makes the link self-describing so
 * the new tab pins to the right backend regardless of what storage holds.
 */
export const BACKEND_QUERY_PARAM = "backend";
export const ORG_QUERY_PARAM = "org";

/**
 * Append the active backend identity to an in-app path so opening it in a new
 * browsing context resolves against the same backend.
 */
export function withBackendSelectionParams(
  path: string,
  active: ResolvedActiveBackend,
): string {
  const { backend, orgId } = active;
  if (!backend.id) return path;

  const [pathname, existingSearch = ""] = path.split("?");
  const params = new URLSearchParams(existingSearch);
  params.set(BACKEND_QUERY_PARAM, backend.id);
  if (orgId) params.set(ORG_QUERY_PARAM, orgId);

  return `${pathname}?${params.toString()}`;
}

/**
 * Read a backend selection off the current URL, keeping it only when it names
 * a registered backend. An unknown id (a link from another machine, or a
 * backend that has since been removed) is ignored so the caller falls back to
 * the stored selection.
 */
export function readBackendSelectionFromUrl(
  backends: Backend[],
  search: string,
): BackendSelection | null {
  if (!search) return null;

  let params: URLSearchParams;
  try {
    params = new URLSearchParams(search);
  } catch {
    return null;
  }

  const backendId = params.get(BACKEND_QUERY_PARAM);
  if (!backendId) return null;
  if (!backends.some((backend) => backend.id === backendId)) return null;

  const orgId = params.get(ORG_QUERY_PARAM);
  return { backendId, orgId: orgId || null };
}

/** The current tab's query string, or "" outside a browser. */
export function currentLocationSearch(): string {
  if (typeof window === "undefined") return "";
  return window.location.search;
}
