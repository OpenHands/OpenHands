import { LOCKED_CLOUD_BACKEND_ID } from "./default-backend";
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
 * browsing context resolves against the same backend. Any fragment on the
 * path is preserved verbatim and kept after the query string.
 */
export function withBackendSelectionParams(
  path: string,
  active: ResolvedActiveBackend,
): string {
  const { backend, orgId } = active;
  if (!backend.id) return path;

  const hashIndex = path.indexOf("#");
  const fragment = hashIndex === -1 ? "" : path.slice(hashIndex);
  const withoutFragment = hashIndex === -1 ? path : path.slice(0, hashIndex);

  // Split at the *first* `?` only. A later `?` is ordinary query data (a
  // `next=` redirect carrying its own query, say), and `split("?")` would
  // silently drop everything past it.
  const queryIndex = withoutFragment.indexOf("?");
  const pathname =
    queryIndex === -1 ? withoutFragment : withoutFragment.slice(0, queryIndex);
  const existingSearch =
    queryIndex === -1 ? "" : withoutFragment.slice(queryIndex + 1);

  const params = new URLSearchParams(existingSearch);
  params.set(BACKEND_QUERY_PARAM, backend.id);
  if (orgId) params.set(ORG_QUERY_PARAM, orgId);

  return `${pathname}?${params.toString()}${fragment}`;
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

/**
 * Read the org the cloud's own web app hands back when it links here. Its
 * "Back to App" link carries only `?org=<id>` — the parameter Canvas puts on
 * its links into the cloud's Settings — because Canvas locked to that cloud
 * host has a single backend. Anything else (no org, a backend named too, or
 * no locked cloud backend) is left to the other readers.
 */
export function readLockedCloudOrgFromUrl(
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
  if (params.has(BACKEND_QUERY_PARAM)) return null;
  const orgId = params.get(ORG_QUERY_PARAM);
  if (!orgId) return null;
  if (!backends.some((backend) => backend.id === LOCKED_CLOUD_BACKEND_ID)) {
    return null;
  }
  return { backendId: LOCKED_CLOUD_BACKEND_ID, orgId };
}

/** Drop `?org=` from the address bar, keeping every other part of the URL. */
export function removeOrgFromCurrentUrl(): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (!url.searchParams.has(ORG_QUERY_PARAM)) return;
  url.searchParams.delete(ORG_QUERY_PARAM);
  window.history.replaceState(
    window.history.state,
    "",
    `${url.pathname}${url.search}${url.hash}`,
  );
}

/** The current tab's query string, or "" outside a browser. */
export function currentLocationSearch(): string {
  if (typeof window === "undefined") return "";
  return window.location.search;
}
