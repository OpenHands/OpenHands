import { create } from "zustand";

/**
 * Monotonic counter that ticks every time the agent commits a file-editor
 * mutation in the workspace and every time the user clicks Refresh in the
 * Files tab. Serves two purposes:
 *
 *   1. It's part of the {@link useWorkspaceFileContent} query key, so the
 *      hook refetches the selected file's body (used for text decoding /
 *      binary classification) after each edit or Refresh even when the
 *      selected path hasn't moved.
 *   2. It's appended as a `?v=<count>` cache-buster to the static
 *      workspace fileserver URLs used by `<iframe src>` / `<img src>` for
 *      the rich preview, so the browser re-requests a fresh copy of that
 *      top-level file after each edit or Refresh. Only the top-level URL
 *      changes: sibling assets the rendered HTML references (CSS, images)
 *      keep their own URLs and can still come from the browser cache.
 *
 * The count is seeded from `Date.now()` at page load rather than 0, so the
 * first `?v=` URL of a page load never matches one the browser cached
 * during an earlier load (the fileserver sends no Cache-Control, so a
 * reload would otherwise show the old bytes).
 *
 * Consumers:
 *   - {@link useAutoRefreshFilesOnEdit} bumps this on each mutation event.
 *   - The Files tab Refresh button bumps it so the open file refetches and
 *     the rich preview re-requests.
 *   - {@link useWorkspaceFileContent} reads the count via its query key so
 *     the hook refetches after each bump.
 *   - `FileContentViewer` / files-tab "open in new tab" link append the
 *     count to the static URL via {@link withWorkspaceCacheBuster}.
 */
interface WorkspaceMutationCounterState {
  count: number;
  bump: () => void;
}

export const useWorkspaceMutationCounter =
  create<WorkspaceMutationCounterState>((set) => ({
    count: Date.now(),
    bump: () => set((state) => ({ count: state.count + 1 })),
  }));

/**
 * Append the current mutation counter as a `v=<n>` query parameter so the
 * browser refetches the URL after every agent file-editor edit and every
 * Files tab Refresh. The counter is seeded from `Date.now()`, so a page
 * reload never rebuilds a URL the browser cached during an earlier load.
 * Returns `null` if the input is `null` so callers can pass through
 * optional URLs untouched.
 *
 * The counter is a *query* parameter, which only means anything for a real
 * fileserver URL. Cloud conversations serve text artifacts as base64 `data:`
 * URLs (see `useWorkspaceFileContent`); appending `?v=1` to one of those edits
 * the encoded payload, so the browser can no longer decode the document. Data
 * URLs are returned unchanged — the hook's query key already carries the
 * counter, so a mutation still produces a fresh data URL.
 */
export function withWorkspaceCacheBuster(url: string, version: number): string;
export function withWorkspaceCacheBuster(
  url: string | null,
  version: number,
): string | null;
export function withWorkspaceCacheBuster(
  url: string | null,
  version: number,
): string | null {
  if (url === null) return null;
  if (url.startsWith("data:")) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}v=${version}`;
}
