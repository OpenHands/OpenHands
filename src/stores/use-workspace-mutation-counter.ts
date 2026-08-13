import { create } from "zustand";

/**
 * Monotonic counter that ticks every time the agent commits a file-editor
 * mutation in the workspace. Serves two purposes:
 *
 *   1. It drives the `?v=<count>` cache-buster appended to the static
 *      workspace fileserver URLs used by `<iframe src>` / `<img src>` for
 *      the rich preview, so the browser re-requests a fresh copy after
 *      each edit. This matters because the rendered HTML may reference
 *      sibling assets (CSS, images) that the user can't see directly but
 *      expects to reflect the latest workspace state.
 *   2. It is still bumped by {@link useAutoRefreshFilesOnEdit}, which
 *      invalidates the `workspace-file-content` query to refresh the decoded
 *      file body. Keeping the counter out of the query key means a rapid
 *      series of edits doesn't repeatedly reset the selected file's query
 *      (and its `isLoading` state) while a fetch is still in flight.
 *
 * Consumers:
 *   - {@link useAutoRefreshFilesOnEdit} bumps this on each mutation event.
 *   - `FileContentViewer` / files-tab "open in new tab" link append the
 *     count to the static URL via {@link withWorkspaceCacheBuster}.
 */
interface WorkspaceMutationCounterState {
  count: number;
  bump: () => void;
}

export const useWorkspaceMutationCounter =
  create<WorkspaceMutationCounterState>((set) => ({
    count: 0,
    bump: () => set((state) => ({ count: state.count + 1 })),
  }));

/**
 * Append the current mutation counter as a `v=<n>` query parameter so the
 * browser refetches the URL after every agent-side edit. Returns `null` if
 * the input is `null` so callers can pass through optional URLs untouched.
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
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}v=${version}`;
}
