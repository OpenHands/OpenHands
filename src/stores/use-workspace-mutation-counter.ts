import { create } from "zustand";

/**
 * Monotonic counter that ticks every time the agent commits a file-editor
 * mutation in the workspace. Serves two purposes:
 *
 *   1. It's part of the {@link useWorkspaceFileContent} query key, so the
 *      hook refetches the selected file's body (used for text decoding /
 *      binary classification) after each edit even when the selected path
 *      hasn't moved.
 *   2. It's appended as a `?v=<count>` cache-buster to the static
 *      workspace fileserver URLs used by `<iframe src>` / `<img src>` for
 *      the rich preview, so the browser re-requests a fresh copy after
 *      each edit — important because the rendered HTML may reference
 *      sibling assets (CSS, images) that the user can't see directly but
 *      expects to reflect the latest version of the workspace.
 *
 * Consumers:
 *   - {@link useAutoRefreshFilesOnEdit} bumps this on each mutation event.
 *   - {@link useWorkspaceFileContent} reads the count via its query key so
 *     the hook refetches after each edit.
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
