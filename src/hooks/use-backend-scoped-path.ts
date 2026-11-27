import React from "react";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { withBackendSelectionParams } from "#/api/backend-registry/url-selection";
import { useActiveBackend } from "#/contexts/active-backend-context";

/**
 * Build in-app paths that carry the active backend identity.
 *
 * Use it for links a user can plausibly open in a *new* browsing context
 * (cmd/ctrl-click, middle click, "Open link in new tab"). Such a tab does not
 * reliably inherit the opener's `sessionStorage`, so without the pinned
 * identity it can boot on whichever backend `localStorage` last recorded and
 * fail to resolve the linked conversation.
 */
export function useBackendScopedPath(): (path: string) => string {
  const active = useActiveBackend();

  return React.useCallback(
    (path: string) => {
      // The sentinel isn't a real backend and must never be pinned onto a
      // link — a new tab would only ignore it and fall back to storage.
      if (isNoBackend(active.backend)) return path;
      return withBackendSelectionParams(path, active);
    },
    [active],
  );
}
