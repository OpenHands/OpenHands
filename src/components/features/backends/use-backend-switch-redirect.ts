import React from "react";
import { useNavigation } from "#/context/navigation-context";
import { useActiveBackendContext } from "#/contexts/active-backend-context";

/** The section list a backend-scoped detail page falls back to, if any. */
function getBackendScopedListPath(path: string): string | null {
  if (/^\/automations\/[^/]+/.test(path)) return "/automations";
  if (/^\/conversations\/[^/]+/.test(path)) return "/conversations";
  return null;
}

/**
 * @spec BM-002 — Switching the active backend (or org) makes a
 * backend-scoped detail page the user is viewing belong to the previous
 * backend. Redirect to that section's list so they never see stale data,
 * mirroring the switch-backend redirect in BackendSelector. Call it in the
 * same handler that changes the active backend.
 */
export function useBackendSwitchRedirect() {
  const { currentPath, navigate } = useNavigation();
  return React.useCallback(() => {
    const listPath = getBackendScopedListPath(currentPath);
    if (listPath) navigate(listPath);
  }, [currentPath, navigate]);
}

// Navigation is asynchronous (the destination route module may still be
// loading), so this bounds how long an activation waits for a host whose
// `navigate` never leaves the detail page.
const ACTIVATION_REDIRECT_TIMEOUT_MS = 5000;

/**
 * @spec BM-002 — Activate a backend only after the user has left a detail page
 * that belongs to the previous backend.
 *
 * Changing the active backend is synchronous, but the redirect to the section
 * list is not: until the router commits it, the detail page stays mounted with
 * an id the newly active backend does not own, and any request issued in that
 * gap is sent to the wrong backend. Redirect first, and activate from an effect
 * that runs once the committed route is no longer a detail page.
 *
 * `onActivated` runs right after the backend becomes active.
 */
export function useActivateBackendAfterRedirect(onActivated: () => void) {
  const { currentPath, navigate } = useNavigation();
  const { setActive } = useActiveBackendContext();
  const [pendingBackendId, setPendingBackendId] = React.useState<string | null>(
    null,
  );
  const listPath = getBackendScopedListPath(currentPath);

  const finish = React.useCallback(
    (backendId: string) => {
      setPendingBackendId(null);
      setActive(backendId);
      onActivated();
    },
    [setActive, onActivated],
  );

  React.useEffect(() => {
    if (pendingBackendId === null) return undefined;
    if (!listPath) {
      finish(pendingBackendId);
      return undefined;
    }
    const timeout = setTimeout(
      () => finish(pendingBackendId),
      ACTIVATION_REDIRECT_TIMEOUT_MS,
    );
    return () => clearTimeout(timeout);
  }, [pendingBackendId, listPath, finish]);

  const activate = React.useCallback(
    (backendId: string) => {
      if (!listPath) {
        finish(backendId);
        return;
      }
      setPendingBackendId(backendId);
      navigate(listPath);
    },
    [listPath, finish, navigate],
  );

  return { activate, isActivationPending: pendingBackendId !== null };
}
