import React from "react";
import { useNavigation } from "#/context/navigation-context";

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
    if (/^\/automations\/[^/]+/.test(currentPath)) navigate("/automations");
    else if (/^\/conversations\/[^/]+/.test(currentPath))
      navigate("/conversations");
  }, [currentPath, navigate]);
}
