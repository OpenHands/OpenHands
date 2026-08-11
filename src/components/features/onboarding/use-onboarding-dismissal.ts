import React from "react";

export const ONBOARDING_DISMISSED_SESSION_KEY_PREFIX =
  "openhands-onboarding-dismissed";

function getDismissalStorageKey(backendId: string): string {
  return `${ONBOARDING_DISMISSED_SESSION_KEY_PREFIX}:${backendId}`;
}

function readDismissedFromSession(backendId: string): boolean {
  if (typeof window === "undefined") return false;

  try {
    return (
      window.sessionStorage.getItem(getDismissalStorageKey(backendId)) !== null
    );
  } catch {
    return false;
  }
}

/**
 * Tracks an intentional "Skip for now" dismissal for the active backend.
 * The dismissal survives refreshes in the current browser session, but it is
 * neither shared with another backend nor persisted as backend readiness.
 */
export function useOnboardingDismissal(backendId: string) {
  const [state, setState] = React.useState(() => ({
    backendId,
    isDismissed: readDismissedFromSession(backendId),
  }));

  const isDismissed =
    state.backendId === backendId
      ? state.isDismissed
      : readDismissedFromSession(backendId);

  React.useEffect(() => {
    if (state.backendId === backendId) return;
    setState({
      backendId,
      isDismissed: readDismissedFromSession(backendId),
    });
  }, [backendId, state.backendId]);

  const markDismissed = React.useCallback(() => {
    try {
      window.sessionStorage.setItem(getDismissalStorageKey(backendId), "1");
    } catch {
      // Best effort; the in-memory state still dismisses the current flow.
    }
    setState({ backendId, isDismissed: true });
  }, [backendId]);

  return { isDismissed, markDismissed } as const;
}
