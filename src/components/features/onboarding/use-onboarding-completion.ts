import React from "react";

/**
 * Legacy localStorage completion key retained for the locked-to-Cloud login
 * handoff. Ordinary Local/Cloud onboarding visibility derives from active
 * backend readiness and must not treat this marker as configuration state.
 */
export const ONBOARDING_COMPLETED_STORAGE_KEY = "openhands-onboarded";

function readCompletedFromStorage(): boolean {
  if (typeof window === "undefined") return true;
  try {
    return (
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY) !== null
    );
  } catch {
    // Inaccessible localStorage (private mode, SSR, …) — assume the
    // user has already onboarded so we don't loop on every render.
    return true;
  }
}

/**
 * Tracks the persistent completion signal used by the locked-to-Cloud
 * bootstrap. The hook returns `isCompleted` plus `markCompleted()`, mirrors
 * state to localStorage, and syncs it across tabs via the `storage` event.
 */
export function useOnboardingCompletion() {
  const [isCompleted, setIsCompleted] = React.useState<boolean>(() =>
    readCompletedFromStorage(),
  );

  React.useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key !== ONBOARDING_COMPLETED_STORAGE_KEY) return;
      setIsCompleted(readCompletedFromStorage());
    };
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  const markCompleted = React.useCallback(() => {
    try {
      window.localStorage.setItem(ONBOARDING_COMPLETED_STORAGE_KEY, "1");
    } catch {
      // best-effort; we still flip the in-memory flag below.
    }
    setIsCompleted(true);
  }, []);

  return { isCompleted, markCompleted } as const;
}
