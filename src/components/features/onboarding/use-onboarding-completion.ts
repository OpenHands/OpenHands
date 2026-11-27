import React from "react";

/**
 * localStorage key persisting whether the welcome onboarding flow has
 * been completed (or skipped). Once present, the modal won't auto-show
 * again on subsequent visits.
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
 * Tracks whether the welcome onboarding modal has been completed.
 * The hook returns the current `isCompleted` flag plus an imperative
 * `markCompleted()` callback. State is mirrored to localStorage and
 * synced across tabs via the `storage` event.
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
