import React from "react";

type VisibilityState = "visible" | "hidden";

interface UseVisibilityChangeOptions {
  /** Callback fired when visibility changes to the specified state */
  onVisibilityChange?: (state: VisibilityState) => void;
  /** Callback fired only when tab becomes visible */
  onVisible?: () => void;
  /** Callback fired only when tab becomes hidden */
  onHidden?: () => void;
  /** Whether to listen for visibility changes (default: true) */
  enabled?: boolean;
}

/**
 * Hook that listens for browser tab visibility changes.
 *
 * Useful for:
 * - Resuming operations when user returns to the tab
 * - Pausing expensive operations when tab is hidden
 * - Tracking user engagement
 *
 * @param options.onVisibilityChange - Callback with the new visibility state
 * @param options.onVisible - Callback fired only when tab becomes visible
 * @param options.onHidden - Callback fired only when tab becomes hidden
 * @param options.enabled - Whether to listen for changes (default: true)
 * @returns isVisible - Current visibility state of the tab
 */
export function useVisibilityChange({
  onVisibilityChange,
  onVisible,
  onHidden,
  enabled = true,
}: UseVisibilityChangeOptions = {}) {
  const [isVisible, setIsVisible] = React.useState(
    () => document.visibilityState === "visible",
  );

  React.useEffect(() => {
    if (!enabled) return undefined;

    const handleVisibilityChange = () => {
      const state = document.visibilityState as VisibilityState;
      setIsVisible(state === "visible");

      onVisibilityChange?.(state);

      if (state === "visible") {
        onVisible?.();
      } else {
        onHidden?.();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [enabled, onVisibilityChange, onVisible, onHidden]);

  return { isVisible };
}
