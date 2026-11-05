import React from "react";

export const TIMESTAMP_HOVER_DELAY_MS = 700;

/**
 * Hover state for event timestamp tooltips. The tooltip waits for a resting
 * pointer so quick clicks, or moving past messages and steps, don't pop it
 * over neighbouring content.
 */
export function useTimestampHover() {
  const [isVisible, setIsVisible] = React.useState(false);
  const timer = React.useRef<ReturnType<typeof setTimeout>>(undefined);

  const show = React.useCallback(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(
      () => setIsVisible(true),
      TIMESTAMP_HOVER_DELAY_MS,
    );
  }, []);

  const hide = React.useCallback(() => {
    clearTimeout(timer.current);
    setIsVisible(false);
  }, []);

  React.useEffect(() => () => clearTimeout(timer.current), []);

  return { isVisible, show, hide };
}
