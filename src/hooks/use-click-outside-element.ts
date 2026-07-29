import React from "react";

/**
 * Hook to call a callback function when an element is clicked outside
 * @param callback The callback function to call when the element is clicked outside
 */
export const useClickOutsideElement = <T extends HTMLElement>(
  callback: () => void,
  ignoreOutsideClickRef?: React.RefObject<HTMLElement | null>,
) => {
  const ref = React.useRef<T>(null);
  // Hold the latest callback in a ref so the listener effect doesn't
  // re-register every render when callers pass an inline arrow function.
  const callbackRef = React.useRef(callback);
  React.useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (!ref.current) return;
      if (ref.current.contains(target)) return;
      if (ignoreOutsideClickRef?.current?.contains(target)) return;
      callbackRef.current();
    };

    document.addEventListener("click", handleClickOutside);

    return () => document.removeEventListener("click", handleClickOutside);
  }, [ignoreOutsideClickRef]);

  return ref;
};
