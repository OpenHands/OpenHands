import React from "react";

const ESCAPE_KEY = "Escape";

/**
 * Close a menu or popover when Escape is pressed anywhere while it is open,
 * then return focus to its trigger. Listening on the document (rather than on
 * the popup) covers focus that is still on the trigger or has moved past a
 * portaled menu with Tab. Pairs with `useClickOutsideElement`.
 */
export const useCloseOnEscape = (
  isOpen: boolean,
  onClose: () => void,
  returnFocusRef?: React.RefObject<HTMLElement | null>,
) => {
  // Hold the latest callback in a ref so callers can pass inline closures.
  const onCloseRef = React.useRef(onClose);
  React.useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  React.useEffect(() => {
    if (!isOpen) return undefined;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== ESCAPE_KEY || event.defaultPrevented) return;
      event.preventDefault();
      onCloseRef.current();
      returnFocusRef?.current?.focus();
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, returnFocusRef]);
};
