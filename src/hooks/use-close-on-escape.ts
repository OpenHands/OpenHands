import React from "react";

const ESCAPE_KEY = "Escape";

interface EscapeLayer {
  close: () => void;
}

/**
 * Open layers, oldest first. Escape closes only the last one, which is the
 * layer the user opened most recently and so the one on top (for example the
 * command menu over a card menu that was left open).
 */
const openLayers: EscapeLayer[] = [];

const handleDocumentKeyDown = (event: KeyboardEvent) => {
  if (event.key !== ESCAPE_KEY || event.defaultPrevented) return;
  const topLayer = openLayers[openLayers.length - 1];
  if (!topLayer) return;
  event.preventDefault();
  topLayer.close();
};

const pushLayer = (layer: EscapeLayer) => {
  if (openLayers.length === 0) {
    document.addEventListener("keydown", handleDocumentKeyDown);
  }
  openLayers.push(layer);
};

const removeLayer = (layer: EscapeLayer) => {
  const index = openLayers.indexOf(layer);
  if (index !== -1) openLayers.splice(index, 1);
  if (openLayers.length === 0) {
    document.removeEventListener("keydown", handleDocumentKeyDown);
  }
};

/**
 * Close a menu, popover or drawer when Escape is pressed anywhere while it is
 * open, then return focus to its trigger. Listening on the document (rather
 * than on the popup) covers focus that is still on the trigger or has moved
 * past a portaled menu with Tab. When several layers are open, one Escape
 * closes only the most recently opened one. An Escape whose default was
 * already prevented (handled by something inside the layer) is ignored.
 * Pairs with `useClickOutsideElement`.
 */
export const useCloseOnEscape = (
  isOpen: boolean,
  onClose: () => void,
  returnFocusRef?: React.RefObject<HTMLElement | null>,
) => {
  // Hold the latest values in refs so callers can pass inline closures
  // without moving their layer to the top of the stack on every render.
  const onCloseRef = React.useRef(onClose);
  const returnFocusRefRef = React.useRef(returnFocusRef);
  React.useEffect(() => {
    onCloseRef.current = onClose;
    returnFocusRefRef.current = returnFocusRef;
  });

  React.useEffect(() => {
    if (!isOpen) return undefined;

    const layer: EscapeLayer = {
      close: () => {
        onCloseRef.current();
        returnFocusRefRef.current?.current?.focus();
      },
    };

    pushLayer(layer);
    return () => removeLayer(layer);
  }, [isOpen]);
};
