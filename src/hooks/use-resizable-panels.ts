import { useState, useRef, useCallback, useLayoutEffect } from "react";
import { useLocalStorage } from "@uidotdev/usehooks";

/** Full-screen layer above iframes so parent `document` keeps receiving drag events. */
const PANEL_DRAG_SHIELD_Z_INDEX = 200;

interface UseResizablePanelsOptions {
  defaultLeftWidth?: number;
  minLeftWidth?: number;
  maxLeftWidth?: number;
  storageKey?: string;
}

export function useResizablePanels({
  defaultLeftWidth = 50,
  minLeftWidth = 30,
  maxLeftWidth = 80,
  storageKey = "desktop-layout-panel-width",
}: UseResizablePanelsOptions = {}) {
  const [persistedWidth, setPersistedWidth] = useLocalStorage<number>(
    storageKey,
    defaultLeftWidth,
  );

  const clampWidth = useCallback(
    (width: number) => Math.max(minLeftWidth, Math.min(maxLeftWidth, width)),
    [minLeftWidth, maxLeftWidth],
  );

  // Clamp the persisted value on read so stale localStorage values from older
  // min/max bounds (or other tabs) can't push the divider out of range.
  const [leftWidth, setLeftWidth] = useState(() => clampWidth(persistedWidth));
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const mouseX = e.clientX - containerRect.left;
      const containerWidth = containerRect.width;
      const newLeftWidth = (mouseX / containerWidth) * 100;

      const clampedWidth = clampWidth(newLeftWidth);
      setLeftWidth(clampedWidth);
    },
    [isDragging, clampWidth],
  );

  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      setPersistedWidth(leftWidth);
    }
  }, [isDragging, leftWidth, setPersistedWidth]);

  useLayoutEffect(() => {
    if (!isDragging) return;

    const shield = document.createElement("div");
    shield.setAttribute("aria-hidden", "true");
    shield.dataset.panelDragShield = "";
    Object.assign(shield.style, {
      position: "fixed",
      inset: "0",
      zIndex: String(PANEL_DRAG_SHIELD_Z_INDEX),
      cursor: "ew-resize",
    });
    document.body.appendChild(shield);

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      shield.remove();
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const rightWidth = 100 - leftWidth;

  return {
    leftWidth,
    rightWidth,
    isDragging,
    containerRef,
    handleMouseDown,
  };
}
