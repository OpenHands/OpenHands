import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import { useLocalStorage } from "@uidotdev/usehooks";

const PANEL_DRAG_SHIELD_Z_INDEX = 200;

type ResizableDrawerEdge = "left" | "right";

interface UseResizableDrawerWidthOptions {
  containerRef: RefObject<HTMLElement | null>;
  defaultWidth: number;
  minWidth: number;
  maxWidth: number;
  storageKey: string;
  enabled?: boolean;
  /**
   * Which side of the container the panel is anchored to.
   * - `"right"` (default): width grows leftward from the container's right edge
   * - `"left"`: width grows rightward from the container's left edge
   */
  edge?: ResizableDrawerEdge;
}

export function useResizableDrawerWidth({
  containerRef,
  defaultWidth,
  minWidth,
  maxWidth,
  storageKey,
  enabled = true,
  edge = "right",
}: UseResizableDrawerWidthOptions) {
  const [persistedWidth, setPersistedWidth] = useLocalStorage<number>(
    storageKey,
    defaultWidth,
  );

  const clampWidth = useCallback(
    (width: number) => Math.max(minWidth, Math.min(maxWidth, width)),
    [minWidth, maxWidth],
  );

  const [drawerWidth, setDrawerWidth] = useState(() =>
    clampWidth(persistedWidth),
  );
  const [isDragging, setIsDragging] = useState(false);
  const drawerWidthRef = useRef(drawerWidth);

  drawerWidthRef.current = drawerWidth;

  const handleMouseDown = useCallback(
    (event: React.MouseEvent) => {
      if (!enabled) {
        return;
      }
      event.preventDefault();
      setIsDragging(true);
    },
    [enabled],
  );

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!isDragging || !containerRef.current) {
        return;
      }

      const containerRect = containerRef.current.getBoundingClientRect();
      const nextWidth =
        edge === "left"
          ? event.clientX - containerRect.left
          : containerRect.right - event.clientX;
      setDrawerWidth(clampWidth(nextWidth));
    },
    [clampWidth, containerRef, edge, isDragging],
  );

  const handleMouseUp = useCallback(() => {
    if (!isDragging) {
      return;
    }
    setIsDragging(false);
    setPersistedWidth(drawerWidthRef.current);
  }, [isDragging, setPersistedWidth]);

  useLayoutEffect(() => {
    if (!isDragging) {
      return undefined;
    }

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
  }, [handleMouseMove, handleMouseUp, isDragging]);

  return {
    drawerWidth,
    isDragging,
    handleMouseDown,
  };
}
