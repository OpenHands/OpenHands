import React from "react";
import { clampLeftToViewport } from "#/hooks/use-popover-fixed-placement";

export interface SubmenuPlacementOptions {
  triggerRect: {
    top: number;
    left: number;
    right: number;
    bottom: number;
    width?: number;
    height?: number;
  };
  submenuRect: {
    width: number;
    height: number;
  };
  defaultOffset?: { x: number; y: number };
  gutter?: number;
  viewport?: { width: number; height: number };
}

export interface SubmenuPlacementStyle {
  left: number;
  top: number;
}

/**
 * Calculates local (left, top) offsets for an absolutely-positioned submenu
 * relative to its trigger container, ensuring it stays fully inside the viewport.
 */
export function calculateSubmenuPlacement({
  triggerRect,
  submenuRect,
  defaultOffset = { x: 1, y: -4 },
  gutter = 8,
  viewport,
}: SubmenuPlacementOptions): SubmenuPlacementStyle {
  const vpWidth =
    viewport?.width ??
    (typeof window !== "undefined" ? window.innerWidth : 1024);
  const vpHeight =
    viewport?.height ??
    (typeof window !== "undefined" ? window.innerHeight : 768);

  const submenuWidth = submenuRect.width;
  const submenuHeight = submenuRect.height;

  // 1. Horizontal placement
  const preferredRightPos = triggerRect.right + defaultOffset.x;
  const fitsRight = preferredRightPos + submenuWidth <= vpWidth - gutter;
  const preferredLeftPos = triggerRect.left - defaultOffset.x - submenuWidth;
  const fitsLeft = preferredLeftPos >= gutter;

  let targetLeft: number;
  if (fitsRight) {
    targetLeft = preferredRightPos;
  } else if (fitsLeft) {
    targetLeft = preferredLeftPos;
  } else {
    // Neither side fits outside the trigger without overflowing viewport.
    // Clamp so the submenu stays within the viewport.
    targetLeft = clampLeftToViewport(
      preferredRightPos,
      submenuWidth,
      gutter,
      vpWidth,
    );
  }

  // 2. Vertical placement
  const preferredTopPos = triggerRect.top + defaultOffset.y;
  const overflowsBelow = preferredTopPos + submenuHeight > vpHeight - gutter;

  let targetTop: number;
  if (overflowsBelow) {
    targetTop = vpHeight - gutter - submenuHeight;
    targetTop = Math.max(gutter, targetTop);
  } else {
    targetTop = Math.max(gutter, preferredTopPos);
  }

  return {
    left: Math.round(targetLeft - triggerRect.left),
    top: Math.round(targetTop - triggerRect.top),
  };
}

export interface UseSubmenuPlacementOptions {
  isOpen?: boolean;
  defaultOffset?: { x: number; y: number };
  gutter?: number;
}

export interface UseSubmenuPlacementResult {
  style: React.CSSProperties | undefined;
  updatePlacement: () => void;
}

/**
 * Hook to position an absolutely-positioned submenu so that it stays within the
 * viewport on narrow screens or near viewport boundaries.
 */
export function useSubmenuPlacement(
  triggerRef: React.RefObject<HTMLElement | null>,
  submenuRef: React.RefObject<HTMLElement | null>,
  options?: UseSubmenuPlacementOptions,
): UseSubmenuPlacementResult {
  const {
    isOpen = false,
    defaultOffset = { x: 1, y: -4 },
    gutter = 8,
  } = options ?? {};
  const [style, setStyle] = React.useState<React.CSSProperties | undefined>();

  const updatePlacement = React.useCallback(() => {
    const triggerEl = triggerRef.current;
    const submenuEl = submenuRef.current;
    if (!triggerEl || !submenuEl) return;

    const triggerRect = triggerEl.getBoundingClientRect();
    const submenuRect = submenuEl.getBoundingClientRect();

    const width = submenuRect.width || submenuEl.offsetWidth || 220;
    const height = submenuRect.height || submenuEl.offsetHeight || 200;

    const placement = calculateSubmenuPlacement({
      triggerRect,
      submenuRect: { width, height },
      defaultOffset,
      gutter,
    });

    setStyle({
      left: `${placement.left}px`,
      top: `${placement.top}px`,
      marginLeft: 0,
    });
  }, [triggerRef, submenuRef, defaultOffset.x, defaultOffset.y, gutter]);

  React.useLayoutEffect(() => {
    if (!isOpen) {
      setStyle(undefined);
      return undefined;
    }

    updatePlacement();
    window.addEventListener("resize", updatePlacement);
    window.addEventListener("scroll", updatePlacement, true);

    return () => {
      window.removeEventListener("resize", updatePlacement);
      window.removeEventListener("scroll", updatePlacement, true);
    };
  }, [isOpen, updatePlacement]);

  return { style, updatePlacement };
}
