import React from "react";
import { describe, expect, it } from "vitest";
import { renderHook, act } from "@testing-library/react";
import {
  calculateSubmenuPlacement,
  useSubmenuPlacement,
} from "#/hooks/use-submenu-placement";

describe("calculateSubmenuPlacement", () => {
  const defaultOffset = { x: 1, y: -4 };
  const gutter = 8;

  it("opens to the right and anchors to the top when room is available (desktop)", () => {
    // Parent trigger row at x: 200..400, y: 300..336 on a 1920x1080 viewport
    const triggerRect = {
      left: 200,
      right: 400,
      top: 300,
      bottom: 336,
      width: 200,
      height: 36,
    };
    const submenuRect = { width: 220, height: 180 };
    const viewport = { width: 1920, height: 1080 };

    const placement = calculateSubmenuPlacement({
      triggerRect,
      submenuRect,
      defaultOffset,
      gutter,
      viewport,
    });

    // Submenu target viewport coords:
    // left: triggerRect.right + 1 = 401
    // top: triggerRect.top - 4 = 296
    // Local coords relative to triggerRect:
    // local left = 401 - 200 = 201 (equivalent to left-full + 1px)
    // local top = 296 - 300 = -4 (equivalent to top-[-4px])
    expect(placement).toEqual({
      left: 201,
      top: -4,
    });
  });

  it("flips to the left of the trigger when overflowing the right but fitting on the left", () => {
    // Parent trigger row at x: 500..700 on a 800x600 viewport
    const triggerRect = {
      left: 500,
      right: 700,
      top: 200,
      bottom: 236,
      width: 200,
      height: 36,
    };
    const submenuRect = { width: 220, height: 180 };
    const viewport = { width: 800, height: 600 };

    const placement = calculateSubmenuPlacement({
      triggerRect,
      submenuRect,
      defaultOffset,
      gutter,
      viewport,
    });

    // Opening right would reach 700 + 1 + 220 = 921 > 800 - 8 (792) -> overflows right
    // Opening left reaches 500 - 1 - 220 = 279 >= 8 -> fits left
    // Local coords: 279 - 500 = -221
    expect(placement.left).toBe(-221);
    expect(placement.top).toBe(-4);
  });

  it("clamps within viewport when neither right nor left fits outside the parent (320px narrow viewport)", () => {
    // On a 320px viewport, overflow menu is at x: 68..268, submenu width is 220
    const triggerRect = {
      left: 68,
      right: 268,
      top: 575,
      bottom: 611,
      width: 200,
      height: 36,
    };
    const submenuRect = { width: 220, height: 190 };
    const viewport = { width: 320, height: 700 };

    const placement = calculateSubmenuPlacement({
      triggerRect,
      submenuRect,
      defaultOffset,
      gutter,
      viewport,
    });

    // Right would reach 268 + 1 + 220 = 489 > 312
    // Left would reach 68 - 1 - 220 = -153 < 8
    // Clamped target left in viewport = 320 - 8 - 220 = 92
    // Local left = 92 - 68 = 24
    expect(placement.left).toBe(24);

    // Viewport right edge of the submenu: 68 + 24 + 220 = 312 <= 320 - 8
    const viewportLeft = triggerRect.left + placement.left;
    const viewportRight = viewportLeft + submenuRect.width;
    expect(viewportLeft).toBeGreaterThanOrEqual(gutter);
    expect(viewportRight).toBeLessThanOrEqual(viewport.width - gutter);

    // Vertical placement:
    // Top pos = 575 - 4 = 571
    // Bottom pos = 571 + 190 = 761 > 700 - 8 (692) -> overflows bottom
    // Target top in viewport = 700 - 8 - 190 = 502
    // Local top = 502 - 575 = -73
    expect(placement.top).toBe(-73);

    // Viewport bottom edge of the submenu: 575 - 73 + 190 = 692 <= 700 - 8
    const viewportTop = triggerRect.top + placement.top;
    const viewportBottom = viewportTop + submenuRect.height;
    expect(viewportTop).toBeGreaterThanOrEqual(gutter);
    expect(viewportBottom).toBeLessThanOrEqual(viewport.height - gutter);
  });

  it("keeps a tall submenu from overflowing bottom at 360x800", () => {
    // 3 profiles with longer names: submenu height is 244
    const triggerRect = {
      left: 68,
      right: 268,
      top: 675,
      bottom: 711,
      width: 200,
      height: 36,
    };
    const submenuRect = { width: 220, height: 244 };
    const viewport = { width: 360, height: 800 };

    const placement = calculateSubmenuPlacement({
      triggerRect,
      submenuRect,
      defaultOffset,
      gutter,
      viewport,
    });

    const viewportTop = triggerRect.top + placement.top;
    const viewportBottom = viewportTop + submenuRect.height;
    const viewportLeft = triggerRect.left + placement.left;
    const viewportRight = viewportLeft + submenuRect.width;

    expect(viewportLeft).toBeGreaterThanOrEqual(gutter);
    expect(viewportRight).toBeLessThanOrEqual(viewport.width - gutter);
    expect(viewportTop).toBeGreaterThanOrEqual(gutter);
    expect(viewportBottom).toBeLessThanOrEqual(viewport.height - gutter);
  });
});

describe("useSubmenuPlacement", () => {
  it("returns undefined style when not open", () => {
    const triggerEl = document.createElement("div");
    const submenuEl = document.createElement("div");
    const triggerRef = { current: triggerEl };
    const submenuRef = { current: submenuEl };

    const { result } = renderHook(() =>
      useSubmenuPlacement(triggerRef, submenuRef, { isOpen: false }),
    );

    expect(result.current.style).toBeUndefined();
  });

  it("calculates style when open and clamps to viewport", () => {
    const triggerEl = document.createElement("div");
    const submenuEl = document.createElement("div");
    const triggerRef = { current: triggerEl };
    const submenuRef = { current: submenuEl };

    // Stub layout rects
    triggerEl.getBoundingClientRect = () =>
      DOMRect.fromRect({ x: 68, y: 575, width: 200, height: 36 });
    submenuEl.getBoundingClientRect = () =>
      DOMRect.fromRect({ x: 0, y: 0, width: 220, height: 190 });

    // Set viewport dimensions
    window.innerWidth = 320;
    window.innerHeight = 700;

    const { result } = renderHook(() =>
      useSubmenuPlacement(triggerRef, submenuRef, { isOpen: true }),
    );

    expect(result.current.style).toEqual({
      left: "24px",
      top: "-73px",
      marginLeft: 0,
    });
  });

  it("updates placement via updatePlacement callback", () => {
    const triggerEl = document.createElement("div");
    const submenuEl = document.createElement("div");
    const triggerRef = { current: triggerEl };
    const submenuRef = { current: submenuEl };

    triggerEl.getBoundingClientRect = () =>
      DOMRect.fromRect({ x: 200, y: 300, width: 200, height: 36 });
    submenuEl.getBoundingClientRect = () =>
      DOMRect.fromRect({ x: 0, y: 0, width: 220, height: 180 });

    window.innerWidth = 1920;
    window.innerHeight = 1080;

    const { result } = renderHook(() =>
      useSubmenuPlacement(triggerRef, submenuRef, { isOpen: false }),
    );

    expect(result.current.style).toBeUndefined();

    act(() => {
      result.current.updatePlacement();
    });

    expect(result.current.style).toEqual({
      left: "201px",
      top: "-4px",
      marginLeft: 0,
    });
  });
});
