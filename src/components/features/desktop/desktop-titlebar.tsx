import type { CSSProperties } from "react";

/**
 * Empty macOS drag strip under the hidden-inset traffic lights.
 *
 * Interactive chrome lives below this row so close/minimize/zoom cannot
 * overlap the sidebar logo or conversation header. Height matches
 * `MAC_TITLEBAR_HEIGHT_PX` in electron/lib/window-chrome.mjs.
 *
 * The drag region ships as the Tailwind arbitrary property
 * `[-webkit-app-region:drag]` rather than an inline style: jsdom drops
 * unknown `-webkit-*` inline properties, so the class keeps Electron's
 * window-drag working while remaining test-assertable.
 */
export function DesktopTitlebar() {
  return (
    <div
      data-testid="desktop-titlebar"
      aria-hidden="true"
      className="h-10 w-full shrink-0 select-none bg-base [-webkit-app-region:drag]"
      style={{ WebkitAppRegion: "drag" } as CSSProperties}
    />
  );
}
