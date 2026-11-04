import React from "react";
import {
  isDesktopFullScreen,
  subscribeDesktopFullScreen,
} from "#/utils/desktop-shell";

/**
 * Whether the desktop window is in native fullscreen. Always false in a
 * browser tab.
 */
export function useDesktopFullScreen(): boolean {
  return React.useSyncExternalStore(
    subscribeDesktopFullScreen,
    isDesktopFullScreen,
    () => false,
  );
}
