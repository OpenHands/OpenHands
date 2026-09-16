import React from "react";
import { subscribeDesktopFullScreen } from "#/utils/desktop-shell";

/**
 * Whether the desktop window is in native fullscreen. Always false in a
 * browser tab.
 */
export function useDesktopFullScreen(): boolean {
  const [isFullScreen, setIsFullScreen] = React.useState(false);

  React.useEffect(() => subscribeDesktopFullScreen(setIsFullScreen), []);

  return isFullScreen;
}
