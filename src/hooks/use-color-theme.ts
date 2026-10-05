import React from "react";
import {
  DEFAULT_COLOR_THEME,
  getActiveColorTheme,
  subscribeColorTheme,
} from "#/themes/color-themes";

export function useColorTheme() {
  return React.useSyncExternalStore(
    subscribeColorTheme,
    getActiveColorTheme,
    () => DEFAULT_COLOR_THEME,
  );
}
