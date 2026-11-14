import React from "react";
import { useSidebarStore } from "#/stores/sidebar-store";

const TOGGLE_SIDEBAR_KEY = "s";

/**
 * Toggle the desktop conversation sidebar with Ctrl/Cmd+Shift+S from anywhere
 * in the app, including while the chat composer has focus.
 *
 * Listening on `window` and accepting either modifier follows the existing
 * Ctrl/Cmd+K handling in the command menu, which is what makes the shortcut
 * work regardless of what is focused. The toggle itself goes through the same
 * store action as the click chevron and the command menu's "Toggle sidebar"
 * row, so the collapsed state keeps persisting to localStorage and the mobile
 * drawer is untouched. Alt is excluded so the binding does not steal a
 * neighbouring OS or browser combination.
 */
export const useToggleSidebarShortcut = () => {
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const isPlatformModifier = event.metaKey || event.ctrlKey;
      if (
        !isPlatformModifier ||
        !event.shiftKey ||
        event.altKey ||
        event.key.toLocaleLowerCase() !== TOGGLE_SIDEBAR_KEY
      ) {
        return;
      }

      event.preventDefault();
      // Holding the combination repeats keydown; one press is one toggle.
      if (event.repeat) return;

      useSidebarStore.getState().toggleCollapsed();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);
};
