import React from "react";
import { useSidebarStore } from "#/stores/sidebar-store";
import {
  SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH,
  useBreakpoint,
} from "#/hooks/use-breakpoint";

const TOGGLE_SIDEBAR_KEY = "e";
const TOGGLE_SIDEBAR_CODE = "KeyE";
/**
 * An open dialog owns the keyboard: collapsing the sidebar behind the command
 * menu (or any modal) changes nothing the user can see, and the sidebar is
 * usually what that dialog was opened from.
 */
const OPEN_DIALOG_SELECTOR = '[role="dialog"][aria-modal="true"]';

/**
 * The binding as `aria-keyshortcuts` spells it, for the controls that perform
 * the same action: space-separated, one entry per platform modifier.
 */
export const TOGGLE_SIDEBAR_ARIA_SHORTCUTS = "Control+Shift+E Meta+Shift+E";

/**
 * `code` keeps this independent of the keyboard layout: on a Cyrillic layout
 * the combination arrives as `key: "У"` with `code: "KeyE"`. Events built
 * without a meaningful code still match on the character.
 */
function isToggleSidebarKey(event: KeyboardEvent) {
  return (
    event.code === TOGGLE_SIDEBAR_CODE ||
    event.key.toLocaleLowerCase() === TOGGLE_SIDEBAR_KEY
  );
}

/**
 * Toggle the desktop conversation sidebar with Ctrl/Cmd+Shift+E from anywhere
 * in the app, including while the chat composer has focus.
 *
 * Listening on `window` and accepting either modifier follows the existing
 * Ctrl/Cmd+K handling in the command menu, which is what makes the shortcut
 * work regardless of what is focused. The toggle itself goes through the same
 * store action as the click chevron and the command menu's "Toggle sidebar"
 * row, so the collapsed state keeps persisting to localStorage and the mobile
 * drawer is untouched.
 *
 * `E` rather than `S`: Ctrl/Cmd+Shift+S is Firefox's Screenshots shortcut and
 * Edge's screenshot tool, and neither is reserved, so claiming it would take
 * the browser feature away on every Canvas page. Alt is excluded for the same
 * reason, so neighbouring OS and IME combinations stay with the browser.
 */
export const useToggleSidebarShortcut = () => {
  // Below `md` the rail is `max-md:hidden` and the drawer always renders
  // expanded, so toggling there would only persist a state the user never saw
  // collapse on screen.
  const isRailHidden = useBreakpoint(SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH);

  React.useEffect(() => {
    if (isRailHidden) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        !(event.metaKey || event.ctrlKey) ||
        !event.shiftKey ||
        event.altKey ||
        !isToggleSidebarKey(event) ||
        document.querySelector(OPEN_DIALOG_SELECTOR) !== null
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
  }, [isRailHidden]);
};
