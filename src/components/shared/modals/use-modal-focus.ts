import React from "react";

const TABBABLE_CANDIDATES = [
  "a[href]",
  "area[href]",
  "button",
  "input:not([type='hidden'])",
  "select",
  "textarea",
  "iframe",
  "summary",
  "audio[controls]",
  "video[controls]",
  "[contenteditable]:not([contenteditable='false'])",
  "[tabindex]",
].join(",");

interface OpenModal {
  root: HTMLElement;
  elevated: React.RefObject<boolean>;
}

// Every mounted ModalBackdrop, in mount order. Only the topmost one owns focus.
const openModals: OpenModal[] = [];

function topmostModalRoot(): HTMLElement | null {
  // Elevated modals (the telemetry consent banner) paint above every default
  // modal regardless of mount order, so they own focus the same way.
  const topmost =
    [...openModals].reverse().find((modal) => modal.elevated.current) ??
    openModals.at(-1);
  return topmost?.root ?? null;
}

function isVisible(element: HTMLElement): boolean {
  return typeof element.checkVisibility === "function"
    ? element.checkVisibility({ visibilityProperty: true })
    : true;
}

function isRadioInput(element: Element): element is HTMLInputElement {
  return element instanceof HTMLInputElement && element.type === "radio";
}

/** Controls a user reaches with Tab inside `root`, in DOM order. */
function getTabbableElements(root: HTMLElement): HTMLElement[] {
  const tabbable = Array.from(
    root.querySelectorAll<HTMLElement>(TABBABLE_CANDIDATES),
  ).filter(
    (element) =>
      element.tabIndex >= 0 &&
      !element.matches(":disabled") &&
      !element.closest("[inert]") &&
      isVisible(element),
  );

  // A named radio group is a single Tab stop: its checked radio, else its first.
  return tabbable.filter((element) => {
    if (!isRadioInput(element) || !element.name) return true;
    const group = tabbable.filter(
      (other) => isRadioInput(other) && other.name === element.name,
    ) as HTMLInputElement[];
    return element === (group.find((radio) => radio.checked) ?? group[0]);
  });
}

function focusFirstTabbable(root: HTMLElement) {
  (getTabbableElements(root)[0] ?? root).focus();
}

/**
 * Modal focus contract for `ModalBackdrop`:
 * - on open, focus moves to the first tabbable control unless the content
 *   already focused something inside the dialog (autoFocus or its own effect);
 * - Tab / Shift+Tab wrap inside the dialog, and focus that lands behind it
 *   (programmatic focus, Tab from a non-tabbable element) is pulled back in;
 * - on close, focus returns to the element that was focused when it opened,
 *   or into the dialog below when that element is gone.
 *
 * With nested modals only the topmost one acts. Popovers the dialog content
 * portals elsewhere stay usable because their focus and key events still
 * bubble through the dialog in the React tree.
 */
export function useModalFocus({ elevated }: { elevated: boolean }) {
  const rootRef = React.useRef<HTMLDivElement>(null);
  const elevatedRef = React.useRef(elevated);
  elevatedRef.current = elevated;
  // Read during the first render: children's autoFocus and mount effects run
  // before this component's effects and would already have moved focus.
  const [opener] = React.useState(() =>
    typeof document === "undefined" ? null : document.activeElement,
  );
  // Focus events that went through the dialog's onFocus. A set, not just the
  // last one: a focusin can trigger a nested one (a combobox opening moves
  // virtual focus to its option), and marking that must not unmark the outer.
  const focusEventsFromDialogTree = React.useRef(new WeakSet<Event>());

  React.useEffect(() => {
    const root = rootRef.current;
    if (!root) return undefined;
    const modal: OpenModal = { root, elevated: elevatedRef };
    openModals.push(modal);

    if (topmostModalRoot() === root && !root.contains(document.activeElement)) {
      focusFirstTabbable(root);
    }

    // Runs after React dispatched the same focusin through the dialog's
    // onFocus, so an unmarked event means focus landed outside the dialog.
    const handleFocusIn = (event: FocusEvent) => {
      // A closing dialog is already detached here; let the new focus stand.
      if (!root.isConnected || topmostModalRoot() !== root) return;
      if (focusEventsFromDialogTree.current.has(event)) return;
      focusFirstTabbable(root);
    };
    window.addEventListener("focusin", handleFocusIn);

    return () => {
      window.removeEventListener("focusin", handleFocusIn);
      openModals.splice(openModals.indexOf(modal), 1);

      // A real close detaches the dialog before this runs. StrictMode's
      // simulated unmount keeps it attached; restoring focus then would
      // override the content's own autoFocus when the effect re-runs.
      if (root.isConnected) return;

      // Leave focus alone if something outside the dialog deliberately took it.
      const active = document.activeElement;
      const focusWasInDialog =
        !active || active === document.body || root.contains(active);
      if (!focusWasInDialog) return;
      if (
        opener instanceof HTMLElement &&
        opener !== document.body &&
        opener.isConnected
      ) {
        opener.focus();
        return;
      }
      const below = topmostModalRoot();
      if (below?.isConnected) focusFirstTabbable(below);
    };
  }, []);

  const onFocus = React.useCallback((event: React.FocusEvent) => {
    focusEventsFromDialogTree.current.add(event.nativeEvent);
  }, []);

  const onKeyDown = React.useCallback((event: React.KeyboardEvent) => {
    const root = rootRef.current;
    if (
      event.key !== "Tab" ||
      event.defaultPrevented ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      !root ||
      topmostModalRoot() !== root ||
      !root.contains(event.target as Node)
    ) {
      return;
    }
    const tabbable = getTabbableElements(root);
    const first = tabbable[0];
    const last = tabbable.at(-1);
    const active = document.activeElement;
    if (!first || !last) {
      event.preventDefault();
      root.focus();
    } else if (event.shiftKey && (active === first || active === root)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (active === last || active === root)) {
      event.preventDefault();
      first.focus();
    }
  }, []);

  return { rootRef, onFocus, onKeyDown };
}
