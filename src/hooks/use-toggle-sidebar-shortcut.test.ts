import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useSidebarStore } from "#/stores/sidebar-store";
import { useToggleSidebarShortcut } from "./use-toggle-sidebar-shortcut";

const DESKTOP_WIDTH = 1024;
const BELOW_MD_WIDTH = 700;

const CTRL_SHIFT_E: KeyboardEventInit = {
  key: "E",
  code: "KeyE",
  ctrlKey: true,
  shiftKey: true,
};

function pressKey(init: KeyboardEventInit, target: EventTarget = window) {
  const event = new KeyboardEvent("keydown", {
    bubbles: true,
    cancelable: true,
    ...init,
  });
  target.dispatchEvent(event);
  return event;
}

function setViewportWidth(width: number) {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    writable: true,
    value: width,
  });
  window.dispatchEvent(new Event("resize"));
}

function collapsed() {
  return useSidebarStore.getState().collapsed;
}

function openDialog() {
  const dialog = document.createElement("div");
  dialog.setAttribute("role", "dialog");
  dialog.setAttribute("aria-modal", "true");
  document.body.append(dialog);
  return dialog;
}

describe("useToggleSidebarShortcut", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useSidebarStore.setState({ collapsed: false });
    setViewportWidth(DESKTOP_WIDTH);
  });

  it("toggles the sidebar in both directions on Ctrl+Shift+E", () => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey(CTRL_SHIFT_E);
    expect(collapsed()).toBe(true);

    pressKey(CTRL_SHIFT_E);
    expect(collapsed()).toBe(false);
  });

  it("accepts Cmd+Shift+E on apple platforms", () => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey({ ...CTRL_SHIFT_E, ctrlKey: false, metaKey: true });

    expect(collapsed()).toBe(true);
  });

  it("works while the chat composer has focus", () => {
    renderHook(() => useToggleSidebarShortcut());
    const composer = document.createElement("textarea");
    document.body.append(composer);
    composer.focus();

    // Bubbles up from the focused field, so the window listener still sees it.
    pressKey(CTRL_SHIFT_E, composer);

    expect(collapsed()).toBe(true);
    composer.remove();
  });

  it("matches the physical key on a non-Latin layout", () => {
    renderHook(() => useToggleSidebarShortcut());

    // Cyrillic layout: the same key reports "У" while holding KeyE.
    pressKey({ ...CTRL_SHIFT_E, key: "У" });

    expect(collapsed()).toBe(true);
  });

  it("prevents the browser default for the handled combination", () => {
    renderHook(() => useToggleSidebarShortcut());

    const event = pressKey(CTRL_SHIFT_E);

    expect(event.defaultPrevented).toBe(true);
  });

  it("toggles once while the combination is held down", () => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey(CTRL_SHIFT_E);
    // One held-down repeat is enough: an unguarded handler would toggle back.
    pressKey({ ...CTRL_SHIFT_E, repeat: true });

    expect(collapsed()).toBe(true);
  });

  it("ignores the combination while a dialog is open", () => {
    renderHook(() => useToggleSidebarShortcut());
    const dialog = openDialog();

    pressKey(CTRL_SHIFT_E);

    expect(collapsed()).toBe(false);
    dialog.remove();
  });

  it("ignores the combination below the md breakpoint", async () => {
    setViewportWidth(BELOW_MD_WIDTH);
    renderHook(() => useToggleSidebarShortcut());

    pressKey(CTRL_SHIFT_E);
    expect(collapsed()).toBe(false);

    // Widening the window reveals the rail the shortcut belongs to.
    await act(async () => {
      setViewportWidth(DESKTOP_WIDTH);
    });
    pressKey(CTRL_SHIFT_E);
    expect(collapsed()).toBe(true);
  });

  it.each([
    ["a bare Shift+E, as typed in the composer", { key: "E", shiftKey: true }],
    ["Ctrl+E without Shift", { key: "e", code: "KeyE", ctrlKey: true }],
    ["Ctrl+Shift+A", { key: "a", code: "KeyA", ctrlKey: true, shiftKey: true }],
    [
      "Ctrl+Alt+Shift+E",
      { key: "E", code: "KeyE", ctrlKey: true, altKey: true, shiftKey: true },
    ],
  ])("ignores %s", (_label, init) => {
    renderHook(() => useToggleSidebarShortcut());

    pressKey(init);

    expect(collapsed()).toBe(false);
  });

  it("stops listening once unmounted", () => {
    const { unmount } = renderHook(() => useToggleSidebarShortcut());

    unmount();
    pressKey(CTRL_SHIFT_E);

    expect(collapsed()).toBe(false);
  });
});
